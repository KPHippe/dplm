"""Strict released-model loader with import-only Python 3.12 compatibility.

Native model/sampling function bodies remain unchanged. The optional pipeline
port permutes vocabulary rows explicitly; native and pipeline layouts differ.
"""
import ast
import importlib
import importlib.machinery
import logging
import sys
import types
from pathlib import Path

import torch


def namespace(name, path=None):
    m = types.ModuleType(name)
    m.__path__ = [str(path)] if path else []
    m.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    sys.modules[name] = m
    if '.' in name:
        parent, key = name.rsplit('.', 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], key, m)
    return m


def unsupported(*args, **kwargs):
    raise RuntimeError('Unused legacy training/tokenizer-loading path disabled in released control')


def extract(path, names, scope):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('native source extraction incomplete')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)


def bootstrap(source):
    import dataclasses
    import math
    import numpy as np
    from transformers import EsmTokenizer, PreTrainedTokenizer
    from transformers.tokenization_utils_base import AddedToken
    for name in ('byprot', 'byprot.models', 'byprot.models.dplm2', 'byprot.models.dplm2.modules',
                 'byprot.datamodules', 'byprot.datamodules.dataset'):
        namespace(name, source/name.replace('.', '/'))
    sys.modules['byprot.models'].register_model = lambda name: (lambda cls: cls)
    utils = namespace('byprot.models.utils')
    utils.__dict__.update(torch=torch, math=math, np=np, dataclass=dataclasses.dataclass,
                          field=dataclasses.field, get_net_dplm2=unsupported, get_struct_tokenizer=unsupported)
    extract(source/'byprot/models/utils.py', ['NetConfig', 'LoRAConfig', 'topk_masking',
        'topk_masking_prior', 'sample_from_categorical', 'stochastic_sample_from_categorical',
        'top_k_top_p_filtering'], utils.__dict__)
    tok = namespace('byprot.datamodules.dataset.tokenized_protein')
    tok.__dict__.update(EsmTokenizer=EsmTokenizer, PreTrainedTokenizer=PreTrainedTokenizer,
                        AddedToken=AddedToken, log=logging.getLogger('native-tokenizer'))
    extract(source/'byprot/datamodules/dataset/tokenized_protein.py',
            ['load_vocab_file', 'DPLM2Tokenizer'], tok.__dict__)
    importlib.import_module('byprot.models.dplm2.modules.dplm2_modeling_esm')
    path = source/'byprot/models/dplm2/dplm2.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    changes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'field':
            continue
        for kw in node.keywords:
            if kw.arg == 'default' and isinstance(kw.value, ast.Call) and isinstance(kw.value.func, ast.Name) and kw.value.func.id.endswith('Config'):
                if kw.value.args or kw.value.keywords:
                    raise ValueError('unexpected dataclass constructor')
                changes.append(dict(line=node.lineno, factory=kw.value.func.id))
                kw.arg, kw.value = 'default_factory', kw.value.func
    ast.fix_missing_locations(tree)
    module = namespace('byprot.models.dplm2.dplm2')
    module.__file__ = str(path)
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module, changes


def vocabulary_permutation(vocab):
    """Pipeline index -> native released vocabulary ID, including all specials."""
    expected = vocab[:33] + [f'{i:04d}' for i in range(8192)] + [
        '<cls_struct>', '<eos_struct>', '<mask_struct>', '<unk_struct>']
    if len(vocab) != 8229 or len(set(vocab)) != len(vocab):
        raise ValueError('unexpected released vocabulary')
    mapping = [vocab.index(token) for token in expected]
    if sorted(mapping) != list(range(8229)):
        raise ValueError('vocabulary remapping is not bijective')
    return mapping


def load(source, checkpoint, with_port=False):
    from transformers import EsmConfig
    from fast_dplm.data import sha256
    module, changes = bootstrap(Path(source))
    from byprot.models.dplm2.modules.dplm2_modeling_esm import EsmForDPLM2
    checkpoint = Path(checkpoint)
    config = EsmConfig.from_pretrained(checkpoint, local_files_only=True)
    net = EsmForDPLM2(config)
    state = torch.load(checkpoint/'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    net.load_state_dict(state, strict=True)
    assert net.state_dict().keys() == state.keys()
    assert all(torch.equal(v, net.state_dict()[k]) for k,v in state.items())
    model = module.MultimodalDiffusionProteinLanguageModel(
        cfg={'tokenizer': {'vocab_file': str(checkpoint)}}, net=net).eval().requires_grad_(False)
    vocab = (checkpoint/'vocab.txt').read_text().splitlines()
    mapping = vocabulary_permutation(vocab)
    assert model.tokenizer.all_tokens == vocab
    port = None
    if with_port:
        from fast_dplm.recipe import DPLM2Recipe
        aa_vocab = [v.replace('_aa>', '>') for v in vocab[:33]]
        port = DPLM2Recipe(config.to_dict(), aa_vocab, levels=[2]*13, lora_rank=0)
        mapped = {k:v for k,v in state.items() if not k.startswith('esm.contact_head.')}
        for name in ('esm.embeddings.word_embeddings.weight', 'lm_head.decoder.weight', 'lm_head.bias'):
            mapped[name] = mapped[name][mapping]
        port.net.load_state_dict(mapped, strict=True)
        assert all(torch.equal(v, port.net.state_dict()[k]) for k,v in mapped.items())
        port.eval().requires_grad_(False)
    proof = dict(strict_load=True, all_state_tensors_equal=True, state_tensors=len(state),
        state_numel=sum(v.numel() for v in state.values()), checkpoint_sha256=sha256(checkpoint/'pytorch_model.bin'),
        config_sha256=sha256(checkpoint/'config.json'), vocab_sha256=sha256(checkpoint/'vocab.txt'),
        compatibility='Unchanged inference/sampling bodies; skip legacy training imports; dataclass default factories',
        dataclass_changes=changes, pipeline_vocabulary_permutation=mapping,
        pipeline_excluded_state=['esm.contact_head.regression.weight', 'esm.contact_head.regression.bias'] if with_port else [])
    return model, port, proof
