"""Bounded GPU check of released DPLM650M/PEFT initialization with FAST."""
import argparse
from pathlib import Path

import numpy as np
import torch

from native_recovery_common import released, build_model, Objective, read, write, sha, to_device
from fast_recovery_common import vocabulary_size
from fast_recovery_sampling import official


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    bundle = read(root / 'bundle.json')
    for rel, digest in bundle['files'].items():
        if sha(root / rel) != digest:
            raise ValueError('frozen probe changed')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = read(root / 'config.json')
    api = released(root)
    model = build_model(api, cfg).cuda()
    size = vocabulary_size(model.tokenizer.all_tokens)
    assert size == cfg['vocabulary_size'] and model.struct_mask_id == size+36
    initialization_check = None
    if cfg.get('matched_native_initial_state'):
        path = Path(cfg['matched_native_initial_state'])
        assert size == 8192 and sha(path) == cfg['matched_native_initial_sha256']
        reference = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        actual = model.state_dict()
        assert actual.keys() == reference.keys(), 'initial parameter/buffer membership differs'
        for name, value in actual.items():
            assert value.dtype == reference[name].dtype and torch.equal(value.cpu(), reference[name]), name
        initialization_check = dict(passed=True, tensors=len(reference),
            native_initial_sha256=cfg['matched_native_initial_sha256'], all_state_tensors_bitwise_equal=True)
        del actual, reference
    criterion = api.criterion_class(label_smoothing=0., ignore_index=model.pad_id)
    objective = Objective(model, criterion)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(.9, .98), weight_decay=.01)
    raw = []
    sequence = 'ACDEFGHIKLMNPQRSTVWY' * 3
    for i in range(8):
        codes = [model.tokenizer.all_tokens[36+(j * 307 + i) % size] for j in range(len(sequence))]
        codes[-1] = model.tokenizer.all_tokens[size+35]
        if i == 0:
            codes[13] = '<pad>'
        elif i == 1:
            codes = ['<pad>'] * len(sequence)
        raw.append(dict(struct_tokens='<cls_struct>' + ''.join(codes) + '<eos_struct>',
                        aatype_tokens='<cls_aa>' + sequence + '<eos_aa>'))
    batch = to_device(api.collater(model.tokenizer)(raw), model.device)
    rows = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, metrics = objective(batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, .5, error_if_nonfinite=True)
        optimizer.step()
        rows.append(dict(step=step+1, loss=float(loss), gradient_norm=float(norm),
                         metrics={k: float(v) for k, v in metrics.items()}))
    torch.cuda.synchronize()
    model.eval()
    probe = dict(aa=np.arange(16, dtype=np.uint8), residue_id=np.arange(1, 17, dtype=np.int32),
                 chain_end=np.arange(16) == 15)
    for conditional in (False, True):
        sample = official(model, probe, conditional, 4, 927)
        assert len(sample['fast_id']) == 16 and (sample['fast_id'] < size).all()
    result = dict(passed=True, synthetic_updates=3, seed=42, tokenizer_length=len(model.tokenizer),
        highest_code_used=True, all_missing_structure_row=True, short_joint_and_conditional_sampling=True,
        trainable_parameters=sum(p.numel() for p in params),
        matched_native_initialization=initialization_check,
        memory_peak_gb=torch.cuda.max_memory_allocated()/1e9, updates=rows,
        bundle_sha256=sha(root / 'bundle.json'),
        scope='Released model/PEFT/criterion smoke with FAST and a structure-only PAD slot. Synthetic data; no training-quality evidence or production-readiness claim.')
    write(root / 'result.json', result)
    print(result)


if __name__ == '__main__':
    main()
