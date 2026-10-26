"""Isolated, audited inference loader for the released DPLM-2 tokenizer.

The model forward functions come from the checkout unchanged. Compatibility
adapters only avoid legacy training imports, replace Python 3.12-invalid mutable
dataclass defaults, and satisfy optional extension imports. No training is
supported. All persistent model tensors must be covered by the released state.
Run this in a fresh process, not one that has already imported legacy ByProt.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "outputs/native_tokenizer_audit"
REVISION = "dd9b86fff06889d74ad00d45377a58a17df2d2c2"
COMPATIBILITY: list[dict] = []


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _namespace(name, path=None):
    if name in sys.modules:
        raise RuntimeError(f"Loader requires fresh process; module exists: {name}")
    m = types.ModuleType(name)
    m.__path__ = [] if path is None else [str(path)]
    m.__package__ = name
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = m
    if "." in name:
        parent, child = name.rsplit(".", 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], child, m)
    return m


def _unsupported(*args, **kwargs):
    raise RuntimeError("Legacy training/data/optional CUDA path disabled in audit loader")


class _DataclassLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path).decode("utf-8")
        tree = ast.parse(source, filename=self.path)
        changes = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.AnnAssign):
                continue
            value = node.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id.endswith("Config") and not value.args and not value.keywords):
                changes.append({"line": node.lineno, "factory": value.func.id})
                node.value = ast.Call(func=ast.Name(id="_audit_dataclass_field", ctx=ast.Load()),
                    args=[], keywords=[ast.keyword(arg="default_factory", value=value.func)])
        if changes:
            tree.body.insert(0, ast.ImportFrom(module="dataclasses", names=[
                ast.alias(name="field", asname="_audit_dataclass_field")], level=0))
            ast.fix_missing_locations(tree)
            COMPATIBILITY.append({"path": self.path, "sha256": _sha(self.path),
                                  "change": "mutable dataclass defaults to default_factory", "fields": changes})
        return compile(tree, self.path, "exec")


class _DataclassFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        names = {
            "byprot.models.structok.modules.folding_utils.trunk",
            "byprot.models.structok.modules.folding_utils.decoder",
            "byprot.models.structok.modules.folding_utils.esmfold",
            "esm.esmfold.v1.trunk", "esm.esmfold.v1.esmfold",
        }
        if fullname not in names:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = _DataclassLoader(fullname, spec.origin)
        return spec


def _bootstrap():
    deps = WORK / "deps"
    if not (deps / "esm").exists():
        raise RuntimeError("Install isolated fair-esm==2.0.0, ml-collections==1.0.0, dm-tree==0.1.9 under " + str(deps))
    sys.path.insert(0, str(deps))
    sys.path.insert(0, str(ROOT / "vendor/openfold"))
    import torch
    # Import PyG before its optional torch_scatter name is provided. Its installed
    # native torch backend remains selected; no unavailable compiled package is used.
    import torch_geometric  # noqa: F401
    scatter = types.ModuleType("torch_scatter")
    scatter.__spec__ = importlib.machinery.ModuleSpec("torch_scatter", loader=None)

    scatter.scatter_add = scatter.scatter = _unsupported
    sys.modules["torch_scatter"] = scatter
    COMPATIBILITY.append({"change": "unused ESM direct torch_scatter imports fail closed; actual GVP MessagePassing uses installed PyG native torch backend"})
    import biotite.structure
    if not hasattr(biotite.structure, "filter_backbone"):
        biotite.structure.filter_backbone = _unsupported
        COMPATIBILITY.append({"change": "unused removed Biotite filter_backbone legacy file loader fails closed"})
    kernel = types.ModuleType("attn_core_inplace_cuda")
    kernel.__spec__ = importlib.machinery.ModuleSpec("attn_core_inplace_cuda", loader=None)
    kernel.forward_ = kernel.backward_ = _unsupported
    sys.modules[kernel.__name__] = kernel
    COMPATIBILITY.append({"change": "unused optional OpenFold inplace CUDA extension fails closed if invoked"})
    _namespace("fairscale")
    _namespace("fairscale.nn")
    fair_checkpoint = _namespace("fairscale.nn.checkpoint")
    fair_checkpoint.checkpoint_wrapper = _unsupported
    COMPATIBILITY.append({"change": "unused fairscale training checkpoint wrapper fails closed"})
    for name in ("byprot", "byprot.models", "byprot.models.structok", "byprot.models.structok.modules",
                 "byprot.models.structok.modules.folding_utils", "byprot.utils", "byprot.datamodules",
                 "byprot.datamodules.pdb_dataset"):
        _namespace(name, ROOT / "src" / name.replace(".", "/"))
    sys.modules["byprot.models"].register_model = lambda name: (lambda cls: cls)
    data = _namespace("byprot.datamodules.pdb_dataset.pdb_datamodule")
    data.PdbDataset = type("PdbDataset", (), {"process_chain": staticmethod(_unsupported)})
    for name in ("aatype_to_seq", "collate_fn", "seq_to_aatype", "struct_ids_to_seq", "struct_seq_to_ids"):
        setattr(data, name, _unsupported)
    # Only the native, standalone oxygen adjustment is used by the PDB writer.
    # Extracting this exact AST avoids importing unrelated mdtraj data loaders.
    source = ROOT / "src/byprot/datamodules/pdb_dataset/all_atom.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "adjust_oxygen_pos")
    atom = _namespace("byprot.datamodules.pdb_dataset.all_atom")
    atom.torch = torch
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), atom.__dict__)
    COMPATIBILITY.append({"change": "load unchanged standalone adjust_oxygen_pos AST without unrelated data loaders",
                          "path": str(source), "sha256": _sha(source)})
    COMPATIBILITY.append({"change": "skip registry auto-import and disable unused legacy dataset helpers"})
    sys.meta_path.insert(0, _DataclassFinder())


def load_native_tokenizer():
    """Return (native VQModel on CPU in eval mode, full provenance dictionary)."""
    _bootstrap()
    import torch
    import esm
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf
    from esm.inverse_folding.gvp_transformer import GVPTransformerModel
    from esm.model.esm2 import ESM2

    checkpoint_dir = Path(snapshot_download("airkingbd/struct_tokenizer", revision=REVISION,
        allow_patterns=["config.yaml", "dplm2_struct_tokenizer.ckpt"], local_files_only=True))
    if_path = WORK / "esm_if1_gvp4_t16_142M_UR50.pt"
    if not if_path.is_file():
        raise FileNotFoundError(f"Official ESM-IF checkpoint required for its exact architecture arguments: {if_path}")
    with torch.serialization.safe_globals([argparse.Namespace]):
        if_state = torch.load(if_path, map_location="cpu", weights_only=True, mmap=True)
    args = if_state["args"]
    # The IF decoder is discarded by native GVP wrapper. Instantiate exactly its
    # encoder with official arguments; every resulting persistent weight is then
    # supplied by the strict native tokenizer state below.
    def uninitialized_if():
        alphabet = esm.Alphabet.from_architecture(args.arch)
        emb = GVPTransformerModel.build_embedding(args, alphabet, args.encoder_embed_dim)
        encoder = GVPTransformerModel.build_encoder(args, alphabet, emb)
        return types.SimpleNamespace(encoder=encoder), alphabet

    def uninitialized_esm(name):
        if name != "esm2_t6_8M_UR50D":
            raise ValueError(f"Unexpected auxiliary ESM model: {name}")
        alphabet = esm.Alphabet.from_architecture("ESM-1b")
        return ESM2(num_layers=6, embed_dim=320, attention_heads=20,
                    alphabet=alphabet, token_dropout=True), alphabet

    with patch.object(esm.pretrained, "esm_if1_gvp4_t16_142M_UR50", uninitialized_if), \
         patch.object(esm.pretrained, "load_model_and_alphabet", uninitialized_esm):
        module = importlib.import_module("byprot.models.structok.structok_lfq")
        model = module.VQModel(**OmegaConf.load(checkpoint_dir / "config.yaml"))
    ckpt = checkpoint_dir / "dplm2_struct_tokenizer.ckpt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    # Explicit coverage check supplements strict load and records the full count.
    live = model.state_dict()
    if live.keys() != state.keys():
        raise AssertionError("Checkpoint tensor key coverage differs")
    mismatches = [k for k in state if not torch.equal(live[k], state[k])]
    if mismatches:
        raise AssertionError(f"Loaded tensors differ: {mismatches}")
    used_sources = {}
    roots = (ROOT / "src", ROOT / "vendor/openfold", WORK / "deps/esm")
    for name, imported in list(sys.modules.items()):
        file = getattr(imported, "__file__", None)
        if file and Path(file).is_file() and any(Path(file).is_relative_to(root) for root in roots):
            used_sources[str(file)] = _sha(file)
    provenance = {
        "repo": "airkingbd/struct_tokenizer", "revision": REVISION,
        "checkpoint": str(ckpt), "checkpoint_sha256": _sha(ckpt),
        "config": str(checkpoint_dir / "config.yaml"), "config_sha256": _sha(checkpoint_dir / "config.yaml"),
        "auxiliary_architecture_source": str(if_path), "auxiliary_sha256": _sha(if_path),
        "encoder_args": vars(args),
        "strict_load": True, "missing_keys": [], "unexpected_keys": [],
        "loaded_state_tensors": len(state), "loaded_state_numel": sum(t.numel() for t in state.values()),
        "all_loaded_tensors_equal_checkpoint": True,
        "auxiliary_constructor": "native IF encoder with official arguments and ESM2-8M constructed uninitialized; all weights strictly overwritten from native tokenizer checkpoint",
        "compatibility": COMPATIBILITY.copy(), "source_sha256": used_sources,
        "loader_sha256": _sha(__file__), "torch": torch.__version__,
        "representation": "native tokenize/detokenize; raw five-atom alanine templates; no relaxation",
    }
    return model, provenance


def native_adjust_oxygen_pos(atom37, position_known=None):
    """Run unchanged native PDB-writer oxygen adjustment, on a copy of input."""
    from byprot.datamodules.pdb_dataset.all_atom import adjust_oxygen_pos
    return adjust_oxygen_pos(atom37.clone(), position_known)


adjust_oxygen_pos = native_adjust_oxygen_pos


if __name__ == "__main__":
    torch_threads = 2
    import torch
    torch.set_num_threads(torch_threads)
    model, provenance = load_native_tokenizer()
    out = WORK / "loader_provenance.json"
    out.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"strict_load": True, "state_tensors": provenance["loaded_state_tensors"],
                      "state_numel": provenance["loaded_state_numel"], "provenance": str(out)}))
