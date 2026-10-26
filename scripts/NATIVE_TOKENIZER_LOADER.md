# Native DPLM-2 tokenizer audit loader

`native_tokenizer_loader.py` is an isolated inference adapter for the released
`airkingbd/struct_tokenizer`, revision
`dd9b86fff06889d74ad00d45377a58a17df2d2c2`. It preserves the upstream
`VQModel.tokenize()` and `VQModel.detokenize()` forward functions. It does not
support training or legacy dataset loading. Run it in a fresh process.

The output represents N/CA/C/O/CB using alanine templates; it is not a
full-side-chain tokenizer. The comparison runner saves raw coordinate tensors
and, separately, coordinates processed by the unchanged native PDB writer's
oxygen adjustment. The latter changes O only and cannot repair C–N distances.

## Environment preparation

Start in the DPLM checkout, using its existing dependency overlay. These packages
go into an ignored, isolated target directory; they do not replace Torch, NumPy
or packages in either project's environment.

```bash
mkdir -p outputs/native_tokenizer_audit/deps
uv pip install --python .venv/bin/python \
  --target outputs/native_tokenizer_audit/deps --no-deps \
  fair-esm==2.0.0 ml-collections==1.0.0 dm-tree==0.1.9 \
  biopython==1.85 modelcif==1.8 ihm==2.11 six==1.17.0 absl-py==2.5.0
```

Download the exact native checkpoint and official ESM-IF checkpoint:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import requests
from huggingface_hub import snapshot_download

snapshot_download(
    'airkingbd/struct_tokenizer',
    revision='dd9b86fff06889d74ad00d45377a58a17df2d2c2',
    allow_patterns=['config.yaml', 'dplm2_struct_tokenizer.ckpt'],
)
path = Path('outputs/native_tokenizer_audit/esm_if1_gvp4_t16_142M_UR50.pt')
if not path.exists():
    response = requests.get(
        'https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt',
        stream=True, timeout=120,
    )
    response.raise_for_status()
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as handle:
        for block in response.iter_content(8 * 1024 * 1024):
            handle.write(block)
    temporary.rename(path)
PY
```

The native checkpoint is approximately 473 MB. The official ESM-IF checkpoint
is approximately 1.7 GB, including its training state. Only its exact architecture
arguments are used: the IF decoder is not instantiated, and every persistent
encoder/decoder tensor is supplied by the strict native tokenizer state. The
loader memory maps both checkpoints and allows only `argparse.Namespace` in
addition to Torch's default restricted deserialization types for ESM-IF.

## CPU construction check

```bash
.venv/bin/python scripts/native_tokenizer_loader.py
```

This performs import, model construction and strict state loading without GPU
inference, then checks exact equality of every loaded tensor against the native
checkpoint. The completed check covered **869 tensors / 118,119,848 state
elements**, with no missing or unexpected keys. It writes hashes and compatibility
details to `outputs/native_tokenizer_audit/loader_provenance.json`.

Known content hashes:

- Native checkpoint and official architecture-source hashes are recorded in that
  provenance JSON; inference outputs also embed them.
- Imported native model, ESM and OpenFold sources are hashed. The loader itself
  is hashed independently.

## GPU inference

Submit through Slurm; the comparison runner imports the loader from its sibling
script directory. No additional dependency `PYTHONPATH` is needed.

```python
from native_tokenizer_loader import load_native_tokenizer, adjust_oxygen_pos

model, provenance = load_native_tokenizer()
model = model.cuda().eval()
# Native atom37 order: N=0, CA=1, C=2, CB=3, O=4.
tokens = model.tokenize(xyz, res_mask, seq_length=lengths)
decoded = model.detokenize(tokens, res_mask=res_mask)
```

Use complete contiguous monomers for the current comparison: native detokenize
defaults to contiguous residue indices. `adjust_oxygen_pos` takes one structure
`[L,37,3]`, optionally a known-residue mask, and returns an adjusted copy.

## Compatibility boundary

- Native Python 3.12-invalid dataclass object defaults become `default_factory`
  at import time; no upstream files are edited and no forward functions change.
- ByProt registry auto-import and legacy dataset helpers are disabled to avoid
  loading unrelated training dependencies. Calling disabled helpers raises.
- Removed Biotite `filter_backbone`, direct ESM `torch_scatter` imports,
  optional OpenFold in-place CUDA kernels and fairscale training checkpoint
  wrappers fail closed if called. Their relevant inference paths are unused.
- GVP message passing uses the installed PyG native Torch backend. No scatter
  arithmetic is reimplemented by this loader.
- The native oxygen adjustment function is loaded as its unchanged AST, avoiding
  unrelated mdtraj data utilities. It is only applied as an explicitly separate
  sensitivity arm.
- Auxiliary constructors create the exact official IF encoder and ESM2-8M
  architecture without downloading separate initialization weights. Strict
  checkpoint coverage prevents accidentally retaining random persistent tensors.

The loader intentionally mutates import state only in its dedicated process.
Do not import it into a live training worker or an already-imported legacy
ByProt application.
