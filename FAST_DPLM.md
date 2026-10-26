# FAST–DPLM adapter

**Current state (2026-10-26):** matched FAST8k and recovered native8k generator
runs have both completed 100k updates. The active implementation is
`scripts/train_fast_recovery.py` / `scripts/train_native_recovery.py`, using the
released DPLM model/objective/sampler with rank16 LoRA and self-mixup disabled.
The completed FAST candidate is
`../fast-rewrite/outputs/dplm/fast_vocab_recovery_swap_20261021_v1/free8k/candidates/v1/`;
the native comparator is `../fast-rewrite/outputs/dplm/native_recovery_v1_20261008/candidates/v2/`.
See [the current recipe and results](../fast-rewrite/docs/FAST_DPLM_RECIPE.md).
Historical pilot/adapter descriptions below remain for provenance and do not
describe the final recovered training pipeline. FAST8k has not surpassed the
native baseline or cleared full-atom generation gates.

For migration, preserve both repositories and the full project tree at
`/hpc/projects/group.chi/aiml/data/FAST/`. The handoff directory
`outputs/handoff_20261026/` under that project root contains portable Git
bundles, the three required Hugging Face model caches, paper copies and checksums.
Both checkouts' `outputs` paths are external symlinks. Frozen configurations and
checkpoints contain absolute paths: restore compatibility paths or prepare a
separate relocated bundle, preserving the original evidence. Recreate virtual
environments; do not reuse an old `.venv` as a portable installation.

**Historical pilot preparation:** a full-corpus 5k pilot was frozen and unsubmitted. See
[the pilot review](../fast-rewrite/docs/FAST_DPLM_PILOT.md) for the launch bundle,
eight-GPU canary, periodic evaluation and measured decoder chemistry limitation.
`train_fast_dplm.py` remains a bounded debug wrapper; `train_fast_production.py`
uses the shared `src/fast_dplm/training.py` loop with frozen run settings. The
historical debug-only scope below is superseded where the pilot review says so.

Implemented 2026-09-25 for **bounded infrastructure validation**. Larger training
requires the user's review. `train_fast_dplm.py` accepts total endpoints of at most
500 updates; this limit is not a production training recipe.

The adapter lives in `src/fast_dplm/`, separately from legacy ByProt imports. It
uses the pretrained DPLM ESM trunk, SDPA with DPLM query scaling, and paired rotary
positions. Both tracks have explicit BOS/EOS/padding/mask states. Structure inputs
use either a 4096-entry embedding/head (`--mode index`) or four 8-class
embeddings/heads (`--mode components`, default). Tuple embeddings are summed;
component cross-entropies are summed per residue. The pretrained AA embedding and
head retain their 33-class vocabulary, while generation permits only 20 AA classes.
The unused contact head is removed and unused absolute-position weights are frozen.

The loss uses explicit residue/mask booleans, linear timestep weights, and four
equally sampled tasks: joint generation, structure prediction given sequence,
AA reconstruction given FAST tokens, and independently corrupted tracks with
cross-track attention blocked. The third task is **not a clean inverse-folding
benchmark**: FAST encoding includes AA identity. DDP/accumulation normalize each
modality by the global number of masked residues. The two modality losses have
equal weight; tuple CE is not divided by four.

Sampling uses separate modality remasking, linear mask-fraction decay, categorical
temperature annealing 1.1 to 0.1, and Gumbel-perturbed confidence ranking. Accepted
tokens retain their own confidence until remasked. This is an adaptation of the
upstream unconditioned reparameterized sampler, not an exact reproduction of its
bit tokenizer or top-p filtering. The current adapter uses no top-p truncation.

## Environment

Run from this repository:

```bash
bash scripts/setup_fast_environment.sh
```

This creates a separate `.venv` and a `.pth` link to FAST's installed packages,
then installs the five pinned adapter dependencies into the DPLM environment.
It reuses FAST's torch/CUDA build read-only. It does not change FAST's environment
and is not a standalone portable lockfile: keep the referenced FAST runtime intact.
Tests importing FAST also need its `src` directory in `PYTHONPATH`.

The verified sequence checkpoint is `airkingbd/dplm_650m`, revision
`7a7e651baa667d094aba05e9dc1cf52a3332110a`. The Slurm launcher points to its local
Hugging Face snapshot. To provision another machine, use `snapshot_download`
from the installed `huggingface_hub`, with that exact `repo_id` and `revision`.
No model files or datasets belong in git.

## Test and debug

```bash
env PYTHONPATH="$PWD/src:/hpc/mydata/kyle.hippe/github/fast-rewrite/src" \
    OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_fast_dplm.py

# Check existing Slurm jobs first; choose a genuinely new output directory.
sbatch scripts/submit_fast_debug.sbatch \
    --out /hpc/mydata/kyle.hippe/github/fast-rewrite/outputs/dplm/my_debug \
    --mode components --steps 20 --batch-size 2 --checkpoint-every 10
```

`--steps` is the total endpoint, not an additional number of updates. Explicit
`--resume` reads only the chosen output's `last.ckpt`. It preserves the original
schedule horizon, optimizer, rank RNG, data sequence, source exposure, cache
digests and model/code hashes. Output directories have an exclusive writer lock.
Changing world size, batch, seed, data, code, weights or LR schedule is rejected.
This implements semantic continuation; bitwise GPU replay is not promised.

The default launcher uses the prepared 32-entry training and 8-entry validation
caches under FAST's `outputs/dplm/debug_v1/`. Override `--cache` explicitly for
another cache. Each rank derives the same global cluster-balanced sample list
from `(data_seed=1234, update)` and takes its assigned microbatches. Bucketing
sorts only that list, preserving its sample multiset. Source draws are calibrated
from the cache's exact cluster/member mean lengths for equal residue shares.

Training logs JSONL loss, source exposure, LR, gradient norm, update time and the
maximum allocated memory over ranks. Full-state checkpoints are atomic. The
bounded trainer validates four fixed entries and generates two samples at its
endpoint; it does not yet implement production periodic evaluation or W&B.
Its tiny validation loss is a weighted masked denoising diagnostic, not perplexity
or a generation-quality score.

## Generation exchange

Run GPU commands inside Slurm, from this repository with `PYTHONPATH=src`:

```bash
.venv/bin/python generate_fast_dplm.py --checkpoint /path/to/run/last.ckpt \
    --out /path/to/new/samples --length 100 --count 2 --steps 100 --seed 919
```

No FAST data or native structure is needed for generation. The resulting manifest
and numeric NPZ files contain FAST IDs, AA indices in FAST order, residue numbers
and true chain-end flags. The manifest binds the FAST fingerprint, levels, LM
checkpoint hash, step, sampler seed and steps. Decode in FAST's environment using
`scripts/decode_dplm_samples.py`; see the [FAST runbook](../fast-rewrite/docs/FAST_DPLM_DEBUG.md).

The generated DPLM sequence is never supplied to the frozen FAST decoder. PDB
residue names, atoms and coordinates come from FAST's own predictions. Both
sequences and their disagreement are exported, with the actual `.fastc` file size.
This is monomer heavy-atom co-generation infrastructure, not validated design
quality, motif completion, all-atom chemistry including hydrogens, or docking.

## Remaining production work

Freeze/audit the FAST parent and eligible full corpus; build versioned resumable
corpus shards and aggregate sampling calibration. Extend the bounded trainer with
periodic validation, fixed generated panels, retained milestone weights, complete
launch snapshots and production restart policy. Add independent refolding,
corrected chemistry/packing evaluation and the matched LFQ/packing controls.
Review the pilot's corpus, horizon, schedule and acceptance criteria with the user
before removing the debug limit or submitting a larger run.
# ESMFold2 evaluator (2026-09-29)

The requested ESMFold-v1 replacement uses the hosted `esmfold2-2026-05` full
model with sequence-only input. `scripts/prepare_fast_esmfold2.py` freezes a
separate evaluator overlay referencing the unchanged training bundle.
`scripts/submit_fast_esmfold2.sbatch` decodes both FAST renderers on CPU and
shares one cached API fold per generated sequence/settings. No new environment
dependencies are required. See adjacent FAST
`docs/FAST_DPLM_ESMFOLD2.md` for the protocol, validation and live-launch record.
Run `tests/test_fast_refolding.py` when changing this evaluation contract.
