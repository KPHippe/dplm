"""Freeze an opt-in generation-validation panel without changing historical bundles."""
import argparse
import json
from pathlib import Path

from fast_dplm.data import TokenCache
from fast_dplm.evaluation import make_panel
from fast_dplm.generation_validation import make_generation_panel, validate_generation_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--val-cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--exclude-panel', type=Path, action='append', default=[])
    parser.add_argument('--per-source-bin', type=int, default=4)
    parser.add_argument('--seed', type=int, default=30930)
    parser.add_argument('--sampling-steps', type=int, default=100)
    args = parser.parse_args()
    cache = TokenCache(args.val_cache)
    excluded = []
    for path in args.exclude_panel:
        previous = json.loads(path.read_text())
        entries = previous.get('conditional_entries', previous.get('entries', []))
        excluded += [(e['source'], e['entry_id']) for e in entries]
    panel = make_panel(cache, lengths=(100, 200, 300, 400, 500), sampling_steps=args.sampling_steps)
    panel['generation_validation'] = make_generation_panel(cache, per_source_bin=args.per_source_bin,
        seed=args.seed, sampling_steps=args.sampling_steps, joint_samples=panel['samples'], exclude_identities=excluded)
    validate_generation_panel(cache, panel['generation_validation'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        handle.write(json.dumps(panel, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
