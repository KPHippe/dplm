"""Verify a prepared launch bundle before entering the trainer on every rank."""
import argparse
import json
import os
import sys
from pathlib import Path

from fast_dplm.data import sha256
from fast_dplm.training import main


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int)
    a = p.parse_args()
    bundle = json.loads((a.bundle / 'bundle.json').read_text())
    for rel, digest in bundle['files'].items():
        if sha256(a.bundle / rel) != digest:
            raise ValueError(f'frozen bundle changed: {rel}')
    if int(os.environ.get('WORLD_SIZE', '1')) != bundle['world_size']:
        raise ValueError('world size differs from frozen bundle')
    sys.argv = [sys.argv[0], '--run-config', str(a.bundle / 'config.json')]
    if a.resume:
        sys.argv.append('--resume')
    if a.stop_after is not None:
        sys.argv.extend(['--stop-after', str(a.stop_after)])
    main(debug=bundle['kind'] == 'canary')
