"""Run the frozen target-free decoder and independent folding panel for one milestone."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from fast_dplm.data import sha256
from fast_dplm.provenance import decoder_provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--renderer', choices=['selected', 'parent'], default='selected')
    a = p.parse_args()
    root = a.bundle.resolve()
    manifest = json.loads((root/'bundle.json').read_text())
    for rel, digest in manifest['files'].items():
        if sha256(root/rel) != digest:
            raise ValueError(f'frozen file changed: {rel}')
    cfg = json.loads((root/'config.json').read_text())
    cache = json.loads((Path(cfg['cache'])/'manifest.json').read_text())
    decoder = decoder_provenance(cache, Path(cfg['decoder_lineage']) if cfg.get('decoder_lineage') else None)
    samples = Path(cfg['out'])/'evaluation'/f'step_{a.step:07d}'/'samples'
    meta = json.loads((samples/'manifest.json').read_text())
    if meta['model_step'] != a.step or meta['fingerprint'] != decoder['fingerprint']:
        raise ValueError('sample/checkpoint provenance mismatch')
    out = root/'evaluation'/f'step_{a.step:07d}'
    if a.renderer == 'parent':
        if not cfg.get('decoder_lineage'):
            raise ValueError('parent renderer requires verified decoder lineage')
        out = out/'parent_renderer'
    out.mkdir(parents=True, exist_ok=True)
    fast = Path('/hpc/mydata/kyle.hippe/github/fast-rewrite')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root/'fast_code/src')
    decoded = out/'decoded'
    if decoded.exists():
        saved = json.loads((decoded/'provenance.json').read_text())
        if saved['samples_manifest_sha256'] != sha256(samples/'manifest.json'):
            raise ValueError('decoded samples changed')
    else:
        decode_args = []
        if a.renderer == 'parent':
            decoder = cache
            decode_args = ['--decoder-lineage', cfg['decoder_lineage']]
        subprocess.run([str(fast/'.venv/bin/python'), str(root/'fast_code/scripts/decode_dplm_samples.py'),
            '--config', decoder['config'], '--ckpt', decoder['checkpoint'], '--samples', str(samples),
            '--out', str(decoded), *decode_args], cwd=root/'fast_code', env=env, check=True)
    env['PYTHONPATH'] = str(root/'code/src')+os.pathsep+str(root/'fast_code/src')
    subprocess.run([sys.executable, str(root/'code/evaluate_fast_samples.py'), '--decoded', str(decoded),
        '--out', str(out/'refold'), '--pretrained',
        '/hpc/mydata/kyle.hippe/.cache/huggingface/hub/models--facebook--esmfold_v1/snapshots/75a3841ee059df2bf4d56688166c8fb459ddd97a'], env=env, check=True)


if __name__ == '__main__':
    main()
