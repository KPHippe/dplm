"""Run a frozen ESMFold2 evaluation overlay without modifying the training bundle."""
import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path

from fast_dplm.data import sha256
from fast_dplm.provenance import decoder_provenance
from fast_dplm.refolding import SETTINGS, FoldCache, write_json
from evaluate_fast_samples_esmfold2 import evaluate


def verify_bundle(root):
    manifest = json.loads((root/'bundle.json').read_text())
    for rel, digest in manifest['files'].items():
        path = (root/rel).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError(f'frozen bundle file changed: {rel}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True, help='Evaluation overlay, not training bundle')
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    root = args.bundle.resolve()
    verify_bundle(root)
    cfg = json.loads((root/'config.json').read_text())
    if cfg['fold_settings'] != SETTINGS or args.step not in cfg['steps']:
        raise ValueError('evaluation settings/step differ from frozen plan')
    training = Path(cfg['training_bundle']).resolve()
    if sha256(training/'bundle.json') != cfg['training_bundle_sha256']:
        raise ValueError('training bundle identity changed')
    verify_bundle(training)
    train_cfg = json.loads((training/'config.json').read_text())
    corpus = json.loads((Path(train_cfg['cache'])/'manifest.json').read_text())
    decoder = decoder_provenance(corpus, Path(train_cfg['decoder_lineage']))
    samples = Path(train_cfg['out'])/'evaluation'/f'step_{args.step:07d}'/'samples'
    metadata = json.loads((samples/'manifest.json').read_text())
    panel = json.loads((training/'panel.json').read_text())
    if (metadata['model_step'] != args.step or metadata['fingerprint'] != decoder['fingerprint']
            or metadata['panel'] != panel or len(metadata['samples']) != len(panel['samples'])
            or metadata['decoder_provenance'] != decoder):
        raise ValueError('sample checkpoint/panel/decoder provenance mismatch')
    for item in metadata['samples']:
        if Path(item['file']).name != item['file'] or sha256(samples/item['file']) != item['sha256']:
            raise ValueError('sample content integrity failure')
    out = root/'evaluation'/f'step_{args.step:07d}'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reports = {}
        cache = FoldCache(root/'fold_cache')
        for renderer in ('selected', 'parent'):
            current = decoder if renderer == 'selected' else corpus
            decoded = out/renderer/'decoded'
            script = training/'fast_code/scripts/decode_dplm_samples.py'
            expected = dict(samples_manifest_sha256=sha256(samples/'manifest.json'),
                decoder_lineage_sha256=None if renderer == 'selected' else sha256(train_cfg['decoder_lineage']),
                fingerprint=current['fingerprint'], checkpoint_sha256=current['checkpoint_sha256'],
                decoder_script_sha256=sha256(script))
            if decoded.exists():
                if json.loads((decoded/'provenance.json').read_text()) != expected:
                    raise ValueError('decoded artifact provenance mismatch')
            else:
                env = os.environ.copy()
                env['PYTHONPATH'] = str(training/'fast_code/src')
                command = [cfg['python'], str(script), '--config', current['config'],
                    '--ckpt', current['checkpoint'], '--samples', str(samples), '--out', str(decoded),
                    '--device', cfg['decode_device']]
                if renderer == 'parent':
                    command += ['--decoder-lineage', train_cfg['decoder_lineage']]
                subprocess.run(command, cwd=training/'fast_code', env=env, check=True)
            reports[renderer] = evaluate(decoded, out/renderer/'refold', cache)
        selected, parent = (reports[r]['rows'] for r in ('selected', 'parent'))
        for left, right in zip(selected, parent, strict=True):
            if (left['name'] != right['name'] or left['identity']['sequence'] != right['identity']['sequence']
                    or left['fold_receipt'] != right['fold_receipt']):
                raise ValueError('renderers did not use identical ESMFold2 predictions')
        write_json(out/'paired_report.json', dict(schema='fast-dplm-esmfold2-paired-v1', step=args.step,
            evaluation_bundle_sha256=sha256(root/'bundle.json'),
            training_bundle_sha256=cfg['training_bundle_sha256'],
            samples_manifest_sha256=sha256(samples/'manifest.json'), decode_device=cfg['decode_device'],
            fold_settings=SETTINGS, samples=len(selected), renderers={
                name: dict(report_sha256=sha256(out/name/'refold/report.json'),
                    refolded=report['refolded'], identity_and_backbone_consistent=report['identity_and_backbone_consistent'])
                for name, report in reports.items()}))


if __name__ == '__main__':
    main()
