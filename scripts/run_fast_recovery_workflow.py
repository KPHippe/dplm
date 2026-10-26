"""Gated preparation, canary and submission for the authorized FAST swap.

Every stage fails closed; an incomplete coordinate/token corpus cannot start
training. Successful submissions are recorded immediately to prevent duplicates.
"""
import argparse
import fcntl
import json
import subprocess
from pathlib import Path

from native_recovery_common import read, write, sha


def run(command, env=None):
    subprocess.run([str(x) for x in command], check=True, env=env)


def verify_completed_cache(folder, expected):
    """Reuse a committed cache only under pinned audit, manifests and payloads."""
    if sha(folder / 'audit.json') != expected['audit_sha256']:
        raise ValueError('completed cache audit changed')
    audit = read(folder / 'audit.json')
    if not audit['passed'] or not audit['exact_native_row_and_aa_alignment']:
        raise ValueError('completed cache did not pass exact correspondence')
    if set(expected['manifest_sha256']) != {'train', 'valid'}:
        raise ValueError('both completed cache splits must be pinned')
    for split, digest in expected['manifest_sha256'].items():
        manifest = folder / split / 'manifest.json'
        if sha(manifest) != digest:
            raise ValueError('completed cache manifest changed')
        for name, file_digest in read(manifest)['files'].items():
            if sha(folder / split / name) != file_digest:
                raise ValueError('completed cache payload changed')


def main():
    import os
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'canary', 'gate'])
    p.add_argument('--workflow', type=Path, required=True)
    a = p.parse_args()
    workflow = a.workflow.resolve()
    for name, digest in read(workflow / 'bundle.json')['files'].items():
        if sha(workflow / name) != digest:
            raise ValueError('workflow bundle changed')
    plan = read(workflow / 'plan.json')
    for path, digest in plan['proofs'].items():
        if sha(path) != digest or not read(path)['passed']:
            raise ValueError('preparation proof changed or failed')
    outer, native = Path(plan['root']), Path(plan['native'])
    candidate_name = plan.get('candidate', 'v1')
    candidate = outer / 'candidates' / candidate_name
    if a.stage == 'prepare':
        encoder_stage = Path(plan.get('encoder_stage', outer / 'preparation/encoding_v4'))
        encoding = Path(plan.get('encoding_directory', outer / 'encoding_shards_v4'))
        if plan.get('completed_cache'):
            verify_completed_cache(outer / 'data', plan['completed_cache'])
        elif plan.get('encoding_repair_directory'):
            run([plan['fast_python'], workflow / 'code/fast-rewrite/scripts/merge_fast_recovery_encoding.py',
                 '--original', outer/'encoding_shards_v4', '--repairs', plan['encoding_repair_directory'],
                 '--out', encoding])
        finalize = [plan['fast_python'], workflow / 'code/fast-rewrite/scripts/finalize_fast_recovery_cache.py',
            '--native', native, '--encoding', encoding, '--out', outer / 'data',
            '--lineage', encoder_stage / 'decoder_lineage.json',
            '--encoder-bundle', encoder_stage / 'bundle.json']
        if plan.get('reuse_encoder_bundle'):
            finalize += ['--reuse-encoder-bundle', plan['reuse_encoder_bundle']]
        if not plan.get('completed_cache'):
            run(finalize)
        run([plan['dplm_python'], workflow / 'code/dplm/scripts/prepare_fast_recovery.py',
            '--root', outer, '--native', native, '--candidate', candidate_name,
            '--fast-python', plan['fast_python'], '--encoder-stage', encoder_stage])
        digest = sha(candidate / 'bundle.json')
        run(['bash', candidate / 'submit.sbatch', candidate, digest, 'cpu'])
        write(outer / 'candidate.json', dict(root=str(candidate), bundle_sha256=digest,
            workflow_sha256=sha(workflow / 'bundle.json')))
        return
    prepared = read(outer / 'candidate.json')
    if prepared['root'] != str(candidate) or prepared['bundle_sha256'] != sha(candidate / 'bundle.json'):
        raise ValueError('prepared candidate changed')
    digest = prepared['bundle_sha256']
    if a.stage == 'canary':
        run(['bash', candidate / 'submit.sbatch', candidate, digest, 'canary'])
        run(['bash', candidate / 'submit.sbatch', candidate, digest, 'renderer'])
        return
    run(['bash', candidate / 'submit.sbatch', candidate, digest, 'canary_refold'])
    env = os.environ.copy()
    env['PYTHONPATH'] = ':'.join(str(candidate / x) for x in ('dependencies', 'code/src', 'code/fast_src', 'code'))
    run([plan['dplm_python'], candidate / 'code/verify_native_recovery_ready.py',
        '--root', candidate, '--bundle-sha256', digest], env=env)
    result = read(candidate / 'verification.json')
    if not result['production_ready'] or result['bundle_sha256'] != digest:
        raise ValueError('production gates did not pass')
    with (outer / '.launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        launch = read(outer / 'launch.json')
        if launch.get('training_job'):
            raise RuntimeError('training already submitted; refusing duplicate')
        def submit(stage, extra, resources):
            command = ['sbatch', '--parsable', *resources, str(candidate / 'submit.sbatch'),
                       str(candidate), digest, stage, *extra]
            response = subprocess.check_output(command, text=True).strip()
            job = response.split(';')[0]
            if not job.isdecimal():
                raise RuntimeError('unrecognized sbatch response; inspect scheduler before retrying')
            return int(job)
        train = submit('train', [], ['--partition=gpu', '--gpus=h200:8', '--cpus-per-task=32',
                                   '--mem=512G', '--time=3-00:00:00'])
        launch.update(training_job=train, production_submitted=True, candidate=prepared,
            production_verification_sha256=sha(candidate / 'verification.json'),
            status='production submitted after complete cache and all infrastructure gates passed')
        write(outer / 'launch.json', launch)
        try:
            watcher = submit('watch', [str(train)], ['--partition=cpu', '--cpus-per-task=4',
                                                   '--mem=16G', '--time=4-00:00:00'])
            launch['evaluation_watcher_job'] = watcher
        except Exception as exc:
            launch['watcher_submission_error'] = str(exc)
            write(outer / 'launch.json', launch)
            raise
        write(outer / 'launch.json', launch)
        print(json.dumps(dict(training_job=train, evaluation_watcher_job=watcher)), flush=True)


if __name__ == '__main__':
    main()
