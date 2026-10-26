"""Execute one immutable native-control pipeline stage; failures hold dependents."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'canary', 'train', 'decode', 'refold'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--cell', default='canary')
    a = p.parse_args()
    from fast_dplm.data import sha256
    root = a.root.resolve()
    execution = root/'execution'
    if sha256(execution/'manifest.json') != a.execution_sha256:
        raise ValueError('stage execution manifest changed')
    for relative, expected in json.loads((execution/'manifest.json').read_text())['files'].items():
        path = execution/relative
        if not path.resolve().is_relative_to(execution) or sha256(path) != expected:
            raise ValueError('stage execution file changed')
    settings = json.loads((execution/'settings.json').read_text())
    python = settings['python']
    env = dict(os.environ, PYTHONPATH=f'{execution}/dplm_src:{root}/runtime/fast/src',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1',
               PYTORCH_ALLOC_CONF='expandable_segments:True')
    def run(command):
        print(json.dumps(dict(stage=a.stage, command=command)), flush=True)
        subprocess.run(command, env=env, check=True)
    if a.stage == 'prepare':
        run([python, str(root/'runtime/dplm/scripts/native_control_corpus.py'), 'finalize', '--root', str(root)])
        run([python, str(execution/'prepare_native_control.py'), '--root', str(root)])
        return
    bundle = root/'training'/a.cell
    env['PYTHONPATH'] = f'{bundle}/code/src:{root}/runtime/fast/src'
    train = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
             str(bundle/'code/run_fast_bundle.py'), '--bundle', str(bundle)]
    if a.stage == 'canary':
        run(train+['--stop-after', '10'])
        run(train+['--resume'])
        run([python, str(bundle/'code/verify_dplm2_recipe_checkpoint.py'), '--bundle', str(bundle)])
        run([python, str(bundle/'code/native_control_evaluate.py'), 'decode', '--root', str(root),
             '--bundle', str(bundle), '--steps', '20'])
        proof = json.loads((bundle/'verification.json').read_text())
        if not (proof['strict_reload'] and proof['finite_training'] and proof['global_batch'] == 192
                and proof['world_size'] == 8 and proof['frozen_original_dplm_elements'] == 650671360):
            raise ValueError('native production canary did not pass')
        native_rows = [json.loads(line) for line in (bundle/'run/metrics.jsonl').read_text().splitlines()]
        baseline = Path(settings['fast_screen'])/'control_seed42/run/metrics.jsonl'
        with baseline.open() as handle:
            original_rows = [json.loads(next(handle)) for _ in native_rows]
        for native, original in zip(native_rows, original_rows, strict=True):
            for key in ('step', 'lr', 'exposure', 'valid_residues', 'paired_tokens', 'task_counts', 'objective'):
                if native[key] != original[key]:
                    raise ValueError(f'native canary changes matched FAST data/noising trace: {key}')
        (root/'canary_complete.json').write_text(json.dumps(dict(passed=True,
            matched_fast_data_noising_trace=True,
            checkpoint_verification_sha256=sha256(bundle/'verification.json'),
            geometry_complete_sha256=sha256(root/'evaluation/canary/step_0000020/decode_complete.json')),
            indent=2)+'\n')
    elif a.stage == 'train':
        if not json.loads((root/'canary_complete.json').read_text())['passed']:
            raise ValueError('production requires passing native canary')
        run(train)
    else:
        run([python, str(bundle/'code/native_control_evaluate.py'), a.stage, '--root', str(root), '--bundle', str(bundle)])


if __name__ == '__main__':
    main()
