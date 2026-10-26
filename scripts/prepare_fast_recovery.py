"""Freeze a FAST swap of the successful native recovery, after exact cache completion."""
import argparse
import math
import os
import shutil
from pathlib import Path

from native_recovery_common import read, write, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--fast-python', type=Path, required=True,
                   help='FAST virtualenv Python; independent of run-storage location')
    p.add_argument('--encoder-stage', type=Path,
                   help='Frozen FAST encoder bundle supplying decoder lineage and model sources')
    a = p.parse_args()
    # Preserve the virtualenv symlink: resolving bin/python itself would invoke
    # the base interpreter without the FAST environment.
    fast_python = a.fast_python.absolute()
    if not fast_python.is_file() or not os.access(fast_python, os.X_OK):
        raise ValueError('FAST Python executable is unavailable')
    outer, native = a.root.resolve(), a.native.resolve()
    original = native / 'candidates/v2'
    audit = read(outer / 'data/audit.json')
    if not audit['passed'] or not audit['exact_native_row_and_aa_alignment']:
        raise ValueError('exact corpus alignment required before bundle preparation')
    for split in ('train', 'valid'):
        new, old = read(outer / 'data' / split / 'manifest.json'), read(native / 'data' / split / 'manifest.json')
        if new['native_manifest_sha256'] != sha(native / 'data' / split / 'manifest.json'):
            raise ValueError('wrong native comparison cache')
        for name in ('aa.npy', 'offsets.npy', 'rows.jsonl'):
            if new['files'][name] != old['files'][name] or sha(outer / 'data' / split / name) != old['files'][name]:
                raise ValueError('AA, ordering or lengths changed')
    root = outer / 'candidates' / a.candidate
    root.mkdir(parents=True, exist_ok=False)
    (root / 'data').symlink_to(outer / 'data', target_is_directory=True)
    for name in ('references', 'dependencies'):
        (root / name).symlink_to(native / name, target_is_directory=True)
    for name in ('panel.json', 'reference_audit.json'):
        shutil.copy2(native / name, root / name)
    write(root / 'data_audit.json', audit)
    encoder_stage = (a.encoder_stage.resolve() if a.encoder_stage else outer / 'preparation/encoding_v2')
    for name, digest in read(encoder_stage/'bundle.json')['files'].items():
        if sha(encoder_stage/name) != digest:
            raise ValueError('frozen encoder stage changed')
    lineage = encoder_stage / 'decoder_lineage.json'
    lineage_record = read(lineage)
    vocabulary_size = math.prod(lineage_record['levels'])
    if vocabulary_size not in (4096, 8192, 16384):
        raise ValueError('unsupported FAST vocabulary')
    for split in ('train', 'valid'):
        manifest = read(outer / 'data' / split / 'manifest.json')
        if (manifest['vocabulary_size'] != vocabulary_size
                or manifest['decoder_lineage_sha256'] != sha(lineage)
                or manifest['fingerprint'] != lineage_record['fingerprint']
                or manifest['encoder_bottleneck_sha256'] != lineage_record['encoder_bottleneck_sha256']):
            raise ValueError('FAST cache vocabulary/lineage differs from renderer')
    shutil.copy2(lineage, root / 'decoder_lineage.json')
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(original / 'code/src', root / 'code/src', ignore=ignore)
    shutil.copytree(encoder_stage / 'fast_src', root / 'code/fast_src', ignore=ignore)
    scripts = Path(__file__).resolve().parent
    fast_scripts = scripts.parents[1] / 'fast-rewrite/scripts'
    for name in ('native_recovery_common.py', 'released_dplm2_loader.py',
                 'released_dplm2_control.py', 'native_control_evaluate.py', 'check_native_recovery_resume.py'):
        shutil.copy2(original / 'code' / name, root / 'code' / name)
    for name in ('fast_recovery_common.py', 'train_fast_recovery.py', 'fast_recovery_eval.py',
                 'fast_recovery_decode.py', 'fast_recovery_sampling.py', 'native_tokenizer_loader.py',
                 'prepare_fast_recovery.py', 'verify_native_recovery_ready.py'):
        shutil.copy2(scripts / name, root / 'code' / name)
    for name in ('encode_fast_recovery.py', 'prepare_fast_recovery_coordinates.py',
                 'prepare_dplm_decoder_lineage.py', 'finalize_fast_recovery_cache.py'):
        shutil.copy2(fast_scripts / name, root / 'code' / name)
    cpu = (original / 'code/check_native_recovery_cpu.py').read_text().replace(
        'from native_recovery_common import', 'from fast_recovery_common import')
    (root / 'code/check_fast_recovery_cpu.py').write_text(cpu)
    cfg = read(original / 'config.json')
    vocab_source = Path(cfg['vocab_directory'])
    vocab = (vocab_source / 'vocab.txt').read_text().splitlines()
    assert len(vocab) == 8229 and vocab[36] == '0000'
    (root / 'vocabulary').mkdir()
    for name in ('tokenizer_config.json', 'special_tokens_map.json'):
        shutil.copy2(vocab_source / name, root / 'vocabulary' / name)
    # Released rows concatenate code strings. Uniform width prevents ambiguous
    # splitting when a vocabulary crosses the four-to-five decimal digit boundary.
    width = max(4, len(str(vocabulary_size - 1)))
    new_vocab = vocab[:36] + [f'{i:0{width}d}' for i in range(vocabulary_size)] + ['<mask_struct>']
    (root / 'vocabulary/vocab.txt').write_text('\n'.join(new_vocab) + '\n')
    if vocabulary_size == 8192:
        if (root/'vocabulary/vocab.txt').read_bytes() != (vocab_source/'vocab.txt').read_bytes():
            raise ValueError('matched8k vocabulary differs from native labels/order')
    token_cfg = read(root / 'vocabulary/tokenizer_config.json')
    token_cfg['added_tokens_decoder'] = {str(new_vocab.index(item['content'])): item
        for item in token_cfg.get('added_tokens_decoder', {}).values()}
    write(root / 'vocabulary/tokenizer_config.json', token_cfg)
    cfg.update(schema='released-fast-recovery-swap-v1', vocab_directory=str(root / 'vocabulary'),
        fast_python=str(fast_python),
        decoder_lineage_sha256=sha(root / 'decoder_lineage.json'), vocabulary_size=vocabulary_size,
        structure_token_width=width,
        encoder_stage=str(encoder_stage),encoder_bundle_sha256=sha(encoder_stage/'bundle.json'),
        native_comparison_bundle_sha256=sha(original / 'bundle.json'),
        data_projection=audit,
        representation_differences=['FAST encoder sees AA and full atoms; LFQ encoder is AA-blind',
            (f'{vocabulary_size} vs 8192 categorical codes changes embedding/head size and initialization RNG consumption'
             if vocabulary_size != 8192 else '8192 categorical codes match native vocabulary/head size; token semantics differ'),
            'Coordinates retrieved anew; AlphaFold v4 Foldcomp is lossy; original author encoder inputs are unattested',
            'Missing FAST frame slots are PAD in structure stream only; original AA and crop lengths unchanged'])
    cfg.pop('native_tokenizer_root', None)
    write(root / 'config.json', cfg)
    script = (original / 'submit.sbatch').read_text().replace('native-recovery', 'fast-recovery-swap')
    script = script.replace('native_recovery_%j', 'fast_recovery_%j')
    script = script.replace('train_native_recovery.py', 'train_fast_recovery.py')
    script = script.replace('native_recovery_eval.py', 'fast_recovery_eval.py')
    script = script.replace('check_native_recovery_cpu.py', 'check_fast_recovery_cpu.py')
    script = script.replace('REC_PY=/hpc/mydata/kyle.hippe/github/dplm/.venv/bin/python',
        'REC_PY=/hpc/mydata/kyle.hippe/github/dplm/.venv/bin/python\nREC_FAST_PY=/hpc/mydata/kyle.hippe/github/fast-rewrite/.venv/bin/python')
    script = script.replace('"$REC_PY" "$REC_ROOT/code/fast_recovery_eval.py" decode',
                            '"$REC_FAST_PY" "$REC_ROOT/code/fast_recovery_eval.py" decode')
    (root / 'submit.sbatch').write_text(script)
    files = {str(f.relative_to(root)): sha(f) for folder in ('code', 'vocabulary')
             for f in sorted((root / folder).rglob('*')) if f.is_file()}
    for name in ('config.json', 'panel.json', 'data_audit.json', 'reference_audit.json', 'decoder_lineage.json', 'submit.sbatch'):
        files[name] = sha(root / name)
    external = {}
    for folder in (native / 'dependencies', native / 'references', outer / 'data'):
        for f in sorted(folder.rglob('*')):
            if f.is_file() and f.suffix != '.pyc':
                external[str(f)] = sha(f)
    for name in ('pytorch_model.bin', 'config.json'):
        path = Path(cfg['pretrained']) / name
        external[str(path)] = sha(path)
    for key in ('config', 'checkpoint'):
        path = Path(read(lineage)[key])
        external[str(path)] = sha(path)
    write(root / 'bundle.json', dict(files=files, external_files=external,
        native_bundle_sha256=sha(original / 'bundle.json'), root=str(root)))
    print(dict(root=str(root), bundle_sha256=sha(root / 'bundle.json')))


if __name__ == '__main__':
    main()
