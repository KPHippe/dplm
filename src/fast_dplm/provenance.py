"""Explicit decoder lineage; never bypass a cache/codec fingerprint mismatch."""
import json

from .data import sha256


def decoder_provenance(manifest, lineage_path=None):
    if manifest.get('schema') in ('native-dplm-cache-v1', 'native-dplm-corpus-v1'):
        from .native import validate_native_manifest
        validate_native_manifest(manifest)
        if lineage_path is not None:
            raise ValueError('FAST decoder lineage cannot apply to native LFQ tokens')
        for key in ('checkpoint', 'config'):
            if sha256(manifest[key]) != manifest[key+'_sha256']:
                raise ValueError(f'native tokenizer {key} changed')
        return {k: manifest[k] for k in ('fingerprint', 'levels', 'config', 'config_sha256',
                'checkpoint', 'checkpoint_sha256', 'representation', 'tokenizer_contract')}
    if lineage_path is None:
        return {k: manifest[k] for k in ("fingerprint", "levels", "config", "checkpoint", "checkpoint_sha256")}
    lineage = json.loads(lineage_path.read_text())
    if (lineage.get("schema") != "fast-dplm-decoder-lineage-v1"
            or lineage["cache_fingerprint"] != manifest["fingerprint"]
            or lineage["parent_checkpoint_sha256"] != manifest["checkpoint_sha256"]
            or lineage["levels"] != manifest["levels"]):
        raise ValueError("decoder lineage does not match token cache")
    for key in ("checkpoint", "config"):
        if sha256(lineage[key]) != lineage[key+"_sha256"]:
            raise ValueError(f"decoder lineage {key} changed")
    return lineage | dict(lineage_sha256=sha256(lineage_path))
