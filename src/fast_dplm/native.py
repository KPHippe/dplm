"""Explicit native LFQ provenance; native tokens never enter the FAST codec."""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_native_manifest(manifest):
    contract = manifest['tokenizer_contract']
    if (manifest.get('representation') != 'dplm2_native_lfq8192'
            or manifest['levels'] != [2]*13
            or manifest['fingerprint'] != 'native-lfq8k:'+digest(contract)
            or manifest['checkpoint_sha256'] != contract['checkpoint_sha256']
            or manifest['config_sha256'] != contract['config_sha256']
            or contract['vocabulary_size'] != 8192):
        raise ValueError('native tokenizer identity mismatch')


def remap_panel(panel, cache, source_cache):
    """Keep entry indices, identities, topology and random requests; bind new tokens."""
    from .generation_validation import record_hash, validate_generation_panel
    if panel['cache_digest'] != source_cache.digest or cache.entries != source_cache.entries:
        raise ValueError('native panel must use identical ordered source membership')
    result = json.loads(json.dumps(panel))
    result['cache_digest'] = cache.digest
    generation = result['generation_validation']
    generation['cache_digest'] = cache.digest
    generation['representation'] = 'dplm2_native_lfq8192'
    for entry in generation['entries']:
        original, native = source_cache[entry['index']], cache[entry['index']]
        for key in ('aa', 'residue_id', 'chain_end'):
            if not (original[key] == native[key]).all():
                raise ValueError('sequence/topology changed while retokenizing')
        entry['record_sha256'] = record_hash(native)
    validate_generation_panel(cache, generation)
    return result
