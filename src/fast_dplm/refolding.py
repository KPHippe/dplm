"""Versioned, sequence-only Biohub ESMFold2 requests and shared prediction cache."""
import fcntl
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import sha256

ENDPOINT = 'https://biohub.ai/api/v1/fold'
MODEL = 'esmfold2-2026-05'
SETTINGS = dict(model=MODEL, num_loops=20, num_sampling_steps=100,
                lm_dropout=0.3, lm_mask_pct=0.0, include_distogram=False,
                include_pae=False, include_pair_chains_iptm=False, include_embeddings=False)
ALPHABET = 'ARNDCQEGHILKMFPSTWYV'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    temporary.write_bytes(canonical(value) + b'\n')
    os.replace(temporary, path)


def fold_request(sequence):
    if not 3 <= len(sequence) <= 768 or set(sequence) - set(ALPHABET):
        raise ValueError('ESMFold2 panel requires a canonical single chain of 3–768 residues')
    # Do not send generated structure, templates, an MSA, or FAST-decoded identities.
    return dict(SETTINGS, sequence=sequence)


def parse_fold(response, sequence):
    """Validate the positional atom37 response; API pLDDT is per residue, 0–1."""
    if response.get('model') != MODEL:
        raise ValueError('ESMFold2 response model mismatch')
    if response.get('sequence') is not None and response['sequence'] != sequence:
        raise ValueError('ESMFold2 response sequence mismatch')
    xyz = np.asarray(response.get('coordinates'), dtype=np.float64)
    if xyz.shape != (len(sequence), 37, 3):
        raise ValueError('ESMFold2 atom37 length/shape mismatch')
    finite = np.isfinite(xyz)
    if np.isinf(xyz).any() or (finite.any(-1) != finite.all(-1)).any():
        raise ValueError('ESMFold2 partially missing or infinite atom coordinates')
    present = finite.all(-1)
    if not present[:, 1].all():
        raise ValueError('ESMFold2 CA correspondence incomplete')
    confidence = np.asarray(response.get('plddt'), dtype=np.float64)
    if (confidence.shape != (len(sequence),) or not np.isfinite(confidence).all()
            or ((confidence < 0) | (confidence > 1)).any()):
        raise ValueError('ESMFold2 per-residue pLDDT must be finite on the 0–1 scale')
    for name in ('ptm', 'interface_ptm'):
        value = response.get(name)
        if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f'ESMFold2 invalid {name}')
    # A monomer may be numbered from zero or one, but must retain query ordering.
    index = response.get('residue_index')
    if index is not None:
        index = np.asarray(index)
        if index.shape != (len(sequence),) or not np.all(np.diff(index) == 1):
            raise ValueError('ESMFold2 residue order mismatch')
    entity = response.get('entity_id')
    if entity is not None and (len(entity) != len(sequence) or len(set(entity)) != 1):
        raise ValueError('ESMFold2 response is not one monomer')
    return xyz, present, 100 * confidence


class FoldCache:
    """Reuse one stochastic prediction per sequence/settings across both renderers.

    The public API exposes no random seed or weight digest. Cached returned JSON,
    timestamp and SHA256 are the reproducible prediction artifact, not a promise
    that a fresh API request gives identical coordinates.
    """

    def __init__(self, root, token_file=None):
        self.root = Path(root)
        self.token_file = Path(token_file or Path.home()/'.config/biohub/api-token')

    def _post(self, request):
        if self.token_file.stat().st_mode & 0o077:
            raise ValueError('Biohub token file must have owner-only permissions')
        token = self.token_file.read_text().strip()
        if not token or '\n' in token or '\r' in token:
            raise ValueError('invalid Biohub token file')
        for attempt in range(5):
            req = urllib.request.Request(ENDPOINT, data=canonical(request), headers={
                'Authorization': 'Bearer '+token, 'Content-Type': 'application/json',
                'Accept': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=300) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                if status not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise RuntimeError(f'Biohub fold failed with HTTP {status}') from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise RuntimeError('Biohub fold transport failed after five attempts') from None
            time.sleep(min(60, 5 * 2**attempt))
        raise RuntimeError('Biohub fold retries exhausted')

    def get(self, sequence):
        request = dict(endpoint=ENDPOINT, payload=fold_request(sequence))
        key = hashlib.sha256(canonical(request)).hexdigest()
        folder = self.root/key
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipt_path = folder/'receipt.json'
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if (receipt['cache_key'] != key
                        or json.loads((folder/'request.json').read_text()) != request
                        or sha256(folder/'response.json') != receipt['response_sha256']):
                    raise ValueError('ESMFold2 cache integrity failure')
                response = json.loads((folder/'response.json').read_text())
            else:
                if ((folder/'request.json').exists()
                        and json.loads((folder/'request.json').read_text()) != request):
                    raise ValueError('ESMFold2 cache request collision')
                write_json(folder/'request.json', request)
                start = time.monotonic()
                response = self._post(request['payload'])
                parse_fold(response, sequence)
                write_json(folder/'response.json', response)
                receipt = dict(cache_key=key, response_sha256=sha256(folder/'response.json'),
                               received_utc=datetime.now(timezone.utc).isoformat(),
                               elapsed_seconds=time.monotonic()-start,
                               model=MODEL, created=response.get('created'),
                               tokens_used=response.get('tokens_used'))
                write_json(receipt_path, receipt)
            parse_fold(response, sequence)
            return response, receipt
