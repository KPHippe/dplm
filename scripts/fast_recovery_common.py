"""FAST-specific cache adapter; reuse the recovered released model and objective."""
from pathlib import Path

import numpy as np

from native_recovery_common import *  # noqa: F403
from native_recovery_common import read, sha


def vocabulary_size(vocab):
    size = len(vocab) - 37
    width = max(4, len(str(size - 1)))
    if (size not in (4096, 8192, 16384) or vocab[33:36] != ['<cls_struct>', '<eos_struct>', '<unk_struct>']
            or vocab[36:-1] != [f'{i:0{width}d}' for i in range(size)] or vocab[-1] != '<mask_struct>'):
        raise ValueError('wrong released-order FAST vocabulary')
    return size


class Corpus:
    def __init__(self, path, vocab, verify_files=True):
        import json
        self.path = Path(path)
        self.manifest = read(self.path / 'manifest.json')
        self.vocab = vocab
        size = vocabulary_size(vocab)
        if (self.manifest['schema'] != 'fast-released-projection-v1'
                or self.manifest['vocabulary_size'] != size):
            raise ValueError('wrong FAST corpus schema/vocabulary')
        self.vocabulary_size = size
        if verify_files:
            for name, expected in self.manifest['files'].items():
                if sha(self.path / name) != expected:
                    raise ValueError(f'FAST corpus changed: {name}')
        self.rows = [json.loads(x) for x in (self.path / 'rows.jsonl').read_text().splitlines()]
        self.codes = np.load(self.path / 'codes.npy', mmap_mode='r')
        self.aa = np.load(self.path / 'aa.npy', mmap_mode='r')
        self.offsets = np.load(self.path / 'offsets.npy', mmap_mode='r')
        self.valid = np.load(self.path / 'structure_mask.npy', mmap_mode='r')
        self.lengths = np.diff(self.offsets)
        if not (len(self.rows) == self.manifest['entries']
                and self.offsets[-1] == len(self.codes) == len(self.aa) == len(self.valid)
                and np.array_equal(self.lengths, [r['length'] for r in self.rows])):
            raise ValueError('unaligned FAST cache fields')
        if (self.codes.dtype != np.uint16 or self.valid.dtype != bool
                or np.any(self.codes[self.valid] >= size) or np.any(self.codes[~self.valid] != 65535)):
            raise ValueError('invalid FAST code or missing-frame sentinel')

    def record(self, index):
        lo, hi = self.offsets[index:index+2]
        # Released noising/loss already exclude PAD. Preserve AA and author row
        # length, with no invented structural label for an unobserved frame.
        struct = [self.vocab[36+int(x)] if valid else '<pad>'
                  for x, valid in zip(self.codes[lo:hi], self.valid[lo:hi], strict=True)]
        return dict(pdb_name=self.rows[index]['id'], length=int(hi-lo),
                    aa_seq=''.join(self.vocab[int(x)] for x in self.aa[lo:hi]),
                    struct_seq=','.join(struct))
