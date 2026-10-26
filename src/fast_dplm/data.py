"""Immutable numeric cache, measured source sampling, reproducible update batches."""
import hashlib
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch

ALPHABET = "ARNDCQEGHILKMFPSTWYVX"
PAD, RESIDUE, MASK, BOS, EOS = range(5)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class TokenCache:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.digest = sha256(self.root / "manifest.json")
        native = self.manifest['schema'] in ('native-dplm-corpus-v1', 'native-dplm-cache-v1')
        if not native and self.manifest.get('representation') == 'dplm2_native_lfq8192':
            raise ValueError('native tokenizer requires its explicit cache schema')
        if native:
            from .native import validate_native_manifest
            validate_native_manifest(self.manifest)
        self.shards = None
        if self.manifest['schema'] in ('fast-dplm-corpus-v1', 'native-dplm-corpus-v1'):
            self.shards, self.entries, self.shard_offsets = [], [], [0]
            for item in self.manifest['shards']:
                path = (self.root / item['path']).resolve()
                if not path.is_relative_to(self.root.resolve()) or sha256(path / 'manifest.json') != item['sha256']:
                    raise ValueError('corpus shard integrity failure')
                shard = TokenCache(path)
                if native != (shard.manifest['schema'] == 'native-dplm-cache-v1'):
                    raise ValueError('corpus representation mismatch')
                if shard.shards is not None:
                    raise ValueError('nested corpora are not supported')
                for key in ('fingerprint', 'levels', 'checkpoint_sha256'):
                    if shard.manifest[key] != self.manifest[key]:
                        raise ValueError('corpus tokenizer mismatch')
                self.shards.append(shard)
                self.entries.extend(shard.entries)
                self.shard_offsets.append(len(self.entries))
            if len(self.entries) != self.manifest['entries'] or not self.entries:
                raise ValueError('corpus entry count mismatch')
            if sum(e['length'] for e in self.entries) != self.manifest['residues']:
                raise ValueError('corpus residue count mismatch')
            self._sampling_profile()
            return
        if self.manifest["schema"] not in ("fast-dplm-cache-v1", "native-dplm-cache-v1"):
            raise ValueError("unsupported cache schema")
        if "".join(self.manifest["alphabet"]) != ALPHABET:
            raise ValueError("unsupported AA alphabet")
        expected = {'fast_id.npy', 'aa.npy', 'residue_id.npy', 'chain_end.npy', 'offsets.npy', 'entries.jsonl'}
        if set(self.manifest['files']) != expected:
            raise ValueError('incomplete cache integrity manifest')
        for file, digest in self.manifest["files"].items():
            if Path(file).name != file or sha256(self.root / file) != digest:
                raise ValueError(f"cache integrity failure: {file}")
        self.entries = [json.loads(x) for x in (self.root / "entries.jsonl").read_text().splitlines()]
        self.arrays = {k: np.load(self.root / f"{k}.npy", mmap_mode="r", allow_pickle=False)
                       for k in ("fast_id", "aa", "residue_id", "chain_end", "offsets")}
        off = self.arrays["offsets"]
        dtypes = dict(fast_id=np.uint16, aa=np.uint8, residue_id=np.int32, chain_end=np.bool_, offsets=np.int64)
        if any(a.ndim != 1 or a.dtype != dtypes[k] for k, a in self.arrays.items()):
            raise ValueError('cache dtype/shape mismatch')
        if len(off) != len(self.entries) + 1 or off[0] != 0 or (np.diff(off) <= 0).any():
            raise ValueError("invalid entry offsets")
        if any(len(a) != off[-1] for k, a in self.arrays.items() if k != "offsets"):
            raise ValueError("unaligned cache arrays")
        if (self.arrays["fast_id"] >= np.prod(self.manifest["levels"])).any():
            raise ValueError("out of range cached ID")
        if (self.arrays["aa"] >= 20).any():
            raise ValueError("noncanonical AA in v1 cache")
        for i, e in enumerate(self.entries):
            if e["length"] != off[i + 1] - off[i]:
                raise ValueError("incorrect entry length")
        self._sampling_profile()

    def _sampling_profile(self):
        identities = [(e['source'], e['entry_id']) for e in self.entries]
        if len(set(identities)) != len(identities) or not identities:
            raise ValueError('empty or duplicate cache entries')
        self.groups = {}
        for i, e in enumerate(self.entries):
            self.groups.setdefault(e["source"], {}).setdefault(e["cluster"], []).append(i)
        self.sources = sorted(self.groups)
        self.clusters = [list(self.groups[s].values()) for s in self.sources]
        means = [np.mean([np.mean([self.entries[i]["length"] for i in g]) for g in gs])
                 for gs in self.clusters]
        # Equal target residue shares; measured under cluster -> member sampling.
        probs = 1 / np.asarray(means)
        self.probabilities = probs / probs.sum()
        self.profile = dict(source_names=self.sources, mean_residues=means,
                            draw_probabilities=self.probabilities.tolist(),
                            target_residue_shares=[1 / len(means)] * len(means))

    def __getitem__(self, i):
        if not 0 <= i < len(self.entries):
            raise IndexError(i)
        if self.shards is not None:
            shard = bisect_right(self.shard_offsets, i) - 1
            return self.shards[shard][i - self.shard_offsets[shard]]
        lo, hi = self.arrays["offsets"][i:i + 2]
        return {k: np.array(a[lo:hi], copy=True) for k, a in self.arrays.items() if k != "offsets"}

    def draw_update(self, step, global_batch, seed):
        """Stateless update-addressable plan, independent of model/prefetch RNG."""
        g = np.random.default_rng(np.random.SeedSequence([seed, step]))
        indices = []
        for _ in range(global_batch):
            s = g.choice(len(self.sources), p=self.probabilities)
            clusters = self.clusters[s]
            members = clusters[g.integers(len(clusters))]
            indices.append(members[g.integers(len(members))])
        # Bucketing only permutes the same global multiset, preserving exposure.
        return sorted(indices, key=lambda i: self.entries[i]["length"])


def collate(records, aa_map, special, device="cpu"):
    b, s = len(records), max(len(r["aa"]) for r in records) + 2
    state = torch.full((b, s), PAD, dtype=torch.long)
    fast = torch.zeros((b, s), dtype=torch.long)
    aa = torch.full((b, s), special["pad"], dtype=torch.long)
    end = torch.zeros((b, s), dtype=torch.long)
    for i, r in enumerate(records):
        n = len(r["aa"])
        state[i, 0], state[i, n + 1] = BOS, EOS
        state[i, 1:n + 1] = RESIDUE
        aa[i, 0], aa[i, n + 1] = special["bos"], special["eos"]
        aa[i, 1:n + 1] = torch.tensor([aa_map[int(x)] for x in r["aa"]])
        fast[i, 1:n + 1] = torch.from_numpy(r["fast_id"].astype(np.int64))
        end[i, 1:n + 1] = torch.from_numpy(r["chain_end"].astype(np.int64))
    return {k: v.to(device) for k, v in dict(state=state, fast=fast, aa=aa, end=end).items()}
