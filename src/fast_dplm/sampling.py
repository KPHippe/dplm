"""DPLM reparameterized unconditioned sampling, with explicit modality states."""
import json
from pathlib import Path

import numpy as np
import torch

from .data import RESIDUE, collate, sha256


def categorical(logits, temperature, generator):
    logp = (logits.float() / temperature).log_softmax(-1)
    token = torch.multinomial(logp.exp().reshape(-1, logp.shape[-1]), 1,
                              generator=generator).reshape(logp.shape[:-1])
    return token, logp.gather(-1, token[..., None]).squeeze(-1)


def transition(old, candidate, previous_mask, scores, real, fraction, generator,
               stochastic=True, previous_scores=None):
    """Retain accepted tokens unless remasked; only previously masked slots update."""
    if previous_scores is not None:
        # Confidence belongs to the accepted token, not to an unused fresh draw.
        scores = torch.where(previous_mask, scores, previous_scores)
    remask = torch.zeros_like(real)
    for b in range(len(real)):
        pos = real[b].nonzero().flatten()
        count = int(len(pos) * fraction)
        rank = scores[b, pos].clone()
        if stochastic and count:
            u = torch.rand(rank.shape, device=rank.device, generator=generator).clamp(1e-7, 1-1e-7)
            rank = rank - fraction * (-u.log()).log()
        if count:
            remask[b, pos[rank.topk(count, largest=False).indices]] = True
    result = torch.where(previous_mask & ~remask, candidate, old)
    # Mask is separate from IDs; model hides stale IDs before embedding lookup.
    return result, remask, scores.masked_fill(remask, -torch.inf)


@torch.inference_mode()
def generate(model, length, count=1, steps=100, seed=1, max_temp=1.1, min_temp=0.1):
    if length < 1 or count < 1 or steps < 1 or min_temp <= 0 or max_temp < min_temp:
        raise ValueError("invalid generation settings")
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    records = [dict(aa=np.zeros(length, dtype=np.uint8), fast_id=np.zeros(length, dtype=np.uint16),
                    chain_end=np.arange(length) == length - 1) for _ in range(count)]
    batch = collate(records, model.aa_map, model.special, device)
    real = batch["state"] == RESIDUE
    sm, am = real.clone(), real.clone()
    ss = torch.full_like(batch["fast"], -torch.inf, dtype=torch.float32)
    aa_scores = ss.clone()
    aa_allowed = torch.tensor(model.aa_map, device=device)
    for t in range(steps):
        out = model(batch, sm, am)
        temp = min_temp + (max_temp - min_temp) * (1 - t / max(steps-1, 1))
        aa, asc = categorical(out["aa"][..., aa_allowed], temp, generator)
        aa = aa_allowed[aa]
        draws = [categorical(x, temp, generator) for x in out["struct"]]
        ids = model.pack(torch.stack([d[0] for d in draws], -1)) if model.mode == "components" else draws[0][0]
        score = sum(d[1] for d in draws)
        fraction = 1 - (t + 1) / steps
        batch["fast"], sm, ss = transition(batch["fast"], ids, sm, score, real, fraction,
                                          generator, previous_scores=ss)
        batch["aa"], am, aa_scores = transition(batch["aa"], aa, am, asc, real, fraction,
                                               generator, previous_scores=aa_scores)
    if sm.any() or am.any():
        raise RuntimeError("unfinished denoising")
    reverse = {v: i for i, v in enumerate(model.aa_map)}
    return [dict(fast_id=batch["fast"][b, 1:length+1].cpu().numpy().astype(np.uint16),
                 aa=np.array([reverse[int(x)] for x in batch["aa"][b, 1:length+1]], dtype=np.uint8),
                 residue_id=np.arange(1, length+1, dtype=np.int32),
                 chain_end=np.arange(length) == length-1) for b in range(count)]


def write_samples(out, samples, provenance):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    for i, sample in enumerate(samples):
        path = out / f"sample_{i:04d}.npz"
        np.savez(path, **sample)
        entries.append(dict(file=path.name, sha256=sha256(path)))
    (out / "manifest.json").write_text(json.dumps(dict(
        provenance, schema="fast-dplm-samples-v1", samples=entries), indent=2) + "\n")
