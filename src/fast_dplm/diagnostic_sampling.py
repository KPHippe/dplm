"""Explicit sampling ablations; the historical sampler and run remain immutable."""
import numpy as np
import torch

from .data import RESIDUE, collate
from .sampling import categorical, transition

POLICIES = {
    'legacy': dict(refresh=False, top_p=1., max_temp=1.1, released_schedule=False),
    'fresh_scores': dict(refresh=True, top_p=1., max_temp=1.1, released_schedule=False),
    'release_cool': dict(refresh=True, top_p=.95, max_temp=1.1, released_schedule=True),
    'release_default': dict(refresh=True, top_p=.95, max_temp=2., released_schedule=True),
}


def nucleus(logits, top_p):
    if not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0,1]')
    if top_p == 1:
        return logits
    values, indices = logits.float().sort(descending=True, dim=-1)
    remove = values.softmax(-1).cumsum(-1) > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    return torch.empty_like(values).scatter(-1, indices, values.masked_fill(remove, -torch.inf))


@torch.inference_mode()
def generate_diagnostic(model, records, steps=100, seed=1, policy='legacy', condition=None):
    if policy not in POLICIES or condition not in (None, 'aa', 'fast') or steps < 1:
        raise ValueError('invalid diagnostic sampling policy')
    cfg = POLICIES[policy]
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    batch = collate(records, model.aa_map, model.special, device)
    real = batch['state'] == RESIDUE
    editable_s = real if condition != 'fast' else torch.zeros_like(real)
    editable_a = real if condition != 'aa' else torch.zeros_like(real)
    sm, am = editable_s.clone(), editable_a.clone()
    ss = torch.full_like(batch['fast'], -torch.inf, dtype=torch.float32)
    asc = ss.clone()
    allowed = torch.tensor(model.aa_map, device=device)
    for t in range(steps):
        output = model(batch, sm, am)
        denominator = steps if cfg['released_schedule'] else max(steps-1, 1)
        temperature = .1 + (cfg['max_temp']-.1) * (1-t/denominator)
        aa, aa_score = categorical(nucleus(output['aa'][..., allowed], cfg['top_p']), temperature, generator)
        aa = allowed[aa]
        draws = [categorical(nucleus(logits, cfg['top_p']), temperature, generator) for logits in output['struct']]
        ids = model.pack(torch.stack([d[0] for d in draws], -1)) if model.mode == 'components' else draws[0][0]
        score = sum(d[1] for d in draws)
        fraction = 1-(t+1)/steps
        # Released DPLM-2 ranks fresh candidates at ALL editable positions,
        # including those whose accepted token is retained until remasking.
        batch['fast'], sm, ss = transition(batch['fast'], ids, sm, score, editable_s,
            fraction, generator, previous_scores=None if cfg['refresh'] else ss)
        batch['aa'], am, asc = transition(batch['aa'], aa, am, aa_score, editable_a,
            fraction, generator, previous_scores=None if cfg['refresh'] else asc)
    if sm.any() or am.any():
        raise RuntimeError('unfinished diagnostic sampling')
    reverse = {v: i for i, v in enumerate(model.aa_map)}
    results = []
    for b, record in enumerate(records):
        n = len(record['aa'])
        result = dict(fast_id=batch['fast'][b, 1:n+1].cpu().numpy().astype(np.uint16),
            aa=np.array([reverse[int(x)] for x in batch['aa'][b, 1:n+1]], dtype=np.uint8),
            residue_id=np.asarray(record.get('residue_id', np.arange(1, n+1)), dtype=np.int32),
            chain_end=np.asarray(record['chain_end'], dtype=bool))
        if condition == 'aa' and not np.array_equal(result['aa'], record['aa']):
            raise RuntimeError('conditioned AA changed')
        if condition == 'fast' and not np.array_equal(result['fast_id'], record['fast_id']):
            raise RuntimeError('conditioned structure tokens changed')
        results.append(result)
    return results
