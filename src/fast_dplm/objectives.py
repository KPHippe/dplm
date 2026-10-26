"""Opt-in DPLM-2 objective ablations; no changes to data draws or noising RNG.

Released self-mixup complements the original masks and supervises every residue.
Noise reweighting preserves weight mass per modality/task over the global update,
before any DDP/microbatch slicing. Neither option changes the native targets.
"""
import torch

from .data import RESIDUE


def loss_plan(batch, noising, *, self_mixup='off', noise_weighting='linear'):
    """Make loss selectors independently from the masks used for model input."""
    if self_mixup not in ('off', 'released'):
        raise ValueError('unknown self-mixup objective')
    if noise_weighting not in ('linear', 'constant_mass'):
        raise ValueError('unknown noise weighting')
    result = {k: v.clone() for k, v in noising.items()}
    real = batch['state'] == RESIDUE
    if real.device.type != 'cpu' or any(v.device.type != 'cpu' for v in noising.values()):
        raise ValueError('prepare the objective on the complete CPU global update')
    if self_mixup == 'released':
        result['struct_mask'] = real.clone()
        result['aa_mask'] = real.clone()
    for mask_key, weight_key in [('struct_mask', 'sw'), ('aa_mask', 'aw')]:
        mask = result[mask_key]
        if mask.dtype != torch.bool or mask.shape != real.shape or (mask & ~real).any():
            raise ValueError('loss selection must contain only real residues')
        weight = noising[weight_key]
        if weight.shape != (len(real),) or not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError('invalid diffusion weight')
        if noise_weighting == 'constant_mass':
            counts = mask.sum(-1).to(torch.float64)
            # Preserve relative task strength too: conditioned t=0 tracks in
            # released self-mixup retain their original weight (501/500).
            for task in range(4):
                selected = noising['task'] == task
                count = counts[selected].sum()
                if count > 0:
                    constant = (counts[selected]*weight[selected].double()).sum()/count
                    result[weight_key][selected] = constant.to(weight.dtype)
    return result


@torch.no_grad()
def released_mixup_input(model, batch, noising):
    """Mirror the released argmax/complement-mask construction, without gradients.

The proposal pass remains in the caller's train/eval mode (including dropout).
Argmax spans the shared vocabulary, as in released DPLM-2; constraining proposals
to each track would be a different experiment. BOS/EOS/padding are preserved.
The returned IDs are inputs only. Loss targets are always the untouched batch.
"""
    ids = model.input_ids(batch, noising['struct_mask'], noising['aa_mask'])
    proposal = model(batch, noising['struct_mask'], noising['aa_mask'], noising['independent'])
    if 'struct_loss_logits' not in proposal:
        raise ValueError('released self-mixup requires the shared-vocabulary DPLM-2 recipe')
    logits = torch.cat([proposal['struct_loss_logits'], proposal['aa']], dim=1)
    if not torch.isfinite(logits).all():
        raise FloatingPointError('nonfinite self-mixup proposal logits')
    real = torch.cat([batch['state'] == RESIDUE]*2, dim=1)
    mixed = torch.where(real, logits.argmax(-1), ids)
    n = batch['state'].shape[1]
    sm, am = noising['struct_mask'], noising['aa_mask']
    from .data import MASK
    mixed[:, :n].masked_fill_(real[:, :n] & ~sm, model.struct_special[MASK])
    mixed[:, n:].masked_fill_(real[:, n:] & ~am, model.special['mask'])
    return mixed
