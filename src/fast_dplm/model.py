# Uses the ESM/SDPA construction in ByteDance DPLM (Apache-2.0) and HF Transformers.
"""Pretrained DPLM trunk with explicit FAST states and categorical product heads."""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import EsmConfig, EsmForMaskedLM
from transformers.models.esm.modeling_esm import EsmSelfAttention, rotate_half

from .data import ALPHABET, PAD, RESIDUE, MASK


class PairedAttention(EsmSelfAttention):
    """DPLM's pre-scaled queries and SDPA; repeat RoPE positions for both tracks."""
    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False):
        if any(x is not None for x in (head_mask, encoder_hidden_states, past_key_value)):
            raise ValueError("only bidirectional self-attention is supported")
        if output_attentions:
            raise ValueError("attention-map materialization is disabled")
        q = self.transpose_for_scores(self.query(hidden_states)) * self.attention_head_size**-0.5
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        n = hidden_states.shape[1]
        if n % 2 or self.position_embedding_type != "rotary":
            raise ValueError("paired rotary tracks required")
        pos = torch.arange(n // 2, device=q.device).repeat(2).float()
        phase = torch.outer(pos, self.rotary_embeddings.inv_freq.float())
        phase = torch.cat([phase, phase], -1)[None, None]
        cos, sin = phase.cos().to(q.dtype), phase.sin().to(q.dtype)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=1.0)
        return (ctx.transpose(1, 2).reshape(*hidden_states.shape[:2], self.all_head_size),)


class FastDPLM(nn.Module):
    def __init__(self, config, vocab, levels=(8, 8, 8, 8), mode="components"):
        super().__init__()
        if mode not in ("index", "components") or len(levels) != 4 or any(x < 2 for x in levels):
            raise ValueError("invalid representation")
        self.levels, self.mode, self.vocab = list(levels), mode, list(vocab)
        config = EsmConfig.from_dict(config) if isinstance(config, dict) else config
        config.tie_word_embeddings = False  # DPLM-2 uses separate input/output weights.
        self.net = EsmForMaskedLM(config)
        self.net.esm.contact_head = None
        self.net.esm.embeddings.position_embeddings.weight.requires_grad_(False)
        for layer in self.net.esm.encoder.layer:
            old = layer.attention.self
            new = PairedAttention(config)
            new.load_state_dict(old.state_dict(), strict=True)
            layer.attention.self = new
        d = config.hidden_size
        self.structure_embeddings = nn.ModuleList(
            [nn.Embedding(n, d) for n in (levels if mode == "components" else [math.prod(levels)])])
        self.structure_heads = nn.ModuleList(
            [nn.Linear(d, n) for n in (levels if mode == "components" else [math.prod(levels)])])
        self.special_embedding = nn.Embedding(5, d)
        self.terminal_embedding = nn.Embedding(2, d)
        for module in [*self.structure_embeddings, *self.structure_heads, self.special_embedding]:
            nn.init.normal_(module.weight, std=config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.terminal_embedding.weight)
        basis = [math.prod(levels[:i]) for i in range(len(levels))]
        self.register_buffer("basis", torch.tensor(basis))
        self.register_buffer("level_tensor", torch.tensor(levels))
        self.aa_map = [vocab.index(a) for a in ALPHABET[:20]]
        self.special = {k: vocab.index(v) for k, v in
                        dict(pad="<pad>", bos="<cls>", eos="<eos>", mask="<mask>").items()}

    @classmethod
    def from_dplm(cls, directory, **kwargs):
        directory = Path(directory)
        config = EsmConfig.from_pretrained(directory, local_files_only=True)
        vocab = (directory / "vocab.txt").read_text().splitlines()
        model = cls(config, vocab, **kwargs)
        sd = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True,
                        mmap=True)
        # Only the unused contact head can differ from the sequence checkpoint.
        sd = {k: v for k, v in sd.items() if not k.startswith("esm.contact_head.")}
        result = model.net.load_state_dict(sd, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"pretrained boundary mismatch: {result}")
        return model

    def specification(self):
        return dict(config=self.net.config.to_dict(), vocab=self.vocab,
                    levels=self.levels, mode=self.mode)

    def components(self, ids):
        return (ids[..., None] // self.basis) % self.level_tensor

    def pack(self, components):
        if ((components < 0) | (components >= self.level_tensor)).any():
            raise ValueError("out of range FAST component")
        return (components * self.basis).sum(-1)

    def forward(self, batch, struct_mask, aa_mask, independent=None):
        state, ids = batch["state"], batch["fast"]
        real, valid = state == RESIDUE, state != PAD
        if ((struct_mask | aa_mask) & ~real).any():
            raise ValueError("only residue positions can be masked")
        if ((ids[real] < 0) | (ids[real] >= math.prod(self.levels))).any():
            raise ValueError("FAST ID outside vocabulary")
        clean = real & ~struct_mask
        safe = ids.masked_fill(~clean, 0)
        labels = self.components(safe).unbind(-1) if self.mode == "components" else [safe]
        embed = sum(e(v) for e, v in zip(self.structure_embeddings, labels, strict=True))
        # All hidden target IDs are replaced before lookup, not just before scoring.
        stream_state = state.masked_fill(struct_mask, MASK)
        embed = torch.where(clean[..., None], embed, self.special_embedding(stream_state))
        aa_ids = batch["aa"].masked_fill(aa_mask, self.special["mask"])
        aa_embed = self.net.esm.embeddings(input_ids=aa_ids, attention_mask=valid)
        topology = self.terminal_embedding(batch["end"])
        hidden = torch.cat([embed + topology, aa_embed + topology], 1)
        length = state.shape[1]
        valid2 = torch.cat([valid, valid], 1)
        allowed = valid2[:, None, None, :].expand(-1, 1, 2 * length, -1).clone()
        if independent is not None:
            allowed[independent, :, :length, length:] = False
            allowed[independent, :, length:, :length] = False
        bias = torch.zeros_like(allowed, dtype=hidden.dtype).masked_fill(~allowed, -torch.inf)
        hidden = self.net.esm.encoder(hidden, attention_mask=bias, return_dict=True).last_hidden_state
        st, aa = hidden.chunk(2, 1)
        return dict(struct=[h(st) for h in self.structure_heads], aa=self.net.lm_head(aa))


def noise_plan(batch, generator, timesteps=500, balanced=False):
    """Plan on CPU in a dedicated RNG stream; optionally exact DPLM-2 task quarters."""
    b, n = batch["state"].shape
    if balanced:
        if b % 4:
            raise ValueError("balanced task batch must be divisible by four")
        task = torch.arange(4).repeat_interleave(b // 4)[torch.randperm(b, generator=generator)]
    else:
        task = torch.randint(4, (b,), generator=generator)
    ts = torch.randint(1, timesteps + 1, (b,), generator=generator)
    ta = torch.randint(1, timesteps + 1, (b,), generator=generator)
    ta[task == 0] = ts[task == 0]
    ta[task == 1] = 0
    ts[task == 2] = 0
    real = batch["state"].cpu() == RESIDUE
    sm = (torch.rand(b, n, generator=generator) < ts[:, None] / timesteps) & real
    am = (torch.rand(b, n, generator=generator) < ta[:, None] / timesteps) & real
    return dict(struct_mask=sm, aa_mask=am, independent=task == 3, task=task,
                sw=(timesteps - ts + 1).float() / timesteps,
                aw=(timesteps - ta + 1).float() / timesteps)


def loss_sums(model, output, batch, plan):
    """Return differentiable numerators and exact residue counts (no padding IDs)."""
    sm, am = plan["struct_mask"], plan["aa_mask"]
    targets = model.components(batch["fast"]).unbind(-1) if model.mode == "components" else [batch["fast"]]
    logits_list = output["struct"]
    if "struct_loss_logits" in output:
        logits_list = [output["struct_loss_logits"]]
        targets = [batch["fast"] + output["struct_offset"]]
    sums = []
    for logits, target in zip(logits_list, targets, strict=True):
        selected = logits[sm].float()
        if not torch.isfinite(selected).all():
            raise FloatingPointError("nonfinite structure logits")
        ce = F.cross_entropy(selected, target[sm], reduction="none")
        sums.append((ce * plan["sw"][:, None].expand_as(sm)[sm]).sum())
    aa_logits = output["aa"][am].float()
    if not torch.isfinite(aa_logits).all():
        raise FloatingPointError("nonfinite AA logits")
    aa_ce = F.cross_entropy(aa_logits, batch["aa"][am], reduction="none")
    aa_sum = (aa_ce * plan["aw"][:, None].expand_as(am)[am]).sum()
    return torch.stack([sum(sums), aa_sum]), torch.stack([sm.sum(), am.sum()])
