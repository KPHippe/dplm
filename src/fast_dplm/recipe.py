"""DPLM-2 index-vocabulary/LoRA recipe on the pinned Transformers overlay.

The shared embedding/head, two embedding calls (including ESM mask scaling),
paired RoPE and adapter targets mirror the released ByProt implementation.
No legacy ByProt training dependencies are imported.
"""
import math
from pathlib import Path

import torch
from torch import nn
from transformers import EsmConfig, EsmForMaskedLM

from .data import ALPHABET, BOS, EOS, MASK, RESIDUE
from .model import PairedAttention


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32., dropout=.1):
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid LoRA settings")
        self.base = base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.base(x) + self.b(self.a(self.dropout(x))) * self.scale


class DPLM2Recipe(nn.Module):
    def __init__(self, config, vocab, levels=(8, 8, 8, 8), mode="index",
                 lora_rank=16, lora_alpha=32., lora_dropout=.1):
        super().__init__()
        if mode != "index" or not levels or any(n < 2 for n in levels) or len(vocab) != 33:
            raise ValueError("DPLM-2 recipe requires index tokens and the original 33-token AA vocabulary")
        self.vocab, self.levels, self.mode = list(vocab), list(levels), mode
        self.struct_offset = 33
        self.struct_size = math.prod(levels)
        self.struct_special = dict(zip((BOS, EOS, MASK),
                                      range(33+self.struct_size, 36+self.struct_size)))
        self.lora_rank, self.lora_alpha, self.lora_dropout = lora_rank, lora_alpha, lora_dropout
        config = EsmConfig.from_dict(config) if isinstance(config, dict) else config
        config.vocab_size = 33 + self.struct_size + 4  # Includes reserved structure UNK.
        config.tie_word_embeddings = False
        config.hidden_dropout_prob = .1
        self.net = EsmForMaskedLM(config)
        self.net.esm.contact_head = None
        for layer in self.net.esm.encoder.layer:
            old = layer.attention.self
            layer.attention.self = PairedAttention(config)
            layer.attention.self.load_state_dict(old.state_dict(), strict=True)
        self.aa_map = [vocab.index(a) for a in ALPHABET[:20]]
        self.special = {k: vocab.index(v) for k, v in
                        dict(pad="<pad>", bos="<cls>", eos="<eos>", mask="<mask>").items()}
        self._configure_trainable()

    def _configure_trainable(self):
        self.net.requires_grad_(False)
        if self.lora_rank:
            for layer in self.net.esm.encoder.layer:
                for module, name in ((layer.attention.self, "query"),
                                     (layer.attention.self, "key"),
                                     (layer.attention.self, "value"),
                                     (layer.attention.output, "dense"),
                                     (layer.intermediate, "dense"),
                                     (layer.output, "dense")):
                    setattr(module, name, LoRALinear(getattr(module, name), self.lora_rank,
                                                    self.lora_alpha, self.lora_dropout))
        self.net.esm.embeddings.requires_grad_(True)
        self.net.lm_head.requires_grad_(True)
        # Rotary attention never reads this table. Exclude unused DDP parameters.
        self.net.esm.embeddings.position_embeddings.requires_grad_(False)

    @classmethod
    def from_dplm(cls, directory, **kwargs):
        directory = Path(directory)
        config = EsmConfig.from_pretrained(directory, local_files_only=True)
        vocab = (directory / "vocab.txt").read_text().splitlines()
        rank = kwargs.pop("lora_rank", 16)
        model = cls(config, vocab, lora_rank=0, **kwargs)
        state = torch.load(directory / "pytorch_model.bin", map_location="cpu",
                           weights_only=True, mmap=True)
        state = {k: v for k, v in state.items() if not k.startswith("esm.contact_head.")}
        expanded = model.net.state_dict()
        for key in ("esm.embeddings.word_embeddings.weight", "lm_head.decoder.weight", "lm_head.bias"):
            old, new = state[key], expanded[key].clone()
            new[:33].copy_(old)
            if key == "esm.embeddings.word_embeddings.weight":
                # Released resize implementation samples the original per-feature mean/std.
                new[33:] = torch.normal(old.mean(0).expand_as(new[33:]), old.std(0).expand_as(new[33:]))
            elif key == "lm_head.bias":
                new[33:].zero_()
            state[key] = new
        model.net.load_state_dict(state, strict=True)
        model.lora_rank = rank
        model._configure_trainable()
        return model

    def specification(self):
        return dict(architecture="dplm2_recipe", config=self.net.config.to_dict(), vocab=self.vocab,
                    levels=self.levels, mode=self.mode, lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout)

    def input_ids(self, batch, struct_mask, aa_mask):
        state, ids = batch["state"], batch["fast"]
        real = state == RESIDUE
        if ((struct_mask | aa_mask) & ~real).any():
            raise ValueError("only residue positions can be masked")
        if ((ids[real] < 0) | (ids[real] >= self.struct_size)).any():
            raise ValueError("structure ID outside vocabulary")
        st = torch.full_like(ids, self.special["pad"])
        st[real] = ids[real] + self.struct_offset
        for label, token in self.struct_special.items():
            st[state == label] = token
        st[struct_mask] = self.struct_special[MASK]
        aa = batch["aa"].masked_fill(aa_mask, self.special["mask"])
        return torch.cat([st, aa], 1)

    def forward(self, batch, struct_mask, aa_mask, independent=None, token_input_ids=None):
        # Explicit raw-token input supports the released two-pass self-mixup
        # objective. The ordinary path is unchanged; targets remain separate.
        ids = self.input_ids(batch, struct_mask, aa_mask) if token_input_ids is None else token_input_ids
        if ids.shape != (batch['state'].shape[0], 2*batch['state'].shape[1]):
            raise ValueError('paired raw-token input shape mismatch')
        if ((ids < 0) | (ids >= self.net.config.vocab_size)).any():
            raise ValueError('raw-token input outside shared vocabulary')
        valid = ids != self.special["pad"]
        n = ids.shape[1] // 2
        allowed = valid[:, None, None, :].expand(-1, 1, 2*n, -1).clone()
        if independent is not None:
            allowed[independent, :, :n, n:] = False
            allowed[independent, :, n:, :n] = False
        # Preserve the released wrapper's two embedding calls, including mask scaling.
        hidden = self.net.esm.embeddings(input_ids=ids, attention_mask=valid)
        hidden = self.net.esm.embeddings(input_ids=ids, inputs_embeds=hidden, attention_mask=valid)
        bias = torch.zeros_like(allowed, dtype=hidden.dtype).masked_fill(~allowed, -torch.inf)
        hidden = self.net.esm.encoder(hidden, attention_mask=bias, return_dict=True).last_hidden_state
        st, aa = self.net.lm_head(hidden).chunk(2, 1)
        # Training retains the released shared softmax over both modalities.
        # Sampling is constrained to each track's valid residue vocabulary.
        return dict(struct=[st[..., self.struct_offset:self.struct_offset+self.struct_size]],
                    struct_loss_logits=st, struct_offset=self.struct_offset, aa=aa)


def model_from_spec(spec):
    from .model import FastDPLM
    spec = dict(spec)
    architecture = spec.pop("architecture", "fast_bridge")
    if architecture == "dplm2_recipe":
        return DPLM2Recipe(**spec)
    if architecture == "fast_bridge":
        return FastDPLM(**spec)
    raise ValueError(f"unknown architecture: {architecture}")
