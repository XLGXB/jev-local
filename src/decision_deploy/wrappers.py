"""Tensor-only, raw-logit export interfaces. Never export an LM vocabulary head."""
from __future__ import annotations
import torch
from torch import nn

class LayaGraph(nn.Module):
    names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        # Native logits are raw. Unused auxiliary act logits are not graph outputs.
        logits, _ = self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        return logits.float()

class KevGraph(nn.Module):
    names = ["input_ids", "attention_mask", "position_ids", "decide_idx", "option_idx", "option_mask"]
    def __init__(self, model):
        super().__init__(); self.backbone = model.lm; self.pointer = model.head
        if model.option_isolation:
            raise ValueError("this causal-row export does NOT support option_isolation; use native backend")
        self.pointer.temperature = 1.0
    def forward(self, input_ids, attention_mask, position_ids, decide_idx, option_idx, option_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                          position_ids=position_ids, use_cache=False, return_dict=True).last_hidden_state
        d = h.shape[-1]
        qh = torch.gather(h, 1, decide_idx.unsqueeze(-1).expand(-1, -1, d))
        kh = torch.gather(h, 1, option_idx.unsqueeze(-1).expand(-1, -1, d))
        # Batched equivalent of the native pointer head (not a newly initialized layer).
        score = (self.pointer.k(kh) * self.pointer.q(qh)).sum(-1) * self.pointer.scale
        return score.float().masked_fill(~option_mask, -1e4)
