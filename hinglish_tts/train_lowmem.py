"""Run Pocket TTS training with a reduced GPU memory footprint.

Wraps the upstream `training/train.py` unchanged and swaps two pieces that
dominate memory for a 24-layer model:

* AdamW -> bitsandbytes AdamW8bit: optimizer state drops from 8 to ~2 bytes
  per parameter.
* EMA shadow weights live in CPU RAM and are refreshed every `EMA_EVERY`
  steps (with the decay raised to the matching power), instead of a full
  fp32 copy on the GPU.

Usage (from the pocket-tts checkout, so its relative config paths resolve):
    python -m hinglish_tts.train_lowmem configs/hinglish_24l_local.yaml
"""

import sys

import bitsandbytes as bnb
import torch

import training.train as upstream
from training.checkpointing import EMA

EMA_EVERY = 4


def build_optimizer_8bit(model, args, device, rank):
    table = model.flow_lm.conditioner.embed.weight
    params = [p for p in model.parameters() if p.requires_grad and p is not table]
    no_decay = [table] if table.requires_grad else []
    groups = [{"params": params}, {"params": no_decay, "weight_decay": 0.0}]
    return bnb.optim.AdamW8bit(
        [g for g in groups if g["params"]],
        lr=args.optim.lr,
        betas=args.optim.betas,
        eps=args.optim.eps,
        weight_decay=args.optim.weight_decay,
    )


class CpuEMA(EMA):
    def __init__(self, model, decay):
        self.decay = decay**EMA_EVERY
        self.shadow = {
            k: p.detach().float().cpu().clone()
            for k, p in model.named_parameters()
            if p.requires_grad
        }
        self._tracked = None
        self._calls = 0

    def update(self, model):
        self._calls += 1
        if self._calls % EMA_EVERY:
            return
        with torch.no_grad():
            for k, p in model.named_parameters():
                if k in self.shadow:
                    s = self.shadow[k]
                    s.mul_(self.decay).add_(p.detach().float().cpu(), alpha=1 - self.decay)

    def load_state_dict(self, state):
        self.shadow = {k: v.float().cpu().clone() for k, v in state.items()}


upstream.build_optimizer = build_optimizer_8bit
upstream.EMA = CpuEMA

if __name__ == "__main__":
    upstream.main(sys.argv[1])
