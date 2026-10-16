"""A tiny mixture-of-experts layer, from scratch.

The idea: instead of one big MLP that has to handle every input, keep a
bunch of small MLPs ("experts") and a router that picks which ones see
each input. Each input only pays for the experts it actually uses, so
you can grow total parameters without growing per-token compute.

This is the same trick behind the big sparse models in the news this
week -- hundreds of billions of total parameters, a small fraction
active per token. Here it's 8 experts, top-2 routing, on CPU, in
about 100 lines.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """One small MLP. Nothing fancy -- the routing is the point."""

    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MoELayer(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_in, n_experts)
        self.experts = nn.ModuleList(
            [Expert(d_in, d_hidden) for _ in range(n_experts)]
        )

    def forward(self, x):
        # Router scores every expert, then we keep the top-k per input
        # and renormalize their weights to sum to 1.
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        top_w, top_idx = probs.topk(self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)

        out = torch.zeros(x.size(0), device=x.device)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                mask = top_idx[:, k] == e
                if mask.any():
                    out[mask] += top_w[mask, k] * self.experts[e](x[mask])

        # Switch-style load-balancing loss: penalize the router for
        # piling everything onto one expert. This is the standard
        # safeguard against router collapse. Honest note: on this toy
        # the router mostly sorts itself out even with --no-aux (it
        # still hit 0.965); at real scale, with thousands of experts
        # and skewed data, this term is what keeps one expert from
        # eating the whole workload. f = fraction of inputs dispatched
        # to each expert (hard counts), p = mean router probability
        # per expert (soft). Loss is high when the same expert wins
        # on both.
        with torch.no_grad():
            dispatch = torch.zeros(self.n_experts, device=x.device)
            dispatch.scatter_add_(
                0, top_idx.reshape(-1),
                torch.ones(top_idx.numel(), device=x.device),
            )
            f = dispatch / dispatch.sum()
        p = probs.mean(dim=0)
        aux_loss = self.n_experts * (f * p).sum()

        return out, aux_loss, top_idx
