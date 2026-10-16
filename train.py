"""Train a tiny MoE on a synthetic 4-task problem, and compare it against
a dense MLP with roughly the same total parameter count.

Each example belongs to one of 4 tasks (one-hot in the input), and each
task has its own labeling rule. The MoE should learn to route each task
to its own expert(s) -- specialization you can actually see in the
routing table at the end. The dense net has to cram all four rules into
one set of weights.

Usage:
    python train.py                 # MoE + dense baseline
    python train.py --no-aux       # same, but no load-balancing loss
    python train.py --epochs 50
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe import MoELayer

N_TASKS = 4
D_FEAT = 12
D_IN = D_FEAT + N_TASKS


def make_data(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    task = torch.randint(0, N_TASKS, (n,), generator=g)
    x = torch.randn(n, D_FEAT, generator=g)
    y = torch.zeros(n)
    # Each task gets its own rule. Deliberately mixed: linear, product,
    # threshold, and a two-feature combo, so no single tiny MLP has an
    # easy time on all four at once.
    y[task == 0] = (x[task == 0, 0] + x[task == 0, 1] > 0).float()
    y[task == 1] = (x[task == 1, 2] * x[task == 1, 3] > 0).float()
    y[task == 2] = (x[task == 2, 4].abs() > 0.8).float()
    y[task == 3] = (x[task == 3, 5] - x[task == 3, 6] + 0.5 * x[task == 3, 7] > 0).float()
    onehot = F.one_hot(task, N_TASKS).float()
    return torch.cat([x, onehot], dim=1), y, task


class MoEClassifier(nn.Module):
    def __init__(self, n_experts=8, top_k=2):
        super().__init__()
        self.moe = MoELayer(D_IN, 32, n_experts=n_experts, top_k=top_k)

    def forward(self, x):
        logits, aux, top_idx = self.moe(x)
        return logits, aux, top_idx


class DenseClassifier(nn.Module):
    """Same total params as the MoE (~4.7k), all of them active."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IN, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def train_model(model, x_train, y_train, x_val, y_val, epochs, aux_coef,
                lr=0.01, batch=256, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x_train)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = model(x_train[idx])
            # MoE returns (logits, aux, top_idx); dense returns logits.
            if isinstance(out, tuple):
                logits, aux = out[0], out[1]
            else:
                logits, aux = out, 0.0
            loss = F.binary_cross_entropy_with_logits(logits, y_train[idx])
            if aux_coef > 0:
                loss = loss + aux_coef * aux
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(x_val)
        logits = out[0] if isinstance(out, tuple) else out
        acc = ((logits > 0).float() == y_val).float().mean().item()
    return acc, logits


def per_task_acc(logits, y_val, task_val):
    accs = {}
    with torch.no_grad():
        pred = (logits > 0).float()
        for t in range(N_TASKS):
            m = task_val == t
            accs[t] = ((pred[m] == y_val[m]).float().mean().item())
    return accs


def routing_table(model, x_val, task_val):
    """Fraction of each task's inputs routed to each expert (top-1)."""
    model.eval()
    with torch.no_grad():
        _, _, top_idx = model(x_val)
    table = torch.zeros(N_TASKS, model.moe.n_experts)
    for t in range(N_TASKS):
        m = task_val == t
        for e in range(model.moe.n_experts):
            table[t, e] = (top_idx[m, 0] == e).float().mean()
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-aux", action="store_true",
                    help="turn off the load-balancing loss")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    x_train, y_train, _ = make_data(8000, seed=args.seed)
    x_val, y_val, task_val = make_data(2000, seed=args.seed + 999)

    aux_coef = 0.0 if args.no_aux else 0.01
    moe = MoEClassifier()
    dense = DenseClassifier()
    print(f"MoE params:   {count_params(moe)} (only "
          f"~{count_params(moe) // 4} active per input, top-2 of 8 experts)")
    print(f"Dense params: {count_params(dense)} (all active, every input)")

    acc, out = train_model(moe, x_train, y_train, x_val, y_val,
                           args.epochs, aux_coef, seed=args.seed)
    print(f"\nMoE val accuracy:   {acc:.3f}  (aux_coef={aux_coef})")
    for t, a in per_task_acc(out, y_val, task_val).items():
        print(f"  task {t}: {a:.3f}")

    table = routing_table(moe, x_val, task_val)
    print("\nRouting table (rows = task, cols = expert, top-1 share):")
    print("        " + "  ".join(f"E{e}" for e in range(table.size(1))))
    for t in range(N_TASKS):
        print(f"task {t}:  " + "  ".join(f"{v:.2f}" for v in table[t]))
    used = (table.max(dim=0).values > 0.05).sum().item()
    print(f"Experts doing real work (>5% of some task): {used}/8")

    d_acc, d_out = train_model(dense, x_train, y_train, x_val, y_val,
                               args.epochs, 0.0, seed=args.seed)
    print(f"\nDense val accuracy: {d_acc:.3f}")
    for t, a in per_task_acc(d_out, y_val, task_val).items():
        print(f"  task {t}: {a:.3f}")


if __name__ == "__main__":
    main()
