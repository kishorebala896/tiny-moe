# tiny-moe

This week two giant sparse models dropped on the same day — Meta's Muse Spark Open (816B total parameters, 42B active per token) and Stepfun's Step 5 (600B total, 27B active). The headline number in both cases is the gap between "total" and "active," and that gap is the whole trick. I wanted to see the trick itself, up close, instead of just reading about it — so I built the smallest version of it I could that still actually works.

It's 8 experts, top-2 routing, trained on CPU in about 15 seconds. No model downloads, no API keys.

## What it does

`train.py` builds a synthetic 4-task problem. Every example belongs to one of four tasks (one-hot encoded in the input), and each task has its own labeling rule — one linear, one a product term, one an absolute-value threshold, one a two-feature combo. Then it trains two models with roughly the same total parameter count (~4.7k):

- **MoE**: a router picks the top 2 of 8 small MLP experts per input. Only ~1.2k parameters are active per input.
- **Dense**: one MLP where every parameter fires on every input.

The point of the comparison is the thing the big-model press releases are all about: same total capacity, a quarter of the per-token compute.

## What I got

On my machine (seed 0, 30 epochs, CPU) — numbers move a little run to run:

| Model | Val accuracy |
|---|---|
| MoE (aux loss on) | ~0.97 |
| MoE (aux loss off, `--no-aux`) | ~0.965 |
| Dense baseline | ~0.95 |

The accuracy gap is small — this is a toy, and the dense net is not exactly struggling. The interesting part is the routing table. With the balancing loss on, one run gave:

```
        E0   E1   E2   E3   E4   E5   E6   E7
task 0: 0.00 0.00 0.00 0.65 0.00 0.28 0.01 0.05
task 1: 0.07 0.00 0.02 0.00 0.91 0.00 0.00 0.00
task 2: 0.00 1.00 0.00 0.00 0.00 0.00 0.00 0.00
task 3: 0.01 0.01 0.00 0.38 0.00 0.05 0.46 0.09
```

Nobody told the router what the tasks mean. It figured out that task 2 belongs to expert 1, task 1 mostly lives on expert 4, and so on. Seven of the eight experts are doing real work. That specialization emerging from nothing but gradient descent is the part that got me — you don't hand-assign experts, you just give the router a reason to split the work and it does.

The surprise in the other direction: turning the balancing loss off (`--no-aux`) barely hurt. Still 0.965, still specialized, all 8 experts used. I left the loss in because it's the standard safeguard and it's clearly doing *something*, but on a toy this size the failure mode it guards against — one expert eating everything — doesn't really show up. At real scale, with thousands of experts and skewed data, it's a different story. Toys hide that. Worth knowing before you conclude the loss doesn't matter.

## How to run it

```bash
pip install -r requirements.txt
python train.py            # MoE + dense baseline, ~15s on CPU
python train.py --no-aux   # same, without the load-balancing loss
python train.py --epochs 50
```

## How the routing works

The router is just a linear layer over the input, softmaxed. Take the top-2 experts, renormalize their weights to sum to 1, and take the weighted sum of their outputs. That's it — there's no magic layer, which is kind of the point. The subtlety is all in training: the router only gets gradient through the experts it picked, so an expert that falls behind early can starve forever. The auxiliary loss fixes that by penalizing the router when the fraction of inputs it sends to an expert and its average confidence in that expert line up too hard on the same expert — the Switch Transformer formulation, `N * sum(f_i * p_i)`.

## Rough edges

- The expert loop in `moe.py` iterates experts in Python instead of batching them. Fine for 8 experts, would be embarrassing at 256.
- No capacity factor, no dropped tokens, no expert dropout — the stuff real MoE systems spend most of their engineering on. This is the idea, not the production system.
- The task identity is one-hot in the input, which makes routing easier than the real world, where the router has to infer "what kind of input is this" from the content itself.
- Runs vary a bit run to run (~±0.005) from CPU nondeterminism. Don't read too much into the third decimal.

## TODO

- Swap the synthetic task for something where the router has to infer the domain from content, no one-hot hint.
- Add a capacity factor and token dropping, then measure what it costs.
- Try expert-choice routing (experts pick their top inputs) and see if the specialization looks different.
- Scale to a tiny MoE transformer on real text and watch the router's behavior across layers — that's where it gets genuinely interesting.
