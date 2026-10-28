"""
metrics/counterfactual.py  (Member 3 - Rick Sun)

Counterfactual importance: how strongly can changing ONLY one 4x4 region push
the prediction toward another class?

Shared interface (see README):
    score(model, image) -> torch.Tensor of shape (49,), values in [0, 1]
    image: float tensor (1, 1, 28, 28) in [0, 1]
    region index = row * 7 + col

Method
  For each region, run a masked L-inf PGD that edits only that region and
  minimises  logit[orig] - logit[target]  (target = best alternative class by
  default). Repeat at several budgets eps. Raw score = mean margin drop over
  budgets (rewards regions that move the logits a lot, and early). Then
  min-max normalise to [0, 1].

This is an empirical search, not a proof: a failed attack does NOT show that
no counterfactual exists.

Note:
Used AI to debug and to comment and describe code.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

REGION = 4
GRID = 28 // REGION        # 7
N_REG = GRID * GRID        # 49
BUDGETS = (0.25, 0.5, 0.75, 1.0)
STEPS = 20


# ---------------------------------------------------------------- helpers
def _region_masks(device):
    """(49, 1, 28, 28) binary masks; index = row * 7 + col."""
    m = torch.zeros(N_REG, 1, 28, 28, device=device)
    for r in range(N_REG):
        i, j = divmod(r, GRID)
        m[r, :, i * REGION:(i + 1) * REGION, j * REGION:(j + 1) * REGION] = 1
    return m


def _margin(logits, orig, target=None):
    """> 0 while `orig` still wins. target=None -> best alternative class."""
    if target is not None:
        return logits[:, orig] - logits[:, target]
    other = logits.clone()
    other[:, orig] = -1e9
    return logits[:, orig] - other.max(dim=1).values


def _minmax(a):
    rng = a.max() - a.min()
    return torch.zeros_like(a) if rng < 1e-12 else (a - a.min()) / rng


def _region_attack(model, image, orig, eps, target=None, steps=STEPS):
    """Batched over all 49 regions. Returns (margin_drop[49], flipped[49])."""
    masks = _region_masks(image.device)
    xs = image.detach().expand(N_REG, -1, -1, -1)
    delta = torch.zeros_like(xs, requires_grad=True)
    alpha = eps / 5

    with torch.no_grad():
        m0 = _margin(model(image), orig, target).item()
    best = torch.full((N_REG,), m0, device=image.device)

    for _ in range(steps):
        adv = (xs + delta * masks).clamp(0, 1)
        m = _margin(model(adv), orig, target)
        best = torch.minimum(best, m.detach())
        # grad w.r.t. delta only: does not touch the shared model's .grad
        (g,) = torch.autograd.grad(m.sum(), delta)
        with torch.no_grad():
            delta -= alpha * g.sign()
            delta.clamp_(-eps, eps)

    with torch.no_grad():
        adv = (xs + delta * masks).clamp(0, 1)
        best = torch.minimum(best, _margin(model(adv), orig, target))

    return (m0 - best).clamp(min=0), best < 0


# ------------------------------------------------------------- main API
def score(model, image, target=None, budgets=BUDGETS, return_details=False):
    """
    Returns torch.Tensor (49,) in [0, 1]; higher = more counterfactually
    important. `target`: int to measure push toward one specific class.
    """
    model.eval()
    with torch.enable_grad():
        with torch.no_grad():
            orig = model(image).argmax(dim=1).item()

        drops = []
        min_flip_eps = torch.full((N_REG,), float("inf"))
        for eps in budgets:
            d, flipped = _region_attack(model, image, orig, eps, target)
            drops.append(d)
            newly = flipped.cpu() & torch.isinf(min_flip_eps)
            min_flip_eps[newly] = eps

    raw = torch.stack(drops).mean(dim=0).detach().cpu()
    out = _minmax(raw)
    if return_details:
        return out, {"raw": raw, "min_flip_eps": min_flip_eps, "orig": orig}
    return out


def per_class_scores(model, image):
    """(10, 49): importance of each region for pushing toward each class.
    Row of the original class is zeros. Use for the 'alternative class' plots."""
    with torch.no_grad():
        orig = model(image).argmax(dim=1).item()
    out = torch.zeros(10, N_REG)
    for t in range(10):
        if t != orig:
            out[t] = score(model, image, target=t)
    return out


# ------------------------------------------------------------ evaluation
# These use perturbations DIFFERENT from the PGD used for scoring, so the
# evaluation is not circular:
#   zero  : erase the region            one  : fill with full ink
#   mean  : fill with image mean        noise: uniform random pixels
# "one" and "noise" can ADD ink, which matters because counterfactual search
# often finds blank regions where adding strokes changes the class.
MODES = ("zero", "one", "mean", "noise")


def _perturb(image, regions, mode, gen):
    x = image.clone()
    for r in regions:
        i, j = divmod(int(r), GRID)
        sl = (slice(None), slice(None),
              slice(i * REGION, (i + 1) * REGION),
              slice(j * REGION, (j + 1) * REGION))
        if mode == "zero":
            x[sl] = 0.0
        elif mode == "one":
            x[sl] = 1.0
        elif mode == "mean":
            x[sl] = image.mean()
        elif mode == "noise":
            x[sl] = torch.rand(x[sl].shape, generator=gen)
    return x


def _rand_subsets(n, k, gen):
    return [torch.randperm(N_REG, generator=gen)[:k] for _ in range(n)]


@torch.no_grad()
def deletion_test(model, image, scores, ks=(1, 3, 5, 10), modes=MODES,
                  n_random=50, seed=0):
    """Perturb the top-k regions of `scores` vs the mean over `n_random`
    random k-subsets. Returns flip rate and confidence drop for both."""
    gen = torch.Generator().manual_seed(seed)
    rank = torch.argsort(scores, descending=True)
    p0 = F.softmax(model(image), dim=1)
    c = p0.argmax().item()
    res = {}
    for mode in modes:
        for k in ks:
            xs = [_perturb(image, rank[:k], mode, gen)]
            xs += [_perturb(image, r, mode, gen)
                   for r in _rand_subsets(n_random, k, gen)]
            p = F.softmax(model(torch.cat(xs)), dim=1)
            flip = (p.argmax(dim=1) != c).float()
            drop = p0[0, c] - p[:, c]
            res[(mode, k)] = {
                "top_flip": flip[0].item(), "top_drop": drop[0].item(),
                "rand_flip": flip[1:].mean().item(),
                "rand_drop": drop[1:].mean().item(),
            }
    return res


@torch.no_grad()
def preservation_test(model, image, scores, ks=(5, 10, 15, 20),
                      n_trials=20, n_subsets=10, seed=0):
    """Keep top-k regions, randomise everything else with noise. Fraction of
    trials where the prediction is unchanged. Random baseline is averaged over
    `n_subsets` different random k-subsets."""
    gen = torch.Generator().manual_seed(seed)
    rank = torch.argsort(scores, descending=True)
    c = model(image).argmax().item()

    def frac_same(keep):
        keep = set(int(r) for r in keep)
        others = [r for r in range(N_REG) if r not in keep]
        xs = torch.cat([_perturb(image, others, "noise", gen)
                        for _ in range(n_trials)])
        return (model(xs).argmax(dim=1) == c).float().mean().item()

    out = {}
    for k in ks:
        rnd = [frac_same(r) for r in _rand_subsets(n_subsets, k, gen)]
        out[k] = {"top": frac_same(rank[:k]), "random": float(np.mean(rnd))}
    return out


def timed_score(model, image):
    t = time.perf_counter()
    s = score(model, image)
    return s, time.perf_counter() - t


def evaluate(model, images, csv_path="counterfactual_eval.csv",
             ks=(1, 3, 5, 10), pks=(5, 10, 15, 20)):
    """
    Run everything on a list of (1,1,28,28) images. Writes one CSV row per
    (image, test, setting) and prints image-averaged summaries.
    Returns (rows, summary).
    """
    import csv
    rows, times = [], []
    for n, img in enumerate(images):
        s, dt = timed_score(model, img)
        times.append(dt)
        for (mode, k), v in deletion_test(model, img, s, ks=ks).items():
            rows.append({"image": n, "test": "deletion", "setting": mode,
                         "k": k, **v})
        for k, v in preservation_test(model, img, s, ks=pks).items():
            rows.append({"image": n, "test": "preservation", "setting": "noise",
                         "k": k, "top_same": v["top"], "rand_same": v["random"]})

    # CSV (union of columns)
    cols = sorted({c for r in rows for c in r},
                  key=lambda c: ["image", "test", "setting", "k"].index(c)
                  if c in ("image", "test", "setting", "k") else 99)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # image-averaged summary
    summary = {}
    for r in rows:
        key = (r["test"], r["setting"], r["k"])
        d = summary.setdefault(key, {})
        for m, val in r.items():
            if m not in ("image", "test", "setting", "k"):
                d.setdefault(m, []).append(val)
    summary = {k: {m: float(np.mean(v)) for m, v in d.items()}
               for k, d in summary.items()}
    summary["runtime_sec_per_image"] = float(np.mean(times))
    return rows, summary
