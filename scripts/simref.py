"""Reference implementations of the course sims, in real PyTorch.

The sims re-implement maths in TypeScript. Each sim folder (site/src/sims/<sim>/)
has a golden.py and/or facts.py that call into this module and print JSON:

    golden.json  exact reference values; Vitest checks the TS maths matches them
    facts.json   numbers (and evidence) the sim's narration relies on; the claims
                 themselves are declared in the sim's config.json and checked by
                 scripts/claims.py. Vitest checks the live sim logic reproduces them.

Both are run and kept fresh by scripts/test_snippets.py, like lesson facts.
Every sim reads the same config.json as its TypeScript code, so the two can't
silently disagree about the setup.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SIMS = ROOT / "site" / "src" / "sims"
DTYPE = torch.float64  # JS numbers are float64; compare like with like


def config(sim: str) -> dict:
    return json.loads((SIMS / sim / "config.json").read_text(encoding="utf-8"))


def dump(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True))


# --------------------------------------------------------------- optimizer race
OPT_NAMES = ["SGD", "Momentum", "Adam"]


def valley_loss(p: torch.Tensor, cfg: dict) -> torch.Tensor:
    a, b, c = cfg["valley"]["a"], cfg["valley"]["b"], cfg["valley"]["c"]
    x, y = p[0], p[1]
    return a * x**2 + b * (y - c * torch.sin(x)) ** 2


def make_optimizer(name: str, params: list[torch.Tensor], lr: float, cfg: dict) -> torch.optim.Optimizer:
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr)
    if name == "Momentum":
        return torch.optim.SGD(params, lr=lr, momentum=cfg["momentum"])
    return torch.optim.Adam(params, lr=lr, betas=tuple(cfg["betas"]), eps=cfg["eps"])


def race_path(name: str, lr: float, steps: int, cfg: dict) -> list[list[float]]:
    """Raw path for `steps` updates with real torch.optim (no stopping rules)."""
    p = torch.tensor(cfg["start"], dtype=DTYPE, requires_grad=True)
    opt = make_optimizer(name, [p], lr, cfg)
    out = [p.detach().tolist()]
    for _ in range(steps):
        opt.zero_grad()
        valley_loss(p, cfg).backward()
        opt.step()
        out.append(p.detach().tolist())
    return out


def race_states(name: str, lr: float, steps: int, cfg: dict) -> list[dict]:
    """torch.optim's full state before every step (position + momentum buffer / Adam moments).

    Lets the JS test check ONE update at a time from PyTorch's exact state, which is
    immune to the way a diverging run amplifies last-bit differences in sin/cos.
    """
    p = torch.tensor(cfg["start"], dtype=DTYPE, requires_grad=True)
    opt = make_optimizer(name, [p], lr, cfg)
    states = []
    for _ in range(steps + 1):
        st = opt.state.get(p, {})
        entry = {"p": p.detach().tolist()}
        if "momentum_buffer" in st and st["momentum_buffer"] is not None:
            entry["v"] = st["momentum_buffer"].tolist()
        if "exp_avg" in st:
            entry["m"] = st["exp_avg"].tolist()
            entry["s"] = st["exp_avg_sq"].tolist()
            entry["t"] = int(st["step"].item())
        states.append(entry)
        opt.zero_grad()
        valley_loss(p, cfg).backward()
        opt.step()
    return states


def race(lr: float, cfg: dict) -> dict[str, dict]:
    """The race with the live sim's stopping rules: arrival = loss < arriveLoss, divergence = |x| or |y| > divergeAbs."""
    results = {}
    for name in OPT_NAMES:
        p = torch.tensor(cfg["start"], dtype=DTYPE, requires_grad=True)
        opt = make_optimizer(name, [p], lr, cfg)
        arrived = diverged = None
        for step in range(1, cfg["maxSteps"] + 1):
            opt.zero_grad()
            valley_loss(p, cfg).backward()
            opt.step()
            x, y = p.detach().tolist()
            if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > cfg["divergeAbs"] or abs(y) > cfg["divergeAbs"]:
                diverged = step
                break
            if arrived is None and valley_loss(p.detach(), cfg).item() < cfg["arriveLoss"]:
                arrived = step
        results[name] = {"arrived": arrived, "diverged": diverged}
    return results


def slider_lr(pos: int, cfg: dict) -> float:
    lo, hi = (math.log10(v) for v in cfg["lrRange"])
    return 10 ** (lo + (hi - lo) * pos / cfg["sliderSteps"])


# --------------------------------------------------------------------- backprop
def backprop_round(w: float, b: float, cfg: dict) -> dict:
    """loss = (w*x + b - y)**2; loss.backward(); manual SGD update — exactly the sim's code panel."""
    wt = torch.tensor(w, dtype=DTYPE, requires_grad=True)
    bt = torch.tensor(b, dtype=DTYPE, requires_grad=True)
    loss = (wt * cfg["x"] + bt - cfg["y"]) ** 2
    loss.backward()
    with torch.no_grad():
        w2 = (wt - cfg["lr"] * wt.grad).item()
        b2 = (bt - cfg["lr"] * bt.grad).item()
    return {"w": w, "b": b, "loss": loss.item(), "w_grad": wt.grad.item(), "b_grad": bt.grad.item(), "w2": w2, "b2": b2}


# ------------------------------------------------------------------ convolution
def conv_image(cfg: dict) -> torch.Tensor:
    return torch.tensor([[1.0 if ch == "#" else 0.0 for ch in row] for row in cfg["imageRows"]], dtype=DTYPE)


def conv_output(kernel: list[float], cfg: dict) -> torch.Tensor:
    """F.conv2d: cross-correlation (no flip), stride 1, no padding."""
    img = conv_image(cfg)[None, None]
    k = torch.tensor(kernel, dtype=DTYPE).view(1, 1, 3, 3)
    return torch.nn.functional.conv2d(img, k)[0, 0]


# --------------------------------------------------------------- neural network
def nn_reference(act: str, hidden: int, seed: int, points: list[list[float]], lr: float) -> dict:
    """A 2 → H → H → 1 nn.Sequential with BCEWithLogitsLoss (mean) and one Adam step."""
    gen = torch.Generator().manual_seed(seed)
    act_layer = torch.nn.Tanh if act == "tanh" else torch.nn.ReLU
    model = torch.nn.Sequential(
        torch.nn.Linear(2, hidden), act_layer(),
        torch.nn.Linear(hidden, hidden), act_layer(),
        torch.nn.Linear(hidden, 1),
    ).to(DTYPE)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=DTYPE) * 0.8)
    linears = [m for m in model if isinstance(m, torch.nn.Linear)]
    weights = [{"W": l.weight.tolist(), "b": l.bias.tolist()} for l in linears]

    X = torch.tensor([p[:2] for p in points], dtype=DTYPE)
    y = torch.tensor([[p[2]] for p in points], dtype=DTYPE)
    logits = model(X)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt.zero_grad()
    loss.backward()
    grads = [{"W": l.weight.grad.tolist(), "b": l.bias.grad.tolist()} for l in linears]
    opt.step()
    after = [{"W": l.weight.tolist(), "b": l.bias.tolist()} for l in linears]
    return {
        "act": act,
        "hidden": hidden,
        "weights": weights,
        "points": points,
        "probs": torch.sigmoid(logits).flatten().tolist(),
        "loss": loss.item(),
        "grads": grads,
        "lr": lr,
        "afterAdam": after,
    }
