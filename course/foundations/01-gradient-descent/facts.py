"""Derives and checks every number this lesson quotes. Prints them as JSON.

Run by scripts/test_snippets.py, which compares the output with facts.json
(`--update` rewrites it). lesson.mdx imports facts.json, and the notebook build
fills {{facts.*}} placeholders from it, so prose and code can't drift apart.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from lessonkit import check_copies, literal_arrays, load_json, require  # noqa: E402

data = load_json(HERE / "data.json")

# 1. Every hard-coded copy of the datasets must match data.json exactly.
copies = sorted(HERE.glob("snippets/*/*.py")) + [HERE / "notebook.py"]
problems = check_copies(data["delivery"], copies) + check_copies(data["chai"], copies)
require(not problems, "; ".join(problems))

x = np.array(data["delivery"]["x"])
y = np.array(data["delivery"]["y"])

# 2. Best-fit line (the "×" on the contour map).
w_best, b_best = np.polyfit(x, y, 1)

# 3. Shape of the (w, b) bowl. For MSE on a line, the Hessian is constant:
#    H = 2 * [[mean(x²), mean(x)], [mean(x), 1]]. Its eigenvalues are the
#    curvature along the bowl's two main axes. Gradient descent on a quadratic
#    diverges once lr * (largest eigenvalue) > 2.
H = 2 * np.array([[np.mean(x**2), np.mean(x)], [np.mean(x), 1.0]])
gentle, steep = np.linalg.eigvalsh(H)
max_lr = 2 / steep


# 4. Replay each contour-sim preset exactly as the sim does, and check the
#    lesson's claims about it.
def run(lr, start=data["contour"]["start"], max_steps=data["contour"]["maxSteps"]):
    w, b = float(start["w"]), float(start["b"])
    for step in range(1, max_steps + 1):
        e = w * x + b - y
        nw, nb = w - lr * np.mean(2 * e * x), b - lr * np.mean(2 * e)
        loss = np.mean((nw * x + nb - y) ** 2)
        if not np.isfinite(loss) or loss > 1e8:
            return "exploded", step
        if np.hypot(nw - w, nb - b) < 1e-4:
            return "converged", step
        w, b = nw, nb
    return "gave up", max_steps


presets = {p["id"]: p["lr"] for p in data["contour"]["presets"]}
outcomes = {pid: run(lr) for pid, lr in presets.items()}

require(outcomes["slow"][0] == "gave up", f"'too slow' should not finish in time, got {outcomes['slow']}")
require(outcomes["good"][0] == "converged", f"'just right' should converge, got {outcomes['good']}")
require(presets["zigzag"] < max_lr, "'zig-zag' must be below the stability limit")
require(outcomes["zigzag"][0] == "converged", f"'zig-zag' should still converge, got {outcomes['zigzag']}")
# Visible zig-zag: each step flips sides across the valley and only shrinks a little.
require(1 - presets["zigzag"] * steep < -0.5, "'zig-zag' should visibly bounce across the valley")
require(presets["chaos"] > max_lr, "'chaos' must be above the stability limit")
require(outcomes["chaos"][0] == "exploded", f"'chaos' should explode, got {outcomes['chaos']}")

# 5. The 1D ball sim: replay every preset with the sim's own rules (site/src/sims/ball/math.ts).
rules = data["ballRules"]


def make_curve(spec):
    c = spec["coeffs"]
    f = np.polynomial.Polynomial(c)
    df, ddf = f.deriv(), f.deriv(2)
    lo, hi = spec["domain"]
    grid = np.linspace(lo, hi, 4001)
    best = float(grid[np.argmin(f(grid))])
    for _ in range(20):
        if ddf(best) <= 0:
            break
        best -= df(best) / ddf(best)
    return f, df, ddf, best


def simulate_1d(spec, lr, start):
    f, df, _, gmin = make_curve(spec)
    w = start
    for steps in range(1, rules["maxSteps"] + 1):
        nxt = w - lr * df(w)
        if not np.isfinite(f(nxt)) or abs(nxt) > rules["explodeAbs"]:
            return "exploded", steps
        if abs(nxt - w) < rules["moveTol"] and abs(df(nxt)) < rules["slopeTol"]:
            return ("bottom" if abs(nxt - gmin) < rules["globalTol"] else "local"), steps
        w = nxt
    return "gave up", rules["maxSteps"]


ball = {cid: {p["id"]: simulate_1d(spec, p["lr"], p["start"]) for p in spec["presets"]}
        for cid, spec in data["curves"].items()}
expected = {
    "bowl": {"slow": "bottom", "good": "bottom", "bouncy": "bottom", "chaos": "exploded"},
    "bumpy": {"stuck": "local", "leap": "bottom", "goodstart": "bottom"},
}
for cid, by_preset in expected.items():
    for pid, kind in by_preset.items():
        require(ball[cid][pid][0] == kind, f"{cid} preset '{pid}' should end '{kind}', got {ball[cid][pid]}")

# The bowl's nerd corner: on a quadratic with curvature k = f'', each step multiplies the
# distance to the bottom by (1 - k·lr). It flips sides above lr = 1/k and grows above 2/k.
bowl_spec = data["curves"]["bowl"]
require(len(bowl_spec["coeffs"]) == 3, "the bowl must be a quadratic for the (1 - k·lr) argument")
k = 2 * bowl_spec["coeffs"][2]
flip_lr, chaos_lr = 1 / k, 2 / k
start = bowl_spec["defaultStart"]
require(simulate_1d(bowl_spec, chaos_lr * 0.95, start)[0] == "bottom", "just below the chaos lr it should still settle")
require(simulate_1d(bowl_spec, chaos_lr, start)[0] == "gave up", "at exactly the chaos lr it should bounce forever")
require(simulate_1d(bowl_spec, chaos_lr * 1.05, start)[0] == "exploded", "just above the chaos lr it should explode")

# 6. The optimizer race snippet must agree with the sim's own facts (both real torch.optim).
race_facts = load_json(HERE.parents[2] / "site" / "src" / "sims" / "optimizer-race" / "facts.json")
race_cfg = load_json(HERE.parents[2] / "site" / "src" / "sims" / "optimizer-race" / "config.json")
race_snippet = HERE / "snippets" / "pytorch" / "optimizer_race.py"
require(literal_arrays(race_snippet).get("start") == race_cfg["start"], "optimizer_race.py start differs from the sim's config.json")
race_out = (HERE / "snippets" / "outputs" / "pytorch" / "optimizer_race.txt").read_text(encoding="utf-8")
for name, steps_ in race_facts["arrival"].items():
    require(f"lr 0.1   {name:9s} reached the minimum in {steps_} steps" in race_out,
            f"optimizer_race.py output disagrees with the sim's facts for {name}")
require(f"lr 1.0   {'SGD':9s} blew up at step {race_facts['top']['SGD']['diverged']}" in race_out,
        "optimizer_race.py output disagrees with the sim's facts for SGD at lr 1.0")

facts = {
    "ball": {cid: {pid: {"kind": kind, "steps": s} for pid, (kind, s) in by.items()} for cid, by in ball.items()},
    "bowl": {"curvature": f"{k:g}", "flipLr": f"{flip_lr:g}", "chaosLr": f"{chaos_lr:g}"},
    "bestFit": {"w": f"{w_best:.2f}", "b": f"{b_best:.2f}", "wShort": f"{w_best:.1f}", "bShort": f"{b_best:.1f}"},
    "valley": {
        "steep": f"{steep:.2f}",
        "gentle": f"{gentle:.3f}",
        "ratio": f"{steep / gentle:.0f}",
        "maxLr": f"{max_lr:.3f}",
    },
    "presets": {pid: f"{lr:g}" for pid, lr in presets.items()},
    "presetOutcomes": {pid: f"{kind} after {steps} steps" for pid, (kind, steps) in outcomes.items()},
    "presetSteps": {pid: str(steps) for pid, (_, steps) in outcomes.items()},
}
print(json.dumps(facts, indent=2, ensure_ascii=False, sort_keys=True))
