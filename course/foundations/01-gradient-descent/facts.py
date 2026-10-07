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
from lessonkit import check_copies, load_json, require  # noqa: E402

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

facts = {
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
