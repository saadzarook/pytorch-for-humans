"""Facts the backprop narration relies on, computed with torch autograd -> facts.json.

This script only COMPUTES. The claims are declared in config.json and checked by
scripts/test_snippets.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from simref import backprop_round, config, dump  # noqa: E402

cfg = config("backprop")
w, b = cfg["start"]["w"], cfg["start"]["b"]
first = backprop_round(w, b, cfg)
losses = []
for _ in range(cfg["skipRounds"] * 3):
    r = backprop_round(w, b, cfg)
    losses.append(r["loss"])
    w, b = r["w2"], r["b2"]

dump({
    "startLoss": f"{first['loss']:.2f}",
    "lossRatioPerRound": f"{losses[1] / losses[0]:.2f}",
    "startError": first["w"] * cfg["x"] + first["b"] - cfg["y"],
    "startGrads": {"w": first["w_grad"], "b": first["b_grad"]},
    "lossByRound": losses,
})
