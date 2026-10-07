"""Claims the backprop narration makes, checked with torch autograd -> facts.json."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from simref import backprop_round, config, dump, require  # noqa: E402

cfg = config("backprop")
w, b = cfg["start"]["w"], cfg["start"]["b"]
first = backprop_round(w, b, cfg)
# Round 1: "Negative means the prediction is too low", and "the gradients are negative,
# so subtracting them raises w and b".
require(first["w"] * cfg["x"] + first["b"] - cfg["y"] < 0, "the starting prediction should be too low")
require(first["w_grad"] < 0 and first["b_grad"] < 0, "round-1 gradients should be negative")
# Skip ahead: "it keeps shrinking as the prediction closes in on 7".
losses = []
for _ in range(cfg["skipRounds"] * 3):
    r = backprop_round(w, b, cfg)
    losses.append(r["loss"])
    w, b = r["w2"], r["b2"]
require(all(nxt < cur for cur, nxt in zip(losses, losses[1:])), "loss should shrink every round")
dump({
    "startLoss": f"{first['loss']:.2f}",
    "lossRatioPerRound": f"{losses[1] / losses[0]:.2f}",
})
