"""Claims the optimizer race narration makes, computed with real torch.optim -> facts.json.

The narration renders these numbers instead of typing them, and
optimizer-race.test.ts checks the live JS race reproduces them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from simref import OPT_NAMES, config, dump, race, require, slider_lr  # noqa: E402

cfg = config("optimizer-race")

# Claim: at the default learning rate, Momentum arrives first, then Adam, then SGD.
default = race(cfg["defaultLr"], cfg)
require(all(default[n]["arrived"] and not default[n]["diverged"] for n in OPT_NAMES),
        f"everyone should arrive at the default lr: {default}")
order = sorted(OPT_NAMES, key=lambda n: default[n]["arrived"])
require(order == ["Momentum", "Adam", "SGD"], f"arrival order changed: {order}")

# Claim: at the top of the slider (lr 1.0), SGD diverges and Momentum doesn't.
top_lr = cfg["lrRange"][1]
top = race(top_lr, cfg)
require(top["SGD"]["diverged"] is not None, "SGD should diverge at the top of the slider")
require(top["Momentum"]["diverged"] is None, "Momentum should not diverge at the top of the slider")

# Claim: SGD is the first to blow up as the slider rises. Find where.
sgd_from = None
for pos in range(cfg["sliderSteps"] + 1):
    lr = slider_lr(pos, cfg)
    r = race(lr, cfg)
    if sgd_from is None and r["SGD"]["diverged"]:
        sgd_from = {"pos": pos, "lr": lr}
    require(not r["Momentum"]["diverged"] and not r["Adam"]["diverged"],
            f"Momentum/Adam diverge at lr {lr:.3f}, so SGD isn't the only one that blows up")
require(sgd_from is not None, "SGD never diverges on the slider")

dump({
    "defaultLr": f"{cfg['defaultLr']:g}",
    "arrival": {n: default[n]["arrived"] for n in OPT_NAMES},
    "order": order,
    "orderText": ", then ".join(order),
    "topLr": f"{top_lr:g}",
    "top": {n: top[n] for n in OPT_NAMES},
    "sgdDivergesFromPos": sgd_from["pos"],
    "sgdDivergesFromLr": round(sgd_from["lr"], 6),
    "sgdDivergesFromText": f"{sgd_from['lr']:.2f}",
})
