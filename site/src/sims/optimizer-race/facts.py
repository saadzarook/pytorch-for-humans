"""Facts the optimizer race narration relies on, computed with real torch.optim -> facts.json.

This script only COMPUTES. Whether the narration is true is declared in
config.json ("claims") and checked automatically by scripts/test_snippets.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from simref import OPT_NAMES, config, dump, race, slider_lr  # noqa: E402

cfg = config("optimizer-race")

default = race(cfg["defaultLr"], cfg)
arrived = {n: default[n]["arrived"] for n in OPT_NAMES}
order = sorted(OPT_NAMES, key=lambda n: arrived[n] if arrived[n] is not None else float("inf"))

top_lr = cfg["lrRange"][1]
top = race(top_lr, cfg)

# Where on the slider does each optimizer diverge?
slider_diverged = {n: [] for n in OPT_NAMES}
for pos in range(cfg["sliderSteps"] + 1):
    r = race(slider_lr(pos, cfg), cfg)
    for n in OPT_NAMES:
        if r[n]["diverged"]:
            slider_diverged[n].append(pos)
sgd_from = slider_diverged["SGD"][0] if slider_diverged["SGD"] else None
sgd_from_lr = slider_lr(sgd_from, cfg) if sgd_from is not None else None

dump({
    "defaultLr": f"{cfg['defaultLr']:g}",
    "arrival": arrived,
    "defaultDiverged": {n: default[n]["diverged"] for n in OPT_NAMES},
    "order": order,
    "orderText": ", then ".join(order),
    "topLr": f"{top_lr:g}",
    "top": {n: top[n] for n in OPT_NAMES},
    "sliderDiverged": slider_diverged,
    "sgdDivergesFromPos": sgd_from,
    "sgdDivergesFromLr": round(sgd_from_lr, 6) if sgd_from_lr is not None else None,
    "sgdDivergesFromText": f"{sgd_from_lr:.2f}" if sgd_from_lr is not None else "never",
})
