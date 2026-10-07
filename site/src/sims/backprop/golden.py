"""Loss and gradients from torch autograd -> golden.json (checked by backprop.test.ts)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from simref import backprop_round, config, dump  # noqa: E402

cfg = config("backprop")
cases = [(cfg["start"]["w"], cfg["start"]["b"]), (0.0, 0.0), (3.2, -1.0), (-2.0, 4.0), (2.5, 2.0), (3.4, 0.3)]
rounds = []
w, b = cfg["start"]["w"], cfg["start"]["b"]
for _ in range(cfg["skipRounds"] + 1):  # the sim's "Skip 10 rounds" path
    r = backprop_round(w, b, cfg)
    rounds.append(r)
    w, b = r["w2"], r["b2"]
dump({"cases": [backprop_round(w, b, cfg) for w, b in cases], "rounds": rounds})
