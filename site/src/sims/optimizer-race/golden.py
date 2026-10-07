"""Reference optimizer runs from real torch.optim -> golden.json (checked by optimizer-race.test.ts).

`states[i]` is the optimizer's full state before update i+1 (position, plus the
momentum buffer or Adam's moments), so the test can check every single update in
isolation as well as the whole path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from simref import OPT_NAMES, config, dump, race, race_states  # noqa: E402

cfg = config("optimizer-race")
steps = cfg["golden"]["steps"]
dump({
    "start": cfg["start"],
    "runs": {
        str(lr): {name: race_states(name, lr, steps, cfg) for name in OPT_NAMES}
        for lr in cfg["golden"]["lrs"]
    },
    "outcomes": {str(lr): race(lr, cfg) for lr in cfg["golden"]["lrs"]},
})
