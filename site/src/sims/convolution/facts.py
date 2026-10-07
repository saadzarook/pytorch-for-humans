"""Claims the convolution narration makes, checked with F.conv2d -> facts.json."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from simref import config, conv_image, conv_output, dump, require  # noqa: E402

cfg = config("convolution")
img = conv_image(cfg)
S = img.shape[0]
out = {n: conv_output(k, cfg) for n, k in cfg["presets"].items()}
O = out["vedge"].shape[0]
require(O == S - 3 + 1, "output size should be S - 3 + 1")

bar = next(i for i, row in enumerate(cfg["imageRows"]) if row.count("#") >= 6)  # the crossbar's input row
r = bar - 1  # output row whose window is centred on the crossbar

# vedge: "the middle of the crossbar stayed 0: this kernel ignores horizontal lines".
mid = out["vedge"][r, 3:O - 3]
require(bool((mid == 0).all()), f"vedge should be 0 along the middle of the crossbar, got {mid.tolist()}")
# vedge: "amber on one side of each stroke and teal on the other": every row that
# responds at all has both signs.
for i in range(O):
    row = out["vedge"][i]
    if row.abs().sum() > 0:
        require(bool((row > 0).any() and (row < 0).any()), f"vedge row {i} should have both signs")

# hedge: "the crossbar lit up strongly, amber along its top edge and teal along its
# bottom edge, while the slanted sides respond only weakly".
top, bottom = out["hedge"][r - 1], out["hedge"][r + 1]
require(bool((top[2:O - 2] > 0).all()), "hedge should be positive (amber) along the top of the crossbar")
require(bool((bottom[2:O - 2] < 0).all()), "hedge should be negative (teal) along the bottom of the crossbar")
bar_peak = max(top.abs().max().item(), bottom.abs().max().item())
others = out["hedge"].clone()
others[r - 1] = 0
others[r + 1] = 0
require(others.abs().max().item() < bar_peak, "the crossbar edges should be the strongest hedge responses")

# outline: "In a flat area (all ink or all blank) the weights cancel out to 0".
require(abs(sum(cfg["presets"]["outline"])) < 1e-12, "outline weights should sum to 0")
# blur: "each one counts 1/9".
require(all(abs(v - 1 / 9) < 1e-15 for v in cfg["presets"]["blur"]), "blur weights should all be 1/9")
# identity: "the output is the input with its border trimmed off".
require(bool((out["identity"] == img[1:-1, 1:-1]).all()), "identity should trim the border")

dump({"imageSize": S, "outSize": O, "windows": O * O, "crossbarPeak": int(bar_peak)})
