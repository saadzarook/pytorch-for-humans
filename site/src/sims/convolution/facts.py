"""Facts the convolution narration relies on, computed with F.conv2d -> facts.json.

This script only COMPUTES. The claims are declared in config.json and checked by
scripts/test_snippets.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from simref import config, conv_image, conv_output, dump  # noqa: E402

cfg = config("convolution")
img = conv_image(cfg)
S = img.shape[0]
out = {n: conv_output(k, cfg) for n, k in cfg["presets"].items()}
O = out["vedge"].shape[0]

bar = next(i for i, row in enumerate(cfg["imageRows"]) if row.count("#") >= 6)  # the crossbar's input row
r = bar - 1  # output row whose window is centred on the crossbar
vedge, hedge = out["vedge"], out["hedge"]
others = hedge.clone()
others[r - 1] = 0
others[r + 1] = 0

dump({
    "imageSize": S,
    "outSize": O,
    "windows": O * O,
    "crossbarPeak": int(max(hedge[r - 1].abs().max().item(), hedge[r + 1].abs().max().item())),
    # Evidence for the claims in config.json:
    "vedgeCrossbarMiddle": vedge[r, 3:O - 3].tolist(),
    "vedgeRowsWithBothSigns": [bool((row > 0).any() and (row < 0).any()) for row in vedge if row.abs().sum() > 0],
    "hedgeTopEdge": hedge[r - 1, 2:O - 2].tolist(),
    "hedgeBottomEdge": hedge[r + 1, 2:O - 2].tolist(),
    "hedgeOtherPeak": others.abs().max().item(),
    "outlineWeightSum": sum(cfg["presets"]["outline"]),
    "blurWeights": cfg["presets"]["blur"],
    "identityTrimsBorder": bool((out["identity"] == img[1:-1, 1:-1]).all()),
})
