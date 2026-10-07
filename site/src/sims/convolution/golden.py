"""F.conv2d outputs for every preset kernel -> golden.json (checked by convolution.test.ts)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from simref import config, conv_output, dump  # noqa: E402

cfg = config("convolution")
dump({name: conv_output(k, cfg).flatten().tolist() for name, k in cfg["presets"].items()})
