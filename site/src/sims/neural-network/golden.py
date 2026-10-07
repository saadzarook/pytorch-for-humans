"""Forward pass, BCEWithLogitsLoss gradients and one Adam step from real PyTorch -> golden.json."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from simref import dump, nn_reference  # noqa: E402

points = [[0.1, 0.2, 1], [-0.7, 0.4, 0], [0.5, -0.9, 0], [0.3, 0.3, 1], [-0.2, -0.6, 1], [0.95, 0.05, 0]]
dump({"cases": [
    nn_reference("tanh", 4, seed=1, points=points, lr=0.03),
    nn_reference("relu", 4, seed=2, points=points, lr=0.03),
    nn_reference("tanh", 3, seed=3, points=points, lr=0.1),
]})
