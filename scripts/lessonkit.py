"""Helpers for a lesson's facts.py (imported by them; not run directly).

A lesson's facts.py derives every number the lesson prose quotes (plus the
evidence its claims need) and prints the results as JSON. The claims themselves
are declared in the lesson's data.json and checked by scripts/claims.py. scripts/test_snippets.py
runs it and compares the JSON with the committed facts.json, which the lesson
MDX and the notebook build both read. So a quoted number can't silently drift
away from the code that produces it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def literal_arrays(path: Path) -> dict[str, list[float]]:
    """Every `name = np.array([...])` / `torch.tensor([...])` / `name = [...]` with a literal list, by name."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, list[float]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        value = node.value
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        if isinstance(value, ast.List):
            try:
                found[node.targets[0].id] = [float(v) for v in ast.literal_eval(value)]
            except (ValueError, TypeError):
                pass
    return found


def check_copies(dataset: dict, files: list[Path]) -> list[str]:
    """Check every copy of `dataset` (by its xName / yName variable names) in `files`.

    Returns a list of problems. A file that defines neither name is skipped; a file
    that defines one of them must match exactly.
    """
    problems = []
    for path in files:
        arrays = literal_arrays(path)
        for key in ("x", "y"):
            name = dataset[f"{key}Name"]
            if name in arrays and arrays[name] != [float(v) for v in dataset[key]]:
                problems.append(f"{path.name}: `{name}` differs from data.json")
    return problems
