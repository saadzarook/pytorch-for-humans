"""Run every lesson code snippet with CPython, and keep generated outputs honest.

Snippet layout (per lesson):

    snippets/browser/*.py   Shown in <PyRunner> cells and run in the browser by
                            Pyodide. Must only import NumPy + the standard library.
    snippets/pytorch/*.py   Shown on the page as static code. Need PyTorch.
    snippets/outputs/<folder>/<name>.txt
                            GENERATED: the stdout of each snippet. Lessons import
                            these as their "Output" blocks, so they never drift.
    facts.py / facts.json   Optional. facts.py derives the numbers the prose quotes
                            and checks the lesson's claims; facts.json is its
                            GENERATED output, imported by the lesson and notebook.

Sims (site/src/sims/<sim>/, see docs/SIM_STANDARD.md):

    golden.py / golden.json Reference values from real PyTorch; Vitest checks the
                            sim's TypeScript maths matches them.
    facts.py / facts.json   The numbers the sim's narration relies on (plus evidence).
    config.json "claims"    Declared narration claims, each a `check` expression over
                            facts.json, evaluated here (scripts/claims.py). Lessons
                            declare theirs in data.json. facts.py never decides pass/fail.

Every *.json above is GENERATED from its script and committed; this runner fails
if any is stale. Sims run first, because lesson facts may cross-check them.

Challenge convention (browser/):

    challenge_starter.py    What the reader starts with. Must run without errors
                            and must NOT pass the check.
    challenge_solution.py   A reference answer. Must pass the check.
    challenge_check.py      Runs after the reader's code in the same namespace and
                            prints a line starting with ✅ on success.

Usage:
    python scripts/test_snippets.py            # run all; fail if any generated file is stale
    python scripts/test_snippets.py --update   # run all and rewrite the generated files
    python scripts/test_snippets.py --no-torch # skip anything needing PyTorch (pytorch snippets, sims)
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path

from claims import check_claims

ROOT = Path(__file__).resolve().parent.parent
COURSE = ROOT / "course"
SIMS = ROOT / "site" / "src" / "sims"
TIMEOUT_S = 180

# Modules a browser snippet may import. Pyodide ships the full standard library,
# but we keep the list tight so cells stay fast to load.
BROWSER_ALLOWED = {
    "numpy", "math", "random", "statistics", "itertools", "functools",
    "collections", "time", "__future__",
}


def run_code(code: str, cwd: Path) -> tuple[int, str, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    try:
        proc = subprocess.run(
            [sys.executable, "-"], input=code, capture_output=True, text=True,
            encoding="utf-8", cwd=cwd, env=env, timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return 1, "", f"timed out after {TIMEOUT_S}s"
    return proc.returncode, normalise(proc.stdout), proc.stderr


def run_script(path: Path) -> tuple[int, str, str]:
    """Run a generator script (facts.py / golden.py) by path, from its own folder."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        proc = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True,
            encoding="utf-8", cwd=path.parent, env=env, timeout=TIMEOUT_S * 3,
        )
    except subprocess.TimeoutExpired:
        return 1, "", f"timed out after {TIMEOUT_S * 3}s"
    return proc.returncode, normalise(proc.stdout), proc.stderr


def normalise(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip("\n") + "\n"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


class Runner:
    def __init__(self, update: bool):
        self.update = update
        self.passed = 0
        self.failures: list[str] = []
        self.updated: list[str] = []

    def result(self, ok: bool, label: str, detail: str = "") -> None:
        if ok:
            self.passed += 1
            print(f"  ok    {label}")
        else:
            self.failures.append(f"{label}\n{detail}".rstrip())
            print(f"  FAIL  {label}")

    def golden(self, generated: str, path: Path, label: str) -> None:
        """Compare `generated` with the committed file at `path` (or rewrite it with --update)."""
        rel = path.relative_to(ROOT).as_posix()
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == generated:
            self.result(True, f"{label} (matches {path.name})")
            return
        if self.update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(generated, encoding="utf-8", newline="\n")
            self.updated.append(rel)
            self.result(True, f"{label} (updated {rel})")
            return
        if current is None:
            self.result(False, label, f"{rel} is missing. Run: python scripts/test_snippets.py --update")
            return
        diff = "".join(difflib.unified_diff(
            current.splitlines(keepends=True), generated.splitlines(keepends=True),
            fromfile=f"{rel} (committed)", tofile=f"{rel} (actual run)",
        ))
        self.result(False, label, f"Generated file is stale. Run: python scripts/test_snippets.py --update\n{diff}")

    def snippet(self, path: Path, lesson: Path, check_code: str | None) -> None:
        label = path.relative_to(ROOT).as_posix()
        code = path.read_text(encoding="utf-8")
        is_challenge = path.name.startswith("challenge_")
        if is_challenge and check_code is not None:
            code = code + "\n" + check_code
            label += " + check"
        rc, out, err = run_code(code, path.parent)
        if rc != 0:
            self.result(False, label, f"exit code {rc}\n{out}{err}")
            return
        if path.name == "challenge_solution.py" and "✅" not in out:
            self.result(False, label, f"the reference solution did not pass the check\n{out}")
            return
        if path.name == "challenge_starter.py" and "✅" in out:
            self.result(False, label, f"the unsolved starter already passes the check\n{out}")
            return
        if is_challenge:
            self.result(True, label)
        else:
            out_path = lesson / "snippets" / "outputs" / path.parent.name / f"{path.stem}.txt"
            self.golden(out, out_path, label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="rewrite generated outputs and facts.json")
    parser.add_argument("--no-torch", action="store_true", help="skip snippets/pytorch")
    args = parser.parse_args()
    runner = Runner(update=args.update)

    def generated(script: Path) -> None:
        rc, out, err = run_script(script)
        label = script.relative_to(ROOT).as_posix()
        if rc != 0:
            runner.result(False, label, f"{out}{err}")
            return
        runner.golden(out, script.with_suffix(".json"), label)
        if script.name == "facts.py":
            # The narration's claims are declared next to the facts: a sim's config.json,
            # or a lesson's data.json. facts.py computes; the claims decide.
            decl = script.parent / ("config.json" if script.parent.parent == SIMS else "data.json")
            claims = json.loads(decl.read_text(encoding="utf-8")).get("claims", []) if decl.exists() else []
            if not claims:
                runner.result(False, f"{label} claims", f"{decl.relative_to(ROOT).as_posix()} declares no claims")
                return
            problems = check_claims(json.loads(out), claims)
            runner.result(not problems, f"{label} claims ({len(claims)} declared in {decl.name})", "\n".join(problems))

    # Sims first: lesson facts may cross-check a sim's facts.json.
    if args.no_torch:
        print("\n[sims] skipped (--no-torch)")
    else:
        for sim in sorted(p for p in SIMS.iterdir() if p.is_dir()):
            scripts = [sim / n for n in ("golden.py", "facts.py") if (sim / n).exists()]
            if scripts:
                print(f"\n[{sim.relative_to(ROOT).as_posix()}]")
                for script in scripts:
                    generated(script)

    lessons = sorted({p.parent.parent for p in COURSE.glob("**/snippets/browser")}
                     | {p.parent for p in COURSE.glob("**/facts.py")})
    for lesson in lessons:
        print(f"\n[{lesson.relative_to(ROOT).as_posix()}]")

        browser_dir = lesson / "snippets" / "browser"
        check_file = browser_dir / "challenge_check.py"
        check_code = check_file.read_text(encoding="utf-8") if check_file.exists() else None
        for path in sorted(browser_dir.glob("*.py")):
            bad = imported_modules(path) - BROWSER_ALLOWED
            if bad:
                runner.result(False, path.relative_to(ROOT).as_posix(),
                              f"imports not allowed in browser snippets: {sorted(bad)}")
            elif path.name != "challenge_check.py":
                runner.snippet(path, lesson, check_code)

        if not args.no_torch:
            for path in sorted((lesson / "snippets" / "pytorch").glob("*.py")):
                runner.snippet(path, lesson, None)

        # Last: lesson facts may cross-check snippet outputs.
        if (lesson / "facts.py").exists():
            generated(lesson / "facts.py")

    print(f"\n{runner.passed} passed, {len(runner.failures)} failed")
    if runner.updated:
        print("updated:\n  " + "\n  ".join(runner.updated))
    for f in runner.failures:
        print("\n" + "-" * 60 + "\n" + f)
    return 1 if runner.failures else 0


if __name__ == "__main__":
    sys.exit(main())
