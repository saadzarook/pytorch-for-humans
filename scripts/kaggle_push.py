"""Push built notebooks to Kaggle with the official Kaggle CLI.

Prerequisites (you do these once; this script never reads or stores credentials):
  1. pip install -r requirements-kaggle.txt
  2. Authenticate the Kaggle CLI in any way it supports, e.g. `kaggle auth login`,
     the KAGGLE_API_TOKEN environment variable, or ~/.kaggle/kaggle.json.
     See https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md
  3. Build the Kaggle folders (fails unless both variables are set):
         KAGGLE_USERNAME=<you> COURSE_URL=<site url> python scripts/build_notebooks.py --kaggle

Then:
    python scripts/kaggle_push.py --dry-run     # show what would be pushed
    python scripts/kaggle_push.py               # push every built notebook
    python scripts/kaggle_push.py --only 01-gradient-descent

New notebooks are created PRIVATE ("is_private": "true" in kernel-metadata.json).
Kaggle runs each notebook after the push; check progress with
`kaggle kernels status <user>/<slug>`, then make it public on Kaggle when you're happy.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_BUILD = ROOT / "build" / "kaggle"
# Anything that should have been filled in at build time. None may reach Kaggle.
LEFTOVER_RE = re.compile(r"\{\{[^}]*\}\}|\{KAGGLE_USERNAME\}|^#\s*include:", re.M)


def leftovers(folder: Path) -> list[str]:
    found = []
    for path in [folder / "kernel-metadata.json", *folder.glob("*.ipynb")]:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".ipynb":
            # Check the cell sources, not the JSON escaping around them.
            text = "\n".join("".join(c["source"]) for c in json.loads(text)["cells"])
        found += [f"{path.name}: {m.group(0)!r}" for m in LEFTOVER_RE.finditer(text)]
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--only", help="only push notebooks whose slug contains this text")
    args = parser.parse_args()

    dirs = sorted(p.parent for p in KAGGLE_BUILD.glob("*/kernel-metadata.json"))
    if args.only:
        dirs = [d for d in dirs if args.only in d.name]
    if not dirs:
        print("Nothing to push. Run scripts/build_notebooks.py --kaggle first.")
        return 1

    if not args.dry_run and shutil.which("kaggle") is None:
        print("The `kaggle` command isn't installed. Run: pip install -r requirements-kaggle.txt")
        return 1

    failed = 0
    for d in dirs:
        meta = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))
        bad = leftovers(d)
        if bad:
            print(f"  skip  {d.name}: unfilled placeholders {bad}. Rebuild with --kaggle.")
            failed += 1
            continue
        cmd = ["kaggle", "kernels", "push", "-p", str(d)]
        print(f"  push  {meta['id']}  ({'private' if meta.get('is_private') == 'true' else 'PUBLIC'})")
        if args.dry_run:
            print("        " + " ".join(cmd))
            continue
        if subprocess.run(cmd).returncode != 0:
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
