"""Scaffold a new lesson from templates/lesson/.

    python scripts/new_lesson.py foundations 2 "Tensors" [--slug tensors]

creates course/foundations/02-tensors/ with lesson.mdx, notebook.py,
kernel-metadata.json and runnable stub snippets. The lesson then appears in
the sidebar automatically (ordered by the 02- prefix) at /foundations/tensors/.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "templates" / "lesson"
TIERS = {"foundations": "Foundations", "intermediate": "Intermediate", "advanced": "Advanced"}
KAGGLE_TITLE_MAX = 50  # Kaggle rejects longer notebook titles


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tier", choices=TIERS)
    parser.add_argument("number", type=int, help="position within the course, e.g. 2")
    parser.add_argument("title", help='lesson title, e.g. "Tensors"')
    parser.add_argument("--slug", help="URL slug (default: from the title)")
    args = parser.parse_args()

    slug = args.slug or slugify(args.title)
    num = f"{args.number:02d}"
    dir_name = f"{num}-{slug}"
    dest = ROOT / "course" / args.tier / dir_name
    if dest.exists():
        print(f"{dest.relative_to(ROOT)} already exists.")
        return 1

    kaggle_title = f"PyTorch for Humans {num}: {args.title}"
    if len(kaggle_title) > KAGGLE_TITLE_MAX:
        print(f'Kaggle title "{kaggle_title}" is {len(kaggle_title)} chars (max {KAGGLE_TITLE_MAX}). Use a shorter title.')
        return 1
    # Kaggle derives the notebook URL from the title; keep the id slug consistent with it.
    if slugify(kaggle_title) != f"pytorch-for-humans-{num}-{slug}":
        print(f"note: Kaggle will slugify the title to '{slugify(kaggle_title)}'; "
              f"consider --slug {slugify(args.title)} so the id matches.")

    replacements = {
        "__TITLE__": args.title,
        "__SLUG__": slug,
        "__NUM__": num,
        "__TIER__": args.tier,
        "__TIER_LABEL__": TIERS[args.tier],
        "__DIR__": dir_name,
    }
    shutil.copytree(TEMPLATE, dest)
    for path in dest.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for k, v in replacements.items():
                text = text.replace(k, v)
            path.write_text(text, encoding="utf-8")

    print(f"Created {dest.relative_to(ROOT).as_posix()}/")
    print(f"  page:     /{args.tier}/{slug}/")
    print("  next:     fill in the TODOs, then run")
    print("            python scripts/test_snippets.py && python scripts/build_notebooks.py --execute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
