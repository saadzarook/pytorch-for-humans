"""Convert every lesson's jupytext source (notebook.py) into a .ipynb.

Build-time expansions (so the notebook can't drift from the website):

    # include: snippets/pytorch/torch_way.py
        A code cell containing only this line is replaced by that file, verbatim.
        The site shows and runs the same files, so both always match.
    {{facts.valley.ratio}}
        Replaced from the lesson's generated facts.json (see scripts/test_snippets.py).
    {{COURSE_URL}}
        Replaced with the COURSE_URL environment variable (the published site's base URL).

After expansion, any leftover `{{…}}` or `# include:` line is an error.

Modes:
    (default)       Write build/notebooks/<code_file>. COURSE_URL is optional here: if it's
                    unset, links that use it become plain text. Used by CI and the site.
    --kaggle        Write build/kaggle/<slug>/ (notebook + kernel-metadata.json) ready for
                    scripts/kaggle_push.py. REQUIRES KAGGLE_USERNAME and COURSE_URL, so no
                    placeholder can ever reach a pushed notebook.

Options:
    --execute       Run each built notebook top to bottom (nbclient); any error fails.
                    Executed copies go to build/executed/.
    --site          Also copy each notebook into site/public/notebooks/ for download.
    --only TEXT     Only build lessons whose Kaggle slug contains TEXT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parent.parent
COURSE = ROOT / "course"
BUILD = ROOT / "build"
SITE_NOTEBOOKS = ROOT / "site" / "public" / "notebooks"
USERNAME_PLACEHOLDER = "{KAGGLE_USERNAME}"
URL_PLACEHOLDER = "{{COURSE_URL}}"
INCLUDE_RE = re.compile(r"^#\s*include:\s*(\S+)\s*$")
FACT_RE = re.compile(r"\{\{facts\.([\w.]+)\}\}")


class BuildError(Exception):
    pass


def expand_includes(nb: nbformat.NotebookNode, lesson: Path) -> None:
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        m = INCLUDE_RE.match(cell.source.strip())
        if not m:
            continue
        path = (lesson / m.group(1)).resolve()
        if not path.is_file() or lesson not in path.parents:
            raise BuildError(f"include not found inside the lesson folder: {m.group(1)}")
        cell.source = path.read_text(encoding="utf-8").rstrip("\n")
        # Remember where it came from, so --execute can compare outputs with the website's.
        cell.metadata["pfh_include"] = m.group(1)


def normalise(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip("\n") + "\n"


def compare_with_site_outputs(nb: nbformat.NotebookNode, lesson: Path) -> list[str]:
    """Every included cell must print exactly what the website shows for that snippet."""
    problems = []
    for cell in nb.cells:
        include = cell.get("metadata", {}).get("pfh_include")
        if not include:
            continue
        snippet = Path(include)
        golden = lesson / "snippets" / "outputs" / snippet.parent.name / f"{snippet.stem}.txt"
        if not golden.exists():
            continue  # e.g. challenge files, which have no stored output
        stdout = "".join(o.get("text", "") for o in cell.get("outputs", [])
                         if o.get("output_type") == "stream" and o.get("name") == "stdout")
        if normalise(stdout) != golden.read_text(encoding="utf-8"):
            problems.append(f"{include}: notebook output differs from {golden.relative_to(ROOT).as_posix()}")
    return problems


def fill_facts(nb: nbformat.NotebookNode, lesson: Path) -> None:
    facts_path = lesson / "facts.json"
    facts = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.exists() else {}

    def lookup(m: re.Match) -> str:
        value: object = facts
        for key in m.group(1).split("."):
            if not isinstance(value, dict) or key not in value:
                raise BuildError(f"unknown fact {{{{facts.{m.group(1)}}}}} (is facts.json up to date?)")
            value = value[key]
        return str(value)

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = FACT_RE.sub(lookup, cell.source)


def fill_course_url(nb: nbformat.NotebookNode, course_url: str | None) -> None:
    link_re = re.compile(r"\[([^\]]+)\]\(" + re.escape(URL_PLACEHOLDER) + r"[^)]*\)")
    for cell in nb.cells:
        if cell.cell_type != "markdown" or URL_PLACEHOLDER not in cell.source:
            continue
        if course_url:
            cell.source = cell.source.replace(URL_PLACEHOLDER, course_url.rstrip("/") + "/")
        else:
            cell.source = link_re.sub(r"\1", cell.source)


def assert_fully_expanded(nb: nbformat.NotebookNode) -> None:
    for i, cell in enumerate(nb.cells):
        leftover = re.findall(r"\{\{[^}]*\}\}", cell.source)
        if leftover:
            raise BuildError(f"cell {i} still contains {leftover}")
        if any(INCLUDE_RE.match(line.strip()) for line in cell.source.splitlines()):
            raise BuildError(f"cell {i} has an `# include:` line that isn't alone in its cell")


def build_one(src: Path, args: argparse.Namespace, course_url: str | None, username: str) -> bool:
    lesson = src.parent.resolve()
    meta_path = lesson / "kernel-metadata.json"
    if not meta_path.exists():
        print(f"  skip  {src.relative_to(ROOT)} (no kernel-metadata.json)")
        return True
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    slug = meta["id"].split("/", 1)[1]
    if args.only and args.only not in slug:
        return True

    try:
        nb = jupytext.read(src)
        expand_includes(nb, lesson)
        fill_facts(nb, lesson)
        fill_course_url(nb, course_url)
        assert_fully_expanded(nb)
    except BuildError as err:
        print(f"  FAIL  {slug}: {err}")
        return False
    nb.metadata.pop("jupytext", None)  # jupytext bookkeeping isn't useful on Kaggle

    if args.kaggle:
        out_dir = BUILD / "kaggle" / slug
        meta["id"] = meta["id"].replace(USERNAME_PLACEHOLDER, username)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    else:
        out_dir = BUILD / "notebooks"
        out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / meta["code_file"]
    nbformat.write(nb, nb_path)
    print(f"  built {nb_path.relative_to(ROOT).as_posix()}")

    if args.site:
        SITE_NOTEBOOKS.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(nb_path, SITE_NOTEBOOKS / meta["code_file"])
        print(f"  site  site/public/notebooks/{meta['code_file']}")

    if args.execute:
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError

        executed = nbformat.read(nb_path, as_version=4)
        start = time.time()
        client = NotebookClient(executed, timeout=args.timeout, kernel_name="python3",
                                resources={"metadata": {"path": str(lesson)}})
        try:
            client.execute()
        except CellExecutionError as err:
            print(f"  FAIL  {slug}: a cell raised an error\n{err}")
            return False
        exec_dir = BUILD / "executed"
        exec_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(executed, exec_dir / meta["code_file"])
        problems = compare_with_site_outputs(executed, lesson)
        if problems:
            print(f"  FAIL  {slug}: notebook and website disagree\n    " + "\n    ".join(problems))
            return False
        print(f"  ran   {slug} top to bottom in {time.time() - start:.0f}s; included cells match the website's outputs")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kaggle", action="store_true", help="build Kaggle push folders (needs KAGGLE_USERNAME + COURSE_URL)")
    parser.add_argument("--execute", action="store_true", help="execute every notebook; fail on any error")
    parser.add_argument("--site", action="store_true", help="copy .ipynb files into site/public/notebooks")
    parser.add_argument("--only", help="only build lessons whose Kaggle slug contains this text")
    parser.add_argument("--timeout", type=int, default=600, help="per-cell timeout in seconds (default 600)")
    args = parser.parse_args()

    course_url = os.environ.get("COURSE_URL", "").strip() or None
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if args.kaggle:
        missing = [name for name, value in [("KAGGLE_USERNAME", username), ("COURSE_URL", course_url)] if not value]
        if missing:
            print(f"--kaggle needs {' and '.join(missing)} set. Example:\n"
                  "  KAGGLE_USERNAME=you COURSE_URL=https://you.github.io/repo/ "
                  "python scripts/build_notebooks.py --kaggle")
            return 1
        if not re.match(r"^https?://", course_url):
            print(f"COURSE_URL must be a full http(s) URL, got {course_url!r}")
            return 1

    sources = sorted(COURSE.glob("**/notebook.py"))
    if not sources:
        print("No course/**/notebook.py files found.")
        return 1
    ok = all([build_one(src, args, course_url, username) for src in sources])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
