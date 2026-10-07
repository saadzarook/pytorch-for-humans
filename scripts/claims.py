"""Check declared narration claims against generated facts.

Every claim a sim's (or lesson's) narration makes is DECLARED next to its config,
in a "claims" list:

    {
      "id": "arrival-order",
      "says": "at lr 0.1: Momentum, then Adam, then SGD",
      "check": "order == ['Momentum', 'Adam', 'SGD']"
    }

`says` quotes (or paraphrases) the narration the claim backs. `check` is a small
Python expression evaluated against the facts.json the facts.py produced: its
top-level keys are variables. facts.py computes facts and evidence; it never
decides pass/fail. scripts/test_snippets.py runs check_claims() after every
facts.py, so a broken claim fails CI with the narration it would make wrong.
"""

from __future__ import annotations

from typing import Any

# The only names a check may use besides the facts themselves.
SAFE_FUNCTIONS = {
    "abs": abs, "all": all, "any": any, "len": len, "max": max, "min": min,
    "int": int, "float": float, "str": str,
    "round": round, "sorted": sorted, "sum": sum, "zip": zip, "range": range,
    "True": True, "False": False, "None": None,
}


def check_claims(facts: dict[str, Any], claims: list[dict[str, str]]) -> list[str]:
    """Return one message per claim that does not hold (empty list = all good)."""
    problems = []
    seen = set()
    for claim in claims:
        cid = claim.get("id", "?")
        if cid in seen:
            problems.append(f"claim '{cid}' is declared twice")
        seen.add(cid)
        if not {"id", "says", "check"} <= claim.keys():
            problems.append(f"claim '{cid}' needs id, says and check")
            continue
        try:
            # Names go in GLOBALS: generator expressions in a check have their own scope
            # and only see globals, not eval's locals.
            ok = eval(claim["check"], {"__builtins__": {}, **SAFE_FUNCTIONS, **facts})  # noqa: S307
        except Exception as err:  # a typo in a check is a failure, not a crash
            problems.append(f"claim '{cid}' could not be checked ({type(err).__name__}: {err}): {claim['check']}")
            continue
        if ok is not True:
            problems.append(f"claim '{cid}' is FALSE, so this narration would be wrong: \"{claim['says']}\"  [check: {claim['check']}]")
    return problems
