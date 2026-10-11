"""Leakage + validity checks for the held-out set.

Usage (from the repo root):
    python scripts/check_holdout.py training_data/holdout_financial_test.json
"""
import difflib
import glob
import json
import re
import sys

sys.path.insert(0, ".")
from src.schema_validator import validate_node_constraints

holdout = json.load(open(sys.argv[1]))["examples"]


def norm(s):
    return re.sub(r"[^a-z0-9<>= ]", "", s.lower()).strip()


train = []
for path in glob.glob("training_data/*.json") + ["training_data_collected.json"]:
    if "holdout" in path:
        continue
    data = json.load(open(path))
    rows = data.get("examples", data) if isinstance(data, dict) else data
    for r in rows if isinstance(rows, list) else []:
        if isinstance(r, dict) and "input" in r:
            train.append((norm(r["input"]), r["input"], path))

problems = 0
for e in holdout:
    errs = validate_node_constraints({"type": e["field_type"], **e["output"]}, e["id"])
    if errs:
        problems += 1
        print("INVALID", e["id"], errs)
    h = norm(e["input"])
    best = max(train, key=lambda t: difflib.SequenceMatcher(None, h, t[0]).ratio())
    ratio = difflib.SequenceMatcher(None, h, best[0]).ratio()
    if ratio >= 0.85:
        problems += 1
        print(f"LEAK {ratio:.2f} {e['id']}: {e['input']!r} ~ {best[1]!r} ({best[2]})")
    elif ratio >= 0.75:
        print(f"near {ratio:.2f} {e['id']}: {e['input']!r} ~ {best[1]!r}")

ids = [e["id"] for e in holdout]
inputs = [norm(e["input"]) for e in holdout]
assert len(set(ids)) == len(ids), "duplicate ids"
assert len(set(inputs)) == len(inputs), "duplicate inputs"
print(f"{len(holdout)} cases vs {len(train)} training inputs; {problems} problems")
sys.exit(1 if problems else 0)
