"""The S22 results cited in the BCS project proposal must never change.

Pins the SHA-256 of every recorded experiment report (and the methodology notes that quote
them). New runs belong in a separate output directory, never in docs/experiments/.
"""
import glob
import hashlib

import pytest

PINNED = {
    "docs/experiments/confidence_2026-10-11_adapter_financial_v2.json": "00d35b69039826ece8d613a2e6d4312f2ba00c5f301948e2b0b7a501bd1f44d3",
    "docs/experiments/expA_2026-10-11_B0.json": "9faaf06aaa171b9df406fdd43aa916e3745c4dfa907f312eb62deb4f494ef58b",
    "docs/experiments/expA_2026-10-11_Hybrid.json": "bcacef1263d8ef1934b26fc0e6b73b5ba3b0743a49283b8798d00214cb3bba58",
    "docs/experiments/expA_2026-10-11_T5-orig.json": "f3794570fba5d550809199f7c591203d31143d5bff49b8097c976b4f56cb607f",
    "docs/experiments/expA_2026-10-11_summary.json": "8ebd88e48be13cdb387d473086d3fad3fa5f3e6570900a8da0bad7995d0d6b61",
    "docs/experiments/expA_b0_repeats/run2/expA_2026-10-11_B0.json": "795b42c37e5412c0db06bb1f8b529d85efdddc6d3dd0d3d897781acad80fec98",
    "docs/experiments/expA_b0_repeats/run2/expA_2026-10-11_summary.json": "df68dd0b4fcdea0af02728ada3e01d012608c5e000d4838d29534c688be2d323",
    "docs/experiments/expA_b0_repeats/run3/expA_2026-10-11_B0.json": "9b619d01224e4a16788784ab349549c0f69ae64cc37cbd79edf89f6d4f4613cf",
    "docs/experiments/expA_b0_repeats/run3/expA_2026-10-11_summary.json": "49ab51d66ad713740ab6a93907f1436d7d1db458424650e7aa27d02876bc44e5",
    "docs/experiments/expA_hybrid_gate/expA_2026-10-11_Hybrid.json": "5496419ad38254513486b31a152de80a3601c658b742ed2876bdd0ffa4156ec9",
    "docs/experiments/expA_hybrid_gate/expA_2026-10-11_summary.json": "e1b8dea74ced19a61e00ba44fcbe1e9fe3ecd969ee0a1bffc5779da845cdf81c",
    "docs/experiments/expB_2026-10-11_T5-orig.json": "0add571cbcce9baa7aaa155849154be2da1ac505dec4e6e4a277821dd75d31c7",
    "docs/experiments/expB_2026-10-11_T5-retrained.json": "38da0a0fc19bb3ee33163c47c5ba40b72d98dd231066bbd8b9f5b6d873c0bde2",
    "docs/experiments/expB_2026-10-11_summary.json": "d1966e345122f28ab6bbef6895b4c7b3eb0d4908d73403d2c2fd22f48049ec15",
    "docs/s22-methodology-notes.md": "4b4bed80d983036a151578085ae1e0e3b1f223e0a76809db47f73d45ad431a49",
}


@pytest.mark.parametrize("path", sorted(PINNED))
def test_recorded_file_unchanged(path):
    with open(path, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == PINNED[path], f"{path} changed"


def test_no_files_added_to_recorded_experiments():
    recorded = set(glob.glob("docs/experiments/**/*.json", recursive=True))
    assert recorded == {p for p in PINNED if p.startswith("docs/experiments/")}
