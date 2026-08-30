# S22 Methodology Notes — Scientific-Method Framing for the EPA Report

> **Status:** working notes, not the final report text.
> **Purpose:** prepare the *Systematic Methodology* section of the AIL7 EPA proposal so it
> evidences skill **S22** — *"Apply scientific methods in a systematic process through
> experimental design, exploratory data analysis and hypothesis testing to facilitate business
> decision making."*
>
> These notes (a) record an honest audit of what evidence the project currently holds, (b) give a
> drop-in scientific-method structure for the methodology section, and (c) specify the two
> experiments that need to be run to turn the current prose claims into measured evidence.

---

## 1. S22 mapping — what has to be shown, and where it comes from

| S22 clause | What the assessor needs to see | Where it comes from in this project |
| --- | --- | --- |
| **Experimental design** | A stated hypothesis and null position; defined test conditions; a baseline; a held-out dataset; success metrics fixed *before* seeing results | Experiment A (§4). The five metrics already implemented in `src/model_evaluator.py`; a new held-out test set |
| **Exploratory data analysis** | Analysis of the data that *changed a decision*, not just described it | Constraint-type distribution of the training data → the `>=` vs `<=` imbalance → the 278 supplemental examples (§3.4, §4 Experiment B) |
| **Hypothesis testing** | Results compared against the pre-stated thresholds; a conclusion that the hypothesis was supported, challenged, or partially supported | §3.5, filled in once Experiments A and B run |
| **Facilitate business decision making** | A concrete decision the experiment informs, with the cost/benefit at stake | "Deploy hybrid vs prompt-only LLM" and "ship the retrained adapter or not" — both tied to the API-cost and latency argument in `docs/next_steps.md` |

---

## 2. Current-state evidence audit (be honest about this in the report)

The DLC feedback treats **"83.3% T5 success / 16.7% fallback"** as the project's experimental
results. **They are not measured results.** Before the methodology section can frame them as
evidence, this gap has to be closed or disclosed.

| Claim / artifact | Reality | Reference |
| --- | --- | --- |
| "T5 achieves 83% success rate" / "15 (83.3%) … 3 (16.7%)" | Illustrative example written directly into the docs; no evaluation run, dataset, or report produced it. `15/18` and `3/18` are a constructed example of the `HybridStats` print format | `docs/evaluation_metrics.md` "Operational Metrics" → "Example Output"; `docs/next_steps.md` lines 27, 50, 204; introduced in docs-only commit `23eb7ff` |
| Evaluation CLI | `python -m src.evaluate --model t5` and `--model llm` are stubs that print "not yet implemented" and exit 1. Only `--model mock` runs | `src/evaluate.py:90-99` |
| Held-out test set | None exists. The eval recipes in the docs run the model over the **training file** (`financial_domain.json` / `_combined.json`), so scores include memorised examples | `docs/training_workflow.md:364-392`, `README.md:210-224` |
| T5-vs-LLM baseline comparison | `compare_models()` is implemented but only ever exercised with mock interpreters in tests | `src/model_evaluator.py:352-378`; `tests/test_model_evaluator.py:326-353` |
| `t5_failures.json` | A **real** run, but only 3 calls and the adapter did not load (`t5_successes: 0`, `fallback_rate: 66.7%`). Useful as a qualitative record of the validator catching a string-vs-number error, not as a success-rate measurement | `t5_failures.json`; `src/hybrid_interpreter.py:340-353` |
| `audit.json` | A **real** enrichment run of `examples/financial_schema.json` (28 fields: ~14 `t5`, 3 `llm`, rest `no_rule` / `no_match`). The closest thing to real operational evidence, and roughly consistent with "~80% T5", but it is one schema on one run, not a designed experiment | `audit.json`; `src/enricher.py` audit log |
| `enriched_financial.json` vs `t5_financial_enriched.json` | LLM-path vs T5-path output of the *same* schema. They differ on exactly one field (`customer_id`). A curated demo, not a metric | `git diff` of the two files |
| EDA-driven supplemental data | `financial_domain_supplemental.json` (278 contrastive `>=`/`<=` examples) and `financial_domain_combined.json` (787) were created from a gap analysis but **never used to retrain**. The shipped adapter is still the original 509-example one | `adapters/adapter_financial/final_adapter/adapter_config.json`; commit `d640b3b` |

**Implication for the report:** either (preferred) run Experiments A and B below and report the
real numbers, or explicitly state that 83.3% is an illustrative target and not a measurement.
Presenting the illustrative figure as a result would be a serious integrity problem in an EPA.

---

## 3. Scientific-method framing for the methodology section (drop-in structure)

### 3.1 Business question and decision at stake

Organisations hold field-level business rules as free text in a data dictionary and hold JSON
Schemas that lack validation constraints. Translating each rule into the right JSON Schema keyword
by hand is slow (`docs/next_steps.md` estimates 2–4 hours per schema) and error-prone.

**Decision:** which interpretation engine to build the tool around —
1. a prompt-only hosted LLM (simple, no training, but paid per call and ~0.5–2 s latency), or
2. a locally fine-tuned T5 + LoRA model guarded by a deterministic validator, falling back to the
   LLM only when the local model fails (cheaper and faster if the local model handles most rules).

At 10,000 rules/day the documented cost spread is roughly \$50–200/month (100% LLM) versus
\$2–40/month (80–95% handled locally) — `docs/next_steps.md:196-213`. That is the business case
the experiment has to support or refute.

### 3.2 Hypotheses

- **H1 (alternative):** the hybrid engine (fine-tuned T5 + deterministic validation + LLM
  fallback) processes routine constraint-enrichment rules **faster** and **at least as reliably**
  as a prompt-only LLM, at materially lower cost per rule.
- **H0 (null):** the hybrid engine is **no more reliable** than the prompt-only LLM (equal or
  worse on correctness), so the added complexity of training and hosting a local model is not
  justified — the tool should just call the LLM.

### 3.3 What failure looks like (decision thresholds, fixed before running)

The hybrid approach is rejected (H0 retained) if **any** of the following hold on the held-out
test set:

| Dimension | Reject-H1 condition |
| --- | --- |
| Correctness | Hybrid semantic-match rate is more than **5 percentage points below** the prompt-only LLM's |
| Safety | End-to-end schema-valid rate on enriched output is **below 100%** (a wrong-typed constraint reaching the output is a hard failure) |
| Latency | Hybrid median latency is **not at least 5× faster** than the LLM baseline for rules the local model handles |
| Coverage | The local model handles **fewer than 60%** of rules (fallback rate > 40%), eroding the cost case |

These thresholds are judgement calls and should be stated as such — the point for S22 is that
they are set *before* the results are seen, so the results can *decide* rather than merely
*describe*.

### 3.4 Experimental design

**Datasets**

| Set | Source | Use | Disjoint from |
| --- | --- | --- | --- |
| Training | `training_data/financial_domain.json` (509) — original shipped adapter | Train the baseline adapter | Test |
| Training+ | `training_data/financial_domain_combined.json` (787 = 509 + 278 supplemental) | Train the retrained adapter (Experiment B) | Test |
| Validation | 10% ordered/`seed=42` split inside the training file (`src/train_adapter*.py`) | `eval_loss`, early stopping | — |
| **Held-out test (NEW)** | Hand-authored, ~60–100 rules, none copied from any training file | Score all models | Both training sets |

The held-out test set is the missing piece. Requirements:
- stratified by constraint type in roughly the production proportions
  (`pattern` ~22%, `enum` ~17%, `{minimum,maximum}` ~15%, `minimum` ~14%, `maximum` ~10%,
  `maxLength` ~7%, `format` ~6%, `minLength` ~3%, `exclusiveMinimum` ~3%);
- includes a dedicated block of contrastive `>=` / `<=` rules and "positive / non-negative" rules
  so Experiment B's effect is detectable;
- each case is `{"input": <rule text>, "output": <expected constraint dict>, "field_type": <type>,
  "constraint_type": <label>}` — the shape `load_evaluation_dataset()` already expects;
- phrasings deliberately *not* lifted from the training files (paraphrase, don't copy).

**Baseline:** prompt-only DeepSeek-V3 via OpenRouter (`src/llm_interpreter.py`, `temperature=0.0`,
the existing 3-example prompt). This is the "just call the LLM" option H0 defends.

**Conditions compared**

| Condition | Engine |
| --- | --- |
| B0 | Prompt-only LLM (baseline) |
| T5-orig | Shipped LoRA adapter alone (`adapters/adapter_financial/final_adapter`) |
| T5-retrained | LoRA adapter retrained on the 787-example combined set |
| Hybrid | T5-retrained + `validate_node_constraints` gate + LLM fallback |

**Metrics** (all already implemented in `src/model_evaluator.py`, run via `EvaluationSuite`):

| Metric | Definition | Target |
| --- | --- | --- |
| JSON-correctness rate | Output parses as a JSON object | ≥ 99% |
| Schema-valid rate | Constraints legal for the field type (`validate_node_constraints`) | 100% for Hybrid; report as-is for T5-only |
| Exact-match rate | Output dict `==` expected dict | report; not the primary bar |
| Semantic-match rate | Same keys/values, order-insensitive | primary correctness metric; within 5pp of B0 |
| Latency avg / p99 (ms) | Wall-clock per `interpret()` call | Hybrid (local path) ≥ 5× faster than B0 |
| Fallback rate | Share of rules the local model could not handle | < 40% (target < 20%) |

**Procedure**

1. Author and commit the held-out test set (`training_data/holdout_financial_test.json`).
2. Wrap the LLM as an `interpret(business_rule)` object (thin adapter over
   `interpret_business_rule`) so it satisfies the `EvaluationSuite` protocol.
3. Run `EvaluationSuite` for B0 and T5-orig; save reports with `save_report()`.
4. Retrain the adapter (Experiment B, §4); run T5-retrained and Hybrid.
5. Run `compare_models(B0, Hybrid, test_cases, ...)` for the head-to-head table.
6. Use `EvaluationResults.group_by_constraint_type()` for the per-constraint breakdown
   (this is where the `>=`/`<=` story shows up).
7. Commit all report JSONs under `docs/experiments/` with the git SHA and date.

### 3.5 EDA summary (the part the DLC flagged as descriptive)

**Finding.** Counting the output shape of every example in `training_data/financial_domain.json`
(509 examples):

| Output shape | Count |
| --- | --- |
| `pattern` only | 113 |
| `enum` only | 85 |
| `{minimum, maximum}` | 75 |
| `minimum` only | 73 |
| `maximum` only | 52 |
| `maxLength` only | 37 |
| `format` only | 29 |
| `minLength` only | 17 |
| `exclusiveMinimum` only | 14 |
| `{minLength, maxLength}` | 14 |

Two problems are visible:
1. **Bare `minimum` / `maximum` are under-represented** relative to `pattern` / `enum`, and the
   dataset has **no "at most" phrasings at all** (recorded literally in
   `financial_domain_supplemental.json` line 99: *"'AT MOST' PATTERNS (currently 0 in training
   data)"*).
2. **No contrastive pairs** — the same field with a `>=` rule and a `<=` rule — so the model has
   little signal to distinguish the two operators. Observed symptom: T5 emitting `{"minimum": 1,
   "maximum": 50}` for a "between 1 and 50 characters" string rule (`t5_failures.json`), and
   `{"minimum": 100}` for "must be positive" in `enriched_financial_schema.json` (a hallucinated
   bound; should be `exclusiveMinimum: 0`).

**Decision driven by the finding.** Build a targeted supplemental set —
`financial_domain_supplemental.json`, 278 examples — organised (see its `_comment` headers) into:
contrastive `>=` vs `<=` pairs for age / score / value / count / amount / rate / price;
"at most" / "no more than" / "no less than" phrasings; more diverse `>=` values (not just
0/1/100); single-constraint examples to discourage over-generation. Merge into
`financial_domain_combined.json` (787).

**This EDA is currently unfinished as science:** the intervention was designed but never trained
in or measured. Experiment B closes that loop.

**Data-quality checks applied** (from `docs/training_workflow.md:155-164`): every example has
`input` + `output`; outputs are valid JSON Schema constraint objects; contrastive pairs now
present; diverse values used; multiple phrasings per pattern. A residual risk — train/test leakage
— is the reason the held-out test set must be hand-authored rather than sampled from the training
files.

### 3.6 Results as evidence (template — fill after the runs)

For each of B0, T5-orig, T5-retrained, Hybrid, report the metric table and then answer:

- Did Hybrid meet **every** reject-H1 threshold in §3.3? → H1 supported / not supported.
- Per-constraint breakdown: did the retrained adapter's semantic-match rate on the contrastive
  `>=`/`<=` block improve over T5-orig? → EDA intervention validated / not validated.
- Latency: median local-path latency vs B0 median → the "faster" half of H1.
- Fallback rate and implied monthly cost at 10k rules/day → the business recommendation.

State the conclusion as *supported*, *challenged*, or *partially supported* — e.g. "H1 supported
on latency and cost; partially supported on reliability (Hybrid matched the LLM on semantic match
but only after the deterministic validator; T5 alone was 11pp worse)".

---

## 4. The two experiments to run

### Experiment A — headline hypothesis (hybrid vs prompt-only LLM)

- **Tests:** H1 / H0 from §3.2.
- **Conditions:** B0 (prompt-only LLM) vs Hybrid, plus T5-orig alone for context.
- **Method:** `EvaluationSuite` on the held-out test set; `compare_models` for the head-to-head;
  `group_by_constraint_type()` for the breakdown.
- **Outputs:** `docs/experiments/expA_<date>_*.json` + a results table in the report.

### Experiment B — EDA-driven (does rebalancing the training data help?)

- **Tests:** "retraining on the balance-corrected 787-example set reduces `>=`/`<=` and
  over-generation errors versus the original 509-example adapter."
- **Conditions:** T5-orig vs T5-retrained on the *same* held-out test set.
- **Method:** retrain LoRA via `python -m src.train_adapter_simple --dataset
  training_data/financial_domain_combined.json --output adapters/adapter_financial_v2`
  (CPU-only machine — see §5), then `EvaluationSuite` on both adapters, focusing on the contrastive
  block via `group_by_constraint_type()`.
- **Outputs:** `adapters/adapter_financial_v2/`, `docs/experiments/expB_<date>_*.json`.
- **Design limitation to disclose:** 787 is a *superset* of 509, so Experiment B varies both
  class balance **and** total volume — it answers the real business question ("did the gap-filling
  work?") but is not a clean isolation of the balance factor. A stricter follow-up would hold the
  example count constant.

---

## 5. Feasibility notes (current machine)

| Item | Status |
| --- | --- |
| `torch` / `transformers` / `peft` | Present in `.venv` (2.12.0 / 5.8.1 / 0.19.1). Newer than the versions the adapter was trained with (`peft 0.18.1`); loads fine |
| `t5-base` weights | Cached in `~/.cache/huggingface/hub/models--t5-base` → T5 evaluation and retraining run **offline** |
| GPU | **None.** `torch.cuda.is_available()` is `False`. Retraining uses `src/train_adapter_simple.py` (forces CPU); budget ~30–90 min for 787 examples × 3 epochs. Alternative: run `notebooks/finetune_t5_lora.ipynb` on free Colab GPU (~30 min) |
| LLM baseline (B0) | Needs `OPENROUTER_API_KEY` in `.env` and network access. ~60–100 calls at `temperature=0.0`; trivial cost |
| Evaluation harness | `src/model_evaluator.py` is complete and tested. Only the `src/evaluate.py` CLI wiring for `--model t5/llm` is missing (stubs at lines 90-99) — either finish those or drive `EvaluationSuite` from a short script |

---

## 6. Limitations & threats to validity (for the report's limitations section)

- **Synthetic training data.** All training examples are hand-authored/templated, not sampled from
  real production dictionaries. The active-learning collection path exists
  (`training_data_collected.json`) but only ever captured 2 examples, and the LLM
  variation-generation augmentation (`generate_rule_variations`) has never actually run in a saved
  session. Generalisation to unseen real phrasings is untested.
- **Single domain.** Only the financial adapter exists. `healthcare_domain.json` (530 examples)
  was never trained into an adapter, so cross-domain claims are unsupported.
- **Prompt-prefix inconsistency.** Inference and the `train_adapter*.py` scripts use
  `"convert: "`; `notebooks/finetune_t5_lora.ipynb` uses `"convert to json schema: "`. If the
  shipped adapter was trained via the notebook, the inference prefix does not match training —
  this needs confirming before Experiment A, as it would depress T5-orig scores for a reason
  unrelated to the hypothesis.
- **Hybrid does not re-validate LLM output by type** before returning it; only the final
  whole-schema validation pass catches an LLM type error, and it *raises* rather than repairing.
  Report schema-valid rate at both the per-field and whole-schema level.
- **Small held-out set.** ~60–100 cases gives wide confidence intervals on any single rate;
  report differences with that caveat and avoid over-claiming small gaps.
- **`compare_models` runs sequentially**, so latency numbers are sensitive to machine load; run
  on an otherwise-idle machine and report median + p99, not just mean.

---

## 7. Suggested next actions

1. Author `training_data/holdout_financial_test.json` (§3.4 requirements).
2. Add a `src/evaluate` path or a `scripts/run_experiments.py` that runs `EvaluationSuite` for
   B0 / T5-orig / T5-retrained / Hybrid and writes reports to `docs/experiments/`.
3. Confirm the training prompt prefix of the shipped adapter (§6).
4. Retrain on `financial_domain_combined.json` → `adapters/adapter_financial_v2`.
5. Run Experiments A and B; commit reports.
6. Write the methodology section against the §3 structure, with the real numbers in §3.6.
