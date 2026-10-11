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
| Training | `training_data/financial_domain.json` (509) — same data as the original shipped adapter | Train the baseline adapter (local rebuild) | Test |
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
| T5-orig | LoRA adapter rebuilt locally on the 509-example set (`adapters/adapter_financial_v1_rebuild/final_adapter`) — **not** the shipped notebook adapter; see §4 Experiment B |
| T5-retrained | LoRA adapter retrained on the 787-example combined set |
| Hybrid | T5-retrained + `validate_node_constraints` gate + LLM fallback |
| Hybrid (confidence gate) | Hybrid + LLM fallback when T5's least certain token has probability < 0.7 (see §4 Experiment A) |

**Metrics** (all already implemented in `src/model_evaluator.py`, run via `EvaluationSuite`):

| Metric | Definition | Target |
| --- | --- | --- |
| JSON-correctness rate | Output parses as a JSON object | ≥ 99% |
| Schema-valid rate | Constraints legal for the field type (`validate_node_constraints`) | 100% for Hybrid; report as-is for T5-only |
| Exact-match rate | Output dict `==` expected dict | report; not the primary bar |
| Semantic-match rate | Same constraints however written: order-insensitive, plus the JSON Schema equivalences in `_canonical_constraints()` (**changed after the first results — see §6 "Scoring change"**; pre-registered definition was "same keys/values, order-insensitive") | primary correctness metric; within 5pp of B0 |
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

### 3.6 Results as evidence

All runs: held-out set `training_data/holdout_financial_test.json` (102 cases), 2026-10-11, LLM
`deepseek/deepseek-chat` via OpenRouter, T5 on CPU (§5). Reports in `docs/experiments/`.
Paired differences are tested with an exact McNemar test on the per-case results.

| Condition | Semantic match | Exact match | Schema-valid | Median latency | p99 latency | Fallback |
| --- | --- | --- | --- | --- | --- | --- |
| B0 (mean of 3 runs; range) | **96.7%** (96.1–97.1) | 53.3% (48.0–56.9) | 100% | 1,542 ms (1,434–1,755) | 3.2–9.6 s | – |
| T5-orig | 72.5% | 72.5% | 100% | 644 ms | 1.4 s | – |
| T5-retrained | 77.5% | 77.5% | 100% | 698 ms | 1.8 s | – |
| Hybrid, validity gate only | 77.5% | 77.5% | 100% | 719 ms | 2.0 s | 0% |
| Hybrid, confidence gate 0.7 | **89.2%** | 75.5% | 100% | 805 ms | 8.3 s | 19.6% |

Semantic match is the corrected metric (§6 "Scoring change"); exact match is shown alongside it.
T5 only ever produces the training data's output style, so for T5 the two are equal.

**H1: not supported.** Hybrid fails two of the four reject-H1 conditions in §3.3:

| Dimension | Threshold | Result (gated Hybrid) | Verdict |
| --- | --- | --- | --- |
| Correctness | ≤ 5pp below B0 | 7.5pp below the B0 mean (6.9–7.8pp across B0 runs); against B0 run 1, B0 is right on 9 cases Hybrid misses and Hybrid on 1 that B0 misses (p = 0.02) | **Fail** |
| Safety | 100% schema-valid | 100% per field | Pass |
| Latency | ≥ 5× faster on the local path | ≈2.1× (T5-only path 719 ms vs B0 medians, range 2.0–2.4×); this machine only (§6) | **Fail** |
| Coverage | ≤ 40% fallback | 19.6% | Pass |

The ungated Hybrid failed correctness by 19.3pp. The confidence gate (threshold fixed before
this run, §4) fixed 12 cases and broke none (p < 0.001). Of the 20 cases it sent to the LLM, the
LLM was right on 18; 6 of the 20 were already right from T5. Its gains were mostly malformed
patterns and wrong formats (e.g. `^[0-9]{2}$` for "two uppercase letters then 7 digits",
`format: "web"` for a URL), plus one operator error. It left 11 cases wrong: 9 were T5 answers
confident enough to pass the gate — 5 of them with the bound **reversed**, e.g. "Smallest allowed
investment is 1000" → `maximum`, "Balance must stay at or below 2500" → `minimum` — and 2 were
arguable LLM answers (`\d*` for "digits only, any length"; an extra date pattern next to
`format: date`). Routing on model confidence cannot catch errors the model is confident about.
The gate also costs tail latency: a fallback runs T5 then the LLM in sequence, so p99 rises to 8.3 s.

**Experiment B — EDA intervention: not validated.** Retraining on the 787-example rebalanced set
raised semantic match from 72.5% to 77.5% (+5pp), but the per-case change is 10 cases fixed and
5 broken (p = 0.30), so the gain is not distinguishable from noise at n = 102. On the
contrastive `>=`/`<=` block the intervention targeted (22 cases) it went from 18 to 19 correct:

| Contrastive group | n | T5-orig | T5-retrained |
| --- | --- | --- | --- |
| exclusiveMinimum | 4 | 50% | 75% |
| maximum | 6 | 67% | 83% |
| minimum | 10 | 100% | 90% |
| range / length-range | 2 | 100% | 100% |
| **All contrastive** | **22** | **81.8%** | **86.4%** |

Outside the block, `pattern` improved (33% → 50%, n = 18) and `format` fell (60% → 20%, n = 5).
The reversed-bound errors above are in the non-contrastive groups, so the rebalanced data did
not remove direction confusion in general.

**Cost.** At a 19.6% fallback rate the gated Hybrid sits in the "80% T5 + 20% LLM" band of the
documented estimate — about \$10–40/month at 10k rules/day versus \$50–200/month for 100% LLM
(`docs/next_steps.md:196-213`). These are the project's earlier estimates, not costs measured in
these runs, and they exclude hosting the local model.

**Business recommendation.** On this evidence, use the prompt-only LLM. The hybrid cuts API
calls by about 80% and halves median latency, but it is 7.5pp less accurate, and its remaining
errors are reversed bounds — silently wrong constraints, the worst kind for data validation. The
hybrid becomes worth revisiting if a gate can catch confidently wrong answers (see §7: a
rule-text consistency check) and the local path runs on faster hardware.

---

## 4. The two experiments to run

### Experiment A — headline hypothesis (hybrid vs prompt-only LLM)

- **Tests:** H1 / H0 from §3.2.
- **Conditions:** B0 (prompt-only LLM) vs Hybrid, plus T5-orig alone for context.
- **Method:** `EvaluationSuite` on the held-out test set; `compare_models` for the head-to-head;
  `group_by_constraint_type()` for the breakdown.
- **B0 repeats:** B0 is not deterministic even at `temperature=0` (27 of 102 cases changed
  result between two runs on the same day, both served by DeepInfra/StreamLake). Run B0 three
  times and report the mean and range, not a single run; the extra runs are in
  `docs/experiments/expA_b0_repeats/`.
- **Hybrid confidence gate (threshold fixed before the held-out run).** The validity-only gate
  never fired (§6), so Hybrid is re-run with a second trigger: fall back to the LLM when T5's
  confidence, the probability of its least certain generated token, is below **0.7**
  (`--confidence-threshold 0.7`). If that LLM call fails, the valid T5 answer is kept.
  The threshold was chosen on the v2 adapter's own validation split (79 examples it was not
  trained on; `scripts/confidence_calibration.py`,
  `docs/experiments/confidence_2026-10-11_adapter_financial_v2.json`): AUROC 0.93; at 0.7 the gate
  catches 4 of 5 T5 errors at a 10.1% fallback rate. It is recorded here and committed before the
  held-out run so it cannot be tuned to the test set. Caveats: only 5 validation errors; the
  split is easier than the held-out set (93.7% vs 77.5%) because training files hold near-duplicate
  phrasings; none of the 5 errors is a `>`/`>=` confusion; and 2 of the 4 caught were empty outputs
  the validity gate already catches. Expect the gate to help on malformed patterns and little on
  confidently wrong operators (e.g. "over 18" → `minimum` at confidence 0.95). The original
  validity-only Hybrid run is kept as reported; the gated run is a separate condition in
  `docs/experiments/expA_hybrid_gate/`.
- **Outputs:** `docs/experiments/expA_<date>_*.json` + a results table in the report.

### Experiment B — EDA-driven (does rebalancing the training data help?)

- **Tests:** "retraining on the balance-corrected 787-example set reduces `>=`/`<=` and
  over-generation errors versus the original 509-example adapter."
- **Conditions:** T5-orig vs T5-retrained on the *same* held-out test set.
- **Why the baseline is rebuilt, not the shipped adapter:** the shipped adapter
  (`adapters/adapter_financial`) was fine-tuned in `notebooks/finetune_t5_lora.ipynb`, which used a
  different input prefix (`"convert to json schema: "`) and training setup from the local pipeline.
  Evaluated through local inference it failed for reasons unrelated to its training data, so it
  cannot serve as a fair baseline. Both arms are therefore trained with the same local script,
  hyperparameters, and seed (`random` + `torch.manual_seed(42)`), sharing one text format
  (`src/t5_text.py`) with inference — the training data is the only intended difference.
- **Method:** train both LoRA adapters locally (CPU-only machine — see §5):
  ```
  FLAGS="--epochs 10 --learning-rate 1e-3 --batch-size 8"
  python -m src.train_adapter_simple -d training_data/financial_domain.json          -o adapters/adapter_financial_v1_rebuild $FLAGS  # T5-orig, 509
  python -m src.train_adapter_simple -d training_data/financial_domain_combined.json -o adapters/adapter_financial_v2           $FLAGS  # T5-retrained, 787
  ```
  then `EvaluationSuite` on both adapters, focusing on the contrastive block via
  `group_by_constraint_type()`.
- **Hyperparameters (identical for both arms):**

  | Setting | Value | Source |
  | --- | --- | --- |
  | Base model | `t5-base` | notebook |
  | LoRA r / alpha / dropout | 16 / 32 / 0.1 | notebook (matches shipped `adapter_config.json`) |
  | LoRA target modules | `q`, `v` | notebook |
  | Epochs | 10 | notebook |
  | Learning rate | 1e-3, AdamW (weight decay 0.01), linear schedule, 10% warmup | LR from notebook; optimiser/schedule from `train_adapter_simple.py` |
  | Batch size | 8 | notebook |
  | Validation split | 10% (shuffled, `seed=42`) | `train_adapter_simple.py` |
  | Seed | 42 (`random` + `torch`) | `train_adapter_simple.py` |
  | Adapter kept | `final_adapter` (end of epoch 10); `best_checkpoint` = lowest val loss | `train_adapter_simple.py` |

- **Why not the script defaults:** `train_adapter_simple.py` defaults to 3 epochs, LR 1e-4,
  batch 4. A first v1 rebuild with those defaults finished at val loss 1.12 and produced unusable
  output (empty dicts and degenerate repetition such as `18"+"+"+…` for `Age >= 18`). The
  notebook's settings are used instead, so the rebuild is as close to the shipped adapter's
  training as the local pipeline allows; the remaining differences are the prompt/text format
  (`src/t5_text.py`) and the training loop (custom loop vs HF `Trainer`). Record this failed run in the
  report as a pipeline-validation finding, not as an experimental result.
- **Outputs:** `adapters/adapter_financial_v1_rebuild/`, `adapters/adapter_financial_v2/`,
  `docs/experiments/expB_<date>_*.json`.
- **Design limitation to disclose:** 787 is a *superset* of 509, so Experiment B varies both
  class balance **and** total volume — it answers the real business question ("did the gap-filling
  work?") but is not a clean isolation of the balance factor. A stricter follow-up would hold the
  example count constant.

---

## 5. Feasibility notes (current machine)

| Item | Status |
| --- | --- |
| Hardware | HP ZBook Studio G5 (2019 mobile workstation): Intel Core i7-9850H, 6 cores / 12 threads @ 2.60 GHz; 32 GB RAM; NVIDIA Quadro P1000 (4 GB). Windows 11 Pro (build 26200) with virtualisation-based security on (Credential Guard, Hypervisor-enforced Code Integrity) |
| Runtime environment | All training and experiments ran inside **WSL2** (Linux 6.6, `microsoft-standard-WSL2`), which sees 12 logical CPUs and ~15 GB of the 32 GB RAM |
| `torch` / `transformers` / `peft` | Present in `.venv` (2.12.0 / 5.8.1 / 0.19.1). Newer than the versions the adapter was trained with (`peft 0.18.1`); loads fine |
| `t5-base` weights | Cached in `~/.cache/huggingface/hub/models--t5-base` → T5 evaluation and retraining run **offline** |
| GPU | **Present but unused — all runs are CPU-only.** The Quadro P1000 is visible in WSL2 (`nvidia-smi`, driver 573.71, CUDA 12.8), but the installed `torch 2.12.0+cu130` needs a CUDA 13 driver, so `torch.cuda.is_available()` is `False` (every run logs "NVIDIA driver on your system is too old"). A newer driver may not be enough on its own: CUDA 13 is believed to drop Pascal (sm_61) GPUs such as the P1000, so GPU use would likely need a cu126 PyTorch build — untested. Retraining uses `src/train_adapter_simple.py` (forces CPU); measured ~7.2 s/step at batch 8 → ~70 min for v1 (509 × 10 epochs) and ~110 min for v2 (787 × 10 epochs). Alternative: run `notebooks/finetune_t5_lora.ipynb` on free Colab GPU (~30 min) |
| LLM baseline (B0) | Needs `OPENROUTER_API_KEY` in `.env` and network access. ~60–100 calls at `temperature=0.0`; trivial cost |
| Evaluation harness | `src/model_evaluator.py` (metrics, `EvaluationSuite`) driven by `scripts/run_experiments.py`, which runs each condition on the held-out set and writes reports with git SHA, dirty flag, and dataset hash to `docs/experiments/`. It aborts on the first failed LLM call rather than scoring it. The `src/evaluate.py` CLI stubs for `--model t5/llm` (lines 90-99) were left unfinished — the runner replaced them |

---

## 6. Limitations & threats to validity (for the report's limitations section)

- **Synthetic training data.** All training examples are hand-authored/templated, not sampled from
  real production dictionaries. The active-learning collection path exists
  (`training_data_collected.json`) but only ever captured 2 examples, and the LLM
  variation-generation augmentation (`generate_rule_variations`) has never actually run in a saved
  session. Generalisation to unseen real phrasings is untested.
- **Single domain.** Only the financial adapter exists. `healthcare_domain.json` (530 examples)
  was never trained into an adapter, so cross-domain claims are unsupported.
- **Training/inference mismatch (resolved by design).** The shipped adapter was trained in
  `notebooks/finetune_t5_lora.ipynb` with the `"convert to json schema: "` prefix, while local
  inference uses `"convert: "`; it failed under local evaluation. Rather than score it, both
  Experiment B adapters are retrained locally with one shared text format (`src/t5_text.py`).
  Disclose that the reported T5-orig is a local rebuild on the same 509 examples, not the
  originally shipped artefact. Residual risk: CPU training may still not be bit-for-bit
  deterministic across library versions, despite seeding.
- **Hybrid does not re-validate LLM output by type** before returning it; only the final
  whole-schema validation pass catches an LLM type error, and it *raises* rather than repairing.
  Report schema-valid rate at both the per-field and whole-schema level.
- **Small held-out set.** ~60–100 cases gives wide confidence intervals on any single rate;
  report differences with that caveat and avoid over-claiming small gaps.
- **`compare_models` runs sequentially**, so latency numbers are sensitive to machine load; run
  on an otherwise-idle machine and report median + p99, not just mean.
- **Latency is hardware-specific (CPU only, virtualised).** Every T5 and Hybrid latency was
  measured on the CPU of a 2019 mobile i7-9850H inside WSL2, under Windows virtualisation-based
  security (§5); the laptop's Quadro P1000 went unused because of a PyTorch/driver CUDA mismatch.
  B0 latency is network and provider time, and so does not depend on this machine. The measured
  ≈2.1× speed-up (range 2.0–2.4× across the three B0 runs) therefore answers "is the local model
  fast enough *on this laptop's CPU*", not in general: a GPU or a server CPU could plausibly
  close some or all of the gap to the 5× threshold. This does **not** change the Experiment A
  verdict — Hybrid also fails the correctness threshold (by 19.3pp ungated, 7.5pp with the
  confidence gate), and hardware does not affect accuracy — but the report should state the
  latency result as specific to this machine.
- **Scoring change made after the first results were seen (must be disclosed).** The
  pre-registered semantic-match metric compared dicts exactly (order aside), so in practice it
  always equalled exact match. On the first full Experiment A run (2026-10-11, commit `9c1eb51`)
  it scored B0 at **48.0%**, but 40 of B0's 53 "failures" were correct constraints written
  differently: an added `"type"` key, `\d` instead of `[0-9]`, and `minLength`/`maxLength` that
  the pattern already fixed. T5 never shows these because it learned the test set's output style,
  so the metric was biased towards the local model. Commit `d9a2395` added three general
  equivalences — no rule refers to a particular test case:
  1. `"type"` is ignored (the field type comes from the schema; schema-valid rate checks it);
  2. `\d` ≡ `[0-9]` in patterns (ECMA-262 regex semantics, which JSON Schema uses);
  3. `minLength`/`maxLength` are ignored when a fully anchored pattern (`^…$`, no top-level `|`)
     already implies them, i.e. they do not narrow the strings the pattern accepts.

  Safeguards: each rule has unit tests, including cases that must still fail (`\d*` vs `[0-9]+`,
  `exclusiveMinimum` vs `minimum`, a length bound that narrows the pattern); exact match stays
  strict and is reported next to semantic match; one arguably-redundant case (an extra date
  pattern alongside `format: date`) was deliberately *not* given a rule. Re-scoring the same
  first-run outputs: B0 48.0% → **97.1%**; T5-orig (72.5%) and Hybrid/T5-retrained (77.5%)
  unchanged. **Effect on the decision:** under the original definition Hybrid passes the
  correctness threshold (it is 29.5pp *above* B0); under the corrected one it fails it (19.6pp
  below). The overall verdict does not hinge on this change, because Hybrid also fails the
  latency threshold (≈2.2× faster vs the 5× required) under either definition. The same holds
  for the later confidence-gated Hybrid run: by exact match it scores 75.5% against the B0 mean
  of 53.3% (pass); by semantic match, 89.2% against 96.7% (fail). Report both metrics and this
  paragraph.
- **Empty-output scoring (harness fix, no effect on reported numbers).** Interpreters return `{}`
  on failure, which was being scored as valid JSON and schema-valid; a failed-API run of B0 showed
  100% "valid" with 0% matches. Since `9c1eb51` an empty answer is only valid when no constraint was
  expected, and the runner aborts on the first failed LLM call instead of scoring it. No T5 run
  produced an empty answer, so Experiment B is unaffected.
- **The original Hybrid fallback never fired, so a confidence gate was added after the first
  results.** The validity gate only triggers on invalid T5 output, but all of T5-retrained's
  wrong answers were well-formed and type-valid, so the fallback rate was 0% and Hybrid behaved
  exactly like T5-retrained. Adding a confidence trigger (threshold 0.7) was a design change made
  *after* seeing that result, so it must be disclosed as such. Safeguards: the threshold was chosen
  on the adapter's own validation split, not the held-out set, and committed (`f7bbd32`) before
  the gated run (`84bb216`); both runs are reported. Remaining weaknesses: the validation split had
  only 5 T5 errors, so the threshold rests on little evidence, and a different threshold could do
  better or worse on the held-out set — no sensitivity analysis was run, deliberately, as tuning it
  on the held-out set would leak. The gate cannot catch confidently wrong answers: 9 of the 11
  remaining errors passed it, 5 with the bound reversed (§3.6).

---

## 7. Suggested next actions

1. Author `training_data/holdout_financial_test.json` (§3.4 requirements).
2. Add a `src/evaluate` path or a `scripts/run_experiments.py` that runs `EvaluationSuite` for
   B0 / T5-orig / T5-retrained / Hybrid and writes reports to `docs/experiments/`.
3. ~~Confirm the training prompt prefix of the shipped adapter (§6).~~ Resolved: the shipped
   notebook adapter is replaced by a local rebuild (§4 Experiment B).
4. Train both adapters locally with the same seeded settings:
   `financial_domain.json` → `adapters/adapter_financial_v1_rebuild` (T5-orig), then
   `financial_domain_combined.json` → `adapters/adapter_financial_v2` (T5-retrained).
5. Run Experiments A and B; commit reports. Run B0 three times (§4 Experiment A) and score with
   the corrected semantic-match metric, disclosing the change (§6 "Scoring change").
6. Write the methodology section against the §3 structure, with the real numbers in §3.6.
7. (Follow-up, beyond S22 scope) Add a rule-text consistency gate for bound direction — e.g.
   "at least" / "floor" / "smallest" must map to `minimum`, "at or below" / "ceiling" / "within"
   to `maximum` / `maxLength` — and evaluate it on a fresh held-out set, since this one has now
   informed two design changes.
