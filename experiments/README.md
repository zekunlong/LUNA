# Journal LUNA

The journal mode implements the user-specified reliability equations and keeps
the original scripts in `legacy` mode. The original loss helpers, model/prompt
architecture, dataset splits, and legacy `forward_backward` are retained.

## Implemented now

- Explicit `--distill-weighting legacy|equal|static_mean|reliability_adaptive|instance_only`.
- Journal selection requires `--selection-protocol final_epoch`; `best_val`
  and automatic resume are never used by the journal trainer.
- One sample produces both existing weak/strong transforms. The three original
  scalar losses are summed over both views and the existing temperatures
  `[1, 3, 5, 2, 6]`. This corrects the legacy unpaired loaders and omitted strong
  loss only in the journal mode, consistently across all new controls.
- Teacher reliability uses native teacher softmax at the original base
  temperature `1.0`. Entropy and relation calculations promote probabilities
  to at least FP32 so the fixed epsilon `1e-8` does not underflow.
- Frozen teacher, detached scores/weights, prompt-gradient diagnostics,
  batch CSV, epoch means/lambda standard deviations, and training-only static
  mean provenance checks.
- A real DTD ten-batch sanity entry point and a mandatory full-training gate.

Actual DTD sanity and three Exp1 controls have completed. The three-dataset
Exp2/3/4 cohort now runs through `run_journal_exp234_cohort.py`; its immutable
condition plan and current manifests distinguish completed, running and pending
conditions. Original FGVC Aircraft and EuroSAT datasets/splits are also acquired
and checked. Acquisition alone is not a measured model result.

`export_measured_exp234_results.py` incrementally publishes only verified real
training/evaluation evidence, preserves existing measurements, and leaves
pending cells blank. Use it for all later phases; the earlier Exp1-only exporter
is not the current whole-project export entry point.

From the repository root, these two commands regenerate the measured tables,
workbook, availability history, report, Chinese analysis, and PNG/PDF overview:

```powershell
& 'E:/LUNA/.venv_journal/Scripts/python.exe' experiments/export_measured_exp234_results.py
& 'E:/LUNA/.venv_journal/Scripts/python.exe' experiments/summarize_measured_dtd.py
```

For read-only verification or an independent figure/analysis rebuild:

```powershell
& 'E:/LUNA/.venv_journal/Scripts/python.exe' experiments/export_measured_exp234_results.py --validate-only
& 'E:/LUNA/.venv_journal/Scripts/python.exe' experiments/analyze_measured_exp234.py
```

The output root defaults to `E:/LUNA/results_reliability_analysis`. No command
above trains a model or imports hypothetical metrics. The validator checks
completed source runs, final checkpoint/teacher identity, held-out sample order,
image hashes, stored probabilities, CPU recomputation of reliability/distances,
correctness, Spearman statistics and stable quintiles.

The actual first DTD Exp2 analysis has `r_batch` rho `0.0704063`, p `0.308726`:
it does not reach the conventional unadjusted 0.05 threshold. This does not
support a claim that all three reliabilities significantly track correctness.
The p-values use the two-sided asymptotic approximation and are limited by
fixed, class-correlated batches and one seed. Current values and pending
conditions are regenerated in `analysis/REAL_exp234_analysis_zh.md`; the
separate hypothetical positive scenario is never empirical evidence.

Teacher stages 5/12 are raw snapshots of the same 20-epoch trajectory, while
epoch 20 applies official final GPA (mean 15, std 1). Stage comparisons therefore
include both training progress and averaging. Stage names do not imply monotonic
accuracy. Report signed Strong-minus-Weak HM drops, including negative values.

## Fixed mathematics

For detached, aligned teacher probabilities `P` and `Q`, with `C` classes:

```text
eps = 1e-8
H(p) = -sum(p * log(p + eps))
r_ins = clamp(1 - (mean H(P) + mean H(Q)) / (2 log C), 0, 1)
cos(A, B) = dot(vec(A), vec(B)) / (norm(vec(A)) * norm(vec(B)) + eps)
r_batch = clamp(cos(P P^T, Q Q^T), 0, 1)
r_cls   = clamp(cos(P^T P, Q^T Q), 0, 1)
lambda_k = (r_k + eps) / (r_ins + r_batch + r_cls + 3 eps)
L_adapt = lambda_ins L_ins + lambda_batch L_batch + lambda_cls L_cls
```

All three reliabilities are **batch scalars**. Sample logs contain entropy,
confidence and correctness; they do not invent a per-sample `r_ins` or AUROC.
Entropy correlation for `r_ins` is largely implied by its definition and must
not be described as independent evidence of prediction correctness.

## Run the real sanity check first

Run from this repository using its Python 3.9 / PyTorch 2.0.1 environment.
The isolated Windows environment is `E:/LUNA/.venv_journal`; use its
`Scripts/python.exe` explicitly instead of the system Python. For another
machine, create a separate Python 3.9 environment and install
`experiments/requirements-journal.txt`. This leaves the legacy requirements
file unchanged and uses the bundled Dassl implementation.
The preflight additionally checks the original pinned torchvision, NumPy and
ftfy versions and required imports, including the bundled Dassl registry's
`wilds==1.2.2` dependency. It does not load/download a model or
regenerate missing dataset splits.

Set these variables to actual existing assets:

| Variable | Meaning |
|---|---|
| `LUNA_DATA_ROOT` | Contains `dtd/images/` and the original `dtd/split_zhou_DescribableTextures.json` |
| `LUNA_TEACHER_CLIP_PATH` | Original teacher CLIP foundation weights |
| `LUNA_TEACHER_CHECKPOINT` | Explicit final teacher training checkpoint, not `model-best.pth.tar` |
| `LUNA_TEACHER_TOTAL_EPOCHS` | Actual total teacher epochs; not the student's epoch count |

```powershell
python experiments/run_journal_sanity.py
```

Equivalent explicit arguments are available through `--root`,
`--teacher-clip-path`, `--teacher-checkpoint`, and `--teacher-total-epochs`.
`--seed` defaults to the existing queue's seed 1; `--batches` defaults to 10.
`--output-dir` sets the analysis root and defaults to
`E:/LUNA/results_reliability_analysis`.

Outputs include `SANITY_CHECK.md` and `sanity_result.json`, plus scalar and
sample logs when actual training batches run. A blocked report has zero real
batches and cannot unlock a full run. Sanity changes its own debug student,
saves no experiment checkpoint, and never reuses that student to initialize a
full run.

## Full-mode contract after a real pass

`train.py` accepts the new switches while retaining the original dataset and
method YAML arguments. Journal runs additionally require an explicit separate
`--output-dir`, `--teacher-checkpoint`, `--teacher-total-epochs`, and
`--sanity-report` pointing to a **passed real** `sanity_result.json`.
The configured trainer name stays `LUNA` because the existing loaders and
prompt classes use it to construct the full Base/Novel class space; the
separate implementation is `LUNAJournal`.

`static_mean` requires `--static-weights training_mean_weights.json` from a
completed, compatible adaptive training trace. The loader checks source code,
teacher foundation/checkpoint hashes, complete protocol, seed, split contents
and student initialization. Sanity/partial/evaluation traces cannot supply it.
Source-code changes invalidate the sanity gate and require a fresh real check.

The training recorder writes `journal_batches.csv`, `journal_epochs.csv/.json`,
`journal_manifest.json`, and `journal_log.txt` into each isolated run. Raw
reliabilities and sample diagnostics go under
`<analysis root>/raw_reliability/<dataset>/<variant>/...`.

## Software verification

```powershell
python -m unittest discover -s experiments/tests -v
```

These tests use synthetic/temporary fixtures. They verify formulas, autograd,
paired-forward structure, controls, final-epoch lifecycle, logging/provenance,
and byte/AST compatibility with the original `code_luna.zip` when available.
They do not claim real DTD accuracy, real ten-batch lambda statistics, or any
of the four journal experiment results.
