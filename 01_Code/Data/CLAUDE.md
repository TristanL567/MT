# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master thesis: "The Agony and the Ecstasy: Constructing a 'Crash-Filtered' Equity Index using Machine Learning" (Vienna University of Economics and Business). The project builds a full ML pipeline to predict equity crashes and use those predictions to construct a superior equity index.

## Running the Pipeline

There is no build system. Scripts are run sequentially. The entry point is `00_Master.R`, which sources individual steps via `RUN_*` boolean flags — set a flag to `TRUE` to run that step.

**R scripts** are run from RStudio using the `MT.Rproj` project file, or sourced directly. Each script calls `source("config.R")` at the top.

**Python scripts** are run independently with a `MODEL` environment variable controlling which model track to process:

```bash
# VAE — run twice, once per input type
MODEL=fund    python 08B_Autoencoder.py
MODEL=raw     python 08B_Autoencoder.py

# AutoGluon — run once per model ID (M1–M4, B1–B4, S1–S4)
MODEL=ag_fund python 09C_AutoGluon.py

# SHAP analysis
MODEL=ag_fund python 10B_SHAP.py
```

## Pipeline Execution Order

```
01_Universe.R           → CRSP universe construction
02_Prices.R             → Monthly/weekly prices
03_Fundamentals.R       → Compustat fundamentals (via CCM link)
04_Macro.R              → FRED macro variables
05_CSI_Label.R          → Acute crash labels (binary, y(t+1))
05B_Bucket_Label.R      → 5-year forward CAGR buckets (multiclass)
05C_Combined_Label.R    → Structural labels (CSI + Bucket)
06_Merge.R              → Panel merge of all sources
06B_FeatureEngineering.R→ ~460 features per observation
08_Split.R              → Train/test/OOS splits
08B_Autoencoder.py      → Beta-VAE latent features (run ×2)
09C_AutoGluon.py        → AutoML training (run ×12 models)
10B_SHAP.py             → SHAP/PDP analysis
10_Evaluation.R         → Model metrics, ROC/PR curves
11_IndexConstruction.R  → Index strategy backtest
12_Evaluation_Extension.R → Index diagnostics
13_Robustness_Checks.R  → Sensitivity analysis
```

## Architecture

### Configuration

`config.R` is the **single source of truth** for all parameters: directory paths, pipeline thresholds, date ranges, and the reproducibility seed (`SEED = 123`). All R scripts source this file first. The Python equivalent is `Paths.py`.

### Three Label Tracks

The pipeline runs three independent prediction tasks, each producing 4 models (fund features, fund+VAE latent, raw features, raw+VAE latent):

| Track | Label | Type | Models |
|---|---|---|---|
| CSI | Acute crash event | Binary | M1–M4 |
| Bucket | 5-year forward CAGR | Multiclass | B1–B4 |
| Structural | CSI + Bucket combined | Combined | S1–S4 |

### Feature Engineering (~460 features)

Built in `06B_FeatureEngineering.R` from ~40 base accounting ratios, then derived as: YoY changes, acceleration (2nd derivative), expanding mean/vol, peak deterioration, consecutive decline streaks, accounting momentum, 3Y and 5Y rolling stats, price momentum, and macro interaction terms. All features are strictly backward-looking (no lookahead).

### VAE Dimensionality Reduction

`08B_Autoencoder.py` trains a Beta-VAE separately on `fund` (fundamental features) and `raw` (all features). Architecture: `Enc(X) → [256→128→64→z]`, `Dec(z) → X'`. Loss: ELBO + βKL + optional supervised BCE. Outputs `features_latent_fund.parquet` and `features_latent_raw.parquet`.

### AutoGluon

`09C_AutoGluon.py` trains all model types. Each model's predictions and leaderboard are saved to `03_Output/Tables/ag_<model_id>/`.

### Evaluation Windows

- **Train**: 1993–2015
- **Test**: 2016–2019 (sole basis for model selection)
- **OOS**: 2020–2024 (final validation, not used for selection)

### Data Flow

Raw WRDS/FRED data → RDS intermediate files → Parquet (for VAE/Python interop) → AutoGluon model artifacts → evaluation/index construction.

Key intermediate files in `02_Data/` (git-ignored):
- `features_raw.rds`, `features_fund.rds`
- `labels_csi.rds`, `labels_bucket.rds`, `labels_structural.rds`
- `splits.rds`
- `features_latent_fund.parquet`, `features_latent_raw.parquet`

### Index Strategies

`11_IndexConstruction.R` constructs and backtests:
- **S1–S4**: Market-cap weighted index with top 5% crash-score firms excluded (one per model track)
- **C1–C3**: Concentrated long-only (200 names, two-layer veto)
- **Benchmark**: Market-cap weighted S&P 500 equivalent
