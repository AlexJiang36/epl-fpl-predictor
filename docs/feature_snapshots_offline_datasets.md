# Feature Snapshots / Offline Dataset Layer

## Purpose
This document explains the offline dataset layer added for feature snapshot exports and why it matters for the project after the live season ends.

The main goal is to make the project useful not only for live API responses, but also for:

- backtesting
- retraining
- offline analysis
- notebook-based experimentation
- reproducible feature extraction across defined GW windows

## What is a feature snapshot?
A feature snapshot is a reproducible offline dataset export for a selected gameweek window, paired with a metadata artifact.

Each snapshot consists of two parts:

1. **Dataset file**
   - usually a CSV under `artifacts/offline_datasets/`

2. **Metadata artifact**
   - a JSON manifest under `artifacts/feature_snapshots/`

This means the project stores both:
- the extracted dataset
- the context needed to understand how that dataset was produced

## Current snapshot types
The current offline dataset layer supports at least two snapshot types:

- `player_features`
- `match_features`

These correspond to the project’s existing feature export workflows for player-level and match-level data.

## Current exported examples
Examples produced in the current implementation include:

- `artifacts/offline_datasets/player_features_gw1_27_v0.csv`
- `artifacts/offline_datasets/match_features_gw1_27_v0.csv`

With matching metadata artifacts stored under:

- `artifacts/feature_snapshots/`

## Metadata stored for each snapshot
Each feature snapshot metadata artifact records fields such as:

- `snapshot_id`
- `snapshot_type`
- `feature_version`
- `extraction_timestamp`
- `gw_start`
- `gw_end`
- `model_name`
- `source_tables`
- `output_path`
- `row_count`
- `notes`

This makes the export process much more auditable and reproducible than saving CSV files alone.

## Why this matters

### 1. Backtesting
Offline feature snapshots are useful for backtesting because they let you preserve the exact feature dataset used for a chosen gameweek window.

This helps answer questions such as:
- what did the feature table look like for GW 1–27?
- how many rows were exported?
- which source tables were used?
- which feature version was extracted?

Without snapshots, it is easy to lose track of which dataset version was used when evaluating historical model performance.

### 2. Retraining
Offline snapshots are also useful for retraining workflows.

Instead of rebuilding feature datasets from scratch every time, you can:
- export a stable dataset for a defined GW range
- keep its metadata
- retrain models later using the same offline extract

This reduces ambiguity and makes it easier to compare model behavior across dataset versions.

### 3. Analysis notebooks
Feature snapshots are especially helpful for notebook-based work.

For example, they make it easier to:
- inspect feature distributions
- visualize target relationships
- debug strange values
- compare feature versions
- prototype new model ideas

Because the dataset is exported to CSV and paired with metadata, notebooks can load the offline extract directly without relying on live API calls.

### 4. Post-season usefulness
This layer is important because the live season eventually ends.

When that happens, the project can still remain useful for:
- retrospective analysis
- retraining
- feature engineering experiments
- offline model evaluation
- portfolio demos showing reproducible data workflows

This makes the project feel more durable and more like an ML / data platform rather than only a live-season app.

## Reproducibility model
The current offline dataset layer is reproducible based on:

- GW window (`gw_start`, `gw_end`)
- feature version
- model source when relevant
- source tables
- saved output path
- saved extraction metadata

This does not mean the data is immutable forever.
If source tables change and you rerun extraction later, the output may differ.

However, the snapshot layer ensures that when a dataset is exported, the project preserves enough metadata to understand:

- what was extracted
- when it was extracted
- how it was extracted
- which files were produced

## Current command pattern

### Player feature snapshot export
Example pattern:

```bash
python -m ml.validation.export_player_feature_snapshot \
  --gw-start 1 \
  --gw-end 27 \
  --feature-version v0 \
  --model-name baseline_rollavg_v0 \
  --out-csv artifacts/offline_datasets/player_features_gw1_27_v0.csv
```

### Match feature snapshot export
Example pattern:

```bash
python -m ml.validation.export_match_feature_snapshot \
  --gw-start 1 \
  --gw-end 27 \
  --feature-version v0 \
  --out-csv artifacts/offline_datasets/match_features_gw1_27_v0.csv
```

## Why this improves the project story
This offline dataset layer strengthens the project in several ways:

- it adds a reproducible data export workflow
- it makes feature extraction more auditable
- it supports offline experimentation
- it improves the project’s ML / data engineering credibility
- it helps the project remain useful after the live season ends

## Summary
The feature snapshot / offline dataset layer turns feature exports into reproducible artifacts.

Instead of producing only ad hoc CSV files, the project now produces:

- offline datasets
- paired metadata manifests
- reproducible extraction context

This supports:

- backtesting
- retraining
- analysis notebooks
- post-season project usefulness

and gives the project a stronger data engineering and ML platform narrative.
