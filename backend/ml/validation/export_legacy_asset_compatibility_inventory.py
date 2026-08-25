#!/usr/bin/env python3
"""
Day124B — Day1–61 Legacy Asset Compatibility Inventory exporter.

This exporter is intentionally read-only with respect to application code, the
database, prediction tables, and frozen artifacts. It performs static filesystem
and source-code inspection only, then writes a Markdown inventory report.

Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CLASSIFICATIONS = (
    "KEEP",
    "WRAP",
    "MIGRATE",
    "RETIRE_WITH_REASON",
)


@dataclass(frozen=True)
class Asset:
    asset_id: str
    category: str
    name: str
    paths: Tuple[str, ...]
    classification: str
    role: str
    dependencies: str
    data_contract: str
    callability: str
    test_patterns: Tuple[str, ...] = field(default_factory=tuple)
    overlap: str = ""
    rationale: str = ""
    notes: str = ""


ASSETS: Tuple[Asset, ...] = (
    Asset(
        "db_core",
        "Database",
        "Season-aware core ORM and schema",
        (
            "backend/app/models/team.py",
            "backend/app/models/player.py",
            "backend/app/models/gameweek.py",
            "backend/app/models/fixture.py",
            "backend/app/models/player_gw_stat.py",
            "backend/alembic/versions/20260602_day60_add_season_columns.py",
            "backend/alembic/versions/5a76e5d27a59_make_core_uniqueness_season_aware.py",
        ),
        "KEEP",
        "Canonical relational foundation for teams, players, GWs, fixtures, and player actuals.",
        "SQLAlchemy ORM, Alembic, PostgreSQL.",
        "Season-aware keys for gameweeks/fixtures/player_gw_stats; target-season reads and writes depend on these tables.",
        "Runtime ORM models are imported by mounted API routes and ML jobs.",
        ("app.models.fixture", "app.models.gameweek", "app.models.player_gw_stat"),
        "Day60 season columns and later season-aware uniqueness are the base expected by Day71+ artifacts.",
        "Foundational data model should be reused, not duplicated.",
    ),
    Asset(
        "predictions_table",
        "Database",
        "Player predictions table and ORM",
        (
            "backend/app/models/prediction.py",
            "backend/alembic/versions/3d86053cdecc_create_predictions.py",
            "backend/alembic/versions/20260602_day60_add_season_columns.py",
            "backend/alembic/versions/5a76e5d27a59_make_core_uniqueness_season_aware.py",
        ),
        "KEEP",
        "Legacy persistent store for per-player, per-GW predicted points.",
        "SQLAlchemy ORM, PostgreSQL; consumed by predictions/recommendations/chips/wildcard/evaluation code.",
        "Unique logical key is season + player_id + target_gw + model_name; value is predicted_points plus created_at.",
        "Still actively read by mounted APIs and legacy decision utilities.",
        ("adapt_legacy_db_player_prediction", "Prediction"),
        "Day76B provides an explicit adapter from legacy DB Prediction rows to the standard prediction contract.",
        "Preserve the table as a compatibility/persistence surface while newer pipelines keep immutable artifacts as evidence.",
    ),
    Asset(
        "match_predictions_table",
        "Database",
        "Match predictions table and ORM",
        (
            "backend/app/models/match_prediction.py",
            "backend/alembic/versions/d2c76c0c5236_add_match_predictions_table.py",
            "backend/alembic/versions/20260602_day60_add_season_columns.py",
            "backend/alembic/versions/5a76e5d27a59_make_core_uniqueness_season_aware.py",
        ),
        "KEEP",
        "Legacy persistent store for 1X2 probabilities/result labels by fixture and model.",
        "SQLAlchemy ORM, PostgreSQL; consumed by match prediction APIs/evaluation.",
        "Season + fixture_id + model_name with pred_home_win/pred_draw/pred_away_win/pred_result.",
        "Still read and written by the mounted match-predictions route.",
        ("MatchPrediction",),
        "Day76B standard match prediction contracts and the frozen Match Model evidence overlap with this persistence surface.",
        "Keep schema compatibility; production write policy must remain explicitly gated.",
    ),
    Asset(
        "historical_staging",
        "Data ingest",
        "Historical staging schema and import utilities",
        (
            "backend/alembic/versions/b8f3a2c1d4e6_create_historical_staging_tables.py",
            "backend/ml/ingest/import_historical_season_staging.py",
            "backend/ml/ingest/prepare_vaastav_historical_csvs.py",
        ),
        "KEEP",
        "Historical source normalization/staging before canonical reuse.",
        "pandas, SQLAlchemy, historical CSV inputs, staging tables.",
        "Season-scoped raw historical teams/players/fixtures/player-GW rows with mapping metadata.",
        "CLI/import utilities; not a public API surface.",
        ("historical_staging",),
        "Feeds the later multi-season/prior feature work rather than competing with it.",
        "Preserve as the safe boundary for historical data.",
    ),
    Asset(
        "historical_canonical_import",
        "Data ingest",
        "Historical canonical-season importer",
        ("backend/ml/ingest/import_historical_season.py",),
        "KEEP",
        "Maps validated historical CSV data into canonical season-aware fact tables.",
        "pandas, SQLAlchemy SessionLocal, optional player/team mapping CSVs.",
        "Dry-run by default; canonical writes happen only with --write and are season-scoped.",
        "CLI callable.",
        ("import_historical_season",),
        "Complements staging/prior work; does not replace live FPL ingest.",
        "Useful controlled historical ingest path; retain its write gate.",
    ),
    Asset(
        "live_ingest_routes",
        "Data ingest",
        "Mounted FPL bootstrap/fixture/gameweek/player-actual ingest routes",
        (
            "backend/app/api/routes/ingest.py",
            "backend/app/api/routes/gameweeks.py",
            "backend/app/api/routes/ingest_gw_stats.py",
        ),
        "WRAP",
        "Updates official live FPL entities, fixtures, gameweek state, and player actuals.",
        "FastAPI, requests/http client code, SQLAlchemy, FPL endpoints.",
        "Writes season-aware canonical tables used by both legacy and Day71+ pipelines.",
        "Mounted in backend/app/main.py.",
        (),
        "Day128+ canonical runner should orchestrate the same ingest capability without cloning route logic.",
        "Preserve endpoint behavior, but put the capability behind reusable service/runner boundaries.",
    ),
    Asset(
        "baseline_rollavg_v0",
        "Player model",
        "baseline_rollavg_v0 prediction runner",
        ("backend/ml/predict/run_baseline_rollavg_v0.py",),
        "MIGRATE",
        "Historical rolling-average player baseline and DB writer.",
        "player_gw_stats, players, pandas, SQLAlchemy, DATABASE_URL.",
        "Reads season-scoped historical actuals and replaces predictions rows for target season/GW/model.",
        "CLI callable and also represented by legacy API/weekly runner behavior.",
        ("baseline_rollavg_v0",),
        "Day76B standard prediction contracts and Day76D artifact-first refresh create a safer output boundary.",
        "Reuse the baseline computation as a benchmark; migrate direct DB writing behind explicit PRE/publish gates.",
    ),
    Asset(
        "baseline_rollavg_v1",
        "Player model",
        "baseline_rollavg_v1 prediction runner",
        ("backend/ml/predict/run_baseline_rollavg_v1.py",),
        "MIGRATE",
        "Minutes/cost-adjusted rolling player baseline and DB writer.",
        "player_gw_stats, players, pandas, SQLAlchemy, DATABASE_URL.",
        "Reads season-scoped actuals and replaces predictions rows for target season/GW/model.",
        "CLI callable; default legacy weekly runner invokes it.",
        ("baseline_rollavg_v1",),
        "Overlaps standard player prediction output contracts and the newer immutable refresh path.",
        "Keep the algorithm as a benchmark, but move writes behind the canonical runner.",
    ),
    Asset(
        "ridge_rollform_v1",
        "Player model",
        "ridge_rollform_v1 training and next-GW prediction path",
        (
            "backend/ml/train/train_ridge_rollform_v1.py",
            "backend/ml/predict/predict_next_gw_ridge_rollform_v1.py",
        ),
        "MIGRATE",
        "Historical multi-season ridge feature/model path with prediction-table writing.",
        "pandas, scikit-learn Ridge, SQLAlchemy, player_gw_stats/players.",
        "Builds lag/rolling features; training/prediction path can replace target prediction rows.",
        "CLI callable.",
        ("ridge_rollform_v1",),
        "Its reusable modeling logic overlaps later model-training/evidence work, while direct writes conflict with artifact-first live safety.",
        "Separate train/evaluate/infer from publish; retain as a reproducible benchmark.",
    ),
    Asset(
        "player_v2_1_linear",
        "Player model",
        "v2_1 linear player research family",
        (
            "backend/ml/train/train_ridge_player_v2_1.py",
            "backend/ml/train/train_elasticnet_player_v2_1.py",
            "backend/ml/train/train_ridge_player_v2_1_sweep.py",
        ),
        "KEEP",
        "Research/evaluation baselines for v2_1 player features.",
        "pandas, scikit-learn Ridge/ElasticNet, player_features_v2_1 CSV.",
        "CSV in; temporal split by GW; metrics/feature weights or sweep CSV out; no prediction-table write in these scripts.",
        "CLI callable research scripts.",
        (),
        "Future model program can compare against them as reproducible linear baselines.",
        "Keep as evidence/reference code. Standardized training manifests can wrap them later without erasing history.",
    ),
    Asset(
        "player_v2_1_tree",
        "Player model",
        "v2_1 tree player research family",
        (
            "backend/ml/train/train_gbr_player_v2_1.py",
            "backend/ml/train/train_gbr_player_v2_1_sweep.py",
            "backend/ml/train/train_lgbm_player_v2_1.py",
        ),
        "KEEP",
        "GBR/LightGBM player research and sweep scripts.",
        "pandas, scikit-learn GradientBoostingRegressor, optional LightGBM, player_features_v2_1 CSV.",
        "CSV in; temporal split; validation metrics/importance or sweep CSV out; scripts do not serialize an estimator.",
        "CLI callable research scripts.",
        (),
        "Historical model metadata names gbr_player_v2_1 as the old player production default, but GW1 live evidence uses the newer frozen PRE pipeline.",
        "Retain as benchmark/evidence code; do not treat old metadata default as current activation authority.",
    ),
    Asset(
        "player_v2_1_eval",
        "Evaluation",
        "v2_1 player baseline comparison utility",
        ("backend/ml/train/eval_player_baselines_v2_1.py",),
        "KEEP",
        "Compares simple player baselines on the v2_1 feature dataset.",
        "pandas, scikit-learn metrics, player feature CSV.",
        "CSV in; evaluation table/optional CSV out.",
        "CLI callable.",
        (),
        "Complements, rather than replaces, formal frozen live evaluation.",
        "Useful research baseline evidence.",
    ),
    Asset(
        "player_eval_db",
        "Evaluation",
        "Season-aware DB player prediction evaluator",
        ("backend/ml/eval/eval_predictions.py",),
        "WRAP",
        "Computes overall/per-GW MAE by joining persisted predictions to actual player_gw_stats.",
        "pandas, SQLAlchemy, scikit-learn, DATABASE_URL.",
        "Season + model + GW range; predictions joined to actual total_points.",
        "CLI callable.",
        (),
        "GW1 POST requires broader metrics and frozen-artifact lineage, not merely live DB rows.",
        "Keep as a lightweight diagnostic, but wrap formal evaluation around immutable prediction evidence.",
    ),
    Asset(
        "player_eval_legacy_duplicate",
        "Evaluation",
        "Legacy baseline MAE evaluator duplicate",
        ("backend/ml/eval/legacy_eval_baseline_mae.py.py",),
        "RETIRE_WITH_REASON",
        "Older MAE utility joining predictions to player_gw_stats.",
        "SQLAlchemy ORM and application DB session.",
        "Model/GW filters without explicit season in the join/filter.",
        "CLI callable.",
        (),
        "Superseded by season-aware eval_predictions.py and future frozen-artifact POST evaluation.",
        "Keep file for history during Day124B; do not use as an operational evaluator because cross-season ambiguity is possible.",
    ),
    Asset(
        "match_baseline_route",
        "Match model",
        "match_baseline_v0 mounted API inference/writer",
        ("backend/app/api/routes/match_predictions.py",),
        "MIGRATE",
        "Recent-form heuristic 1X2 inference plus match_predictions persistence.",
        "FastAPI, Fixture/Team/MatchPrediction ORM.",
        "Reads current-season fixtures; writes 1X2 probabilities/result label for fixture/model.",
        "Mounted API: /match/predictions and /match/predictions/run[_gw].",
        (),
        "Day76B contracts and final GW1 frozen Match Model create a distinct artifact-first match prediction path.",
        "Preserve read API compatibility; remove baseline-specific mutation from the route when canonical PRE/publish stages exist.",
    ),
    Asset(
        "match_logreg_v0",
        "Match model",
        "match_logreg_v0 historical trainer/writer",
        ("backend/ml/train/train_match_logreg_v0.py",),
        "RETIRE_WITH_REASON",
        "Older logistic-regression match model trained from fixture history and written directly to match_predictions.",
        "pandas, scikit-learn LogisticRegression, SQLAlchemy.",
        "GW-range fixture features/results; direct match_predictions write; old path is not consistently season-scoped.",
        "CLI callable.",
        (),
        "Superseded by v2 match research family and newer frozen match pipeline.",
        "Retain as historical reference only; not safe as current live writer.",
    ),
    Asset(
        "match_backfill_v0",
        "Match model",
        "v0 historical match prediction backfill",
        ("backend/ml/predict/backfill_match_predictions_v0.py",),
        "RETIRE_WITH_REASON",
        "Backfills match_baseline_v0 and/or match_logreg_v0 predictions for finished fixtures.",
        "fixtures/match_predictions tables, pandas, scikit-learn, DATABASE_URL.",
        "GW-range finished fixtures; direct delete/insert into match_predictions; no explicit season column in its write payload.",
        "CLI callable.",
        (),
        "Season-aware schema and immutable evaluation evidence make this old backfill unsafe as a modern operational path.",
        "Preserve only for historical archaeology until archive cleanup.",
    ),
    Asset(
        "match_logreg_v2",
        "Match model",
        "match_logreg_v2 research benchmark",
        ("backend/ml/train/train_match_logreg_v2.py",),
        "KEEP",
        "v2 match-result logistic-regression benchmark.",
        "match_features_v2 CSV, pandas, scikit-learn.",
        "CSV in; temporal train/validation split; accuracy/log-loss/confusion matrix; optional summary CSV.",
        "CLI callable research script.",
        (),
        "Provides a reproducible benchmark against tree-based match models.",
        "Keep as archived benchmark; no live activation implied.",
    ),
    Asset(
        "match_gbc_v2",
        "Match model",
        "match_gbc_v2 research family",
        (
            "backend/ml/train/train_match_gbc_v2.py",
            "backend/ml/train/train_match_gbc_v2_sweep.py",
        ),
        "WRAP",
        "Gradient-boosting classifier match model and hyperparameter sweep.",
        "match_features_v2 CSV, pandas, scikit-learn.",
        "CSV in; temporal split; accuracy/log-loss/confusion matrix and optional CSV summaries; estimator is not serialized by these scripts.",
        "CLI callable research scripts.",
        (),
        "Historical metadata marks match_gbc_v2 as old production default, but modern live use requires a reproducible inference artifact/contract.",
        "Reuse model family, but wrap train/infer/versioning before any future Champion activation.",
    ),
    Asset(
        "match_lgbm_v2",
        "Match model",
        "match_lgbm_v2 research benchmark",
        ("backend/ml/train/train_match_lgbm_v2.py",),
        "KEEP",
        "LightGBM v2 match-result experiment.",
        "match_features_v2 CSV, pandas, LightGBM.",
        "CSV in; temporal evaluation; optional summary CSV.",
        "CLI callable when LightGBM runtime is available.",
        (),
        "Historical alternative to match_gbc_v2.",
        "Keep as benchmark; do not make it operational by metadata alone.",
    ),
    Asset(
        "match_goals_v2",
        "Match model",
        "match_goals_gbr_v2 goals prototype",
        ("backend/ml/train/train_match_goals_v2.py",),
        "WRAP",
        "Separate home/away goal regression prototype for scoreline foundations.",
        "match_features_v2 CSV, pandas, scikit-learn GradientBoostingRegressor.",
        "CSV in; temporal evaluation with home/away MAE/RMSE; optional summary CSV.",
        "CLI callable research script.",
        (),
        "Overlaps newer scoreline prediction contracts and final scoreline freeze track.",
        "Retain the experimental model idea; wrap it in standard scoreline outputs before production use.",
    ),
    Asset(
        "match_eval_db",
        "Evaluation",
        "Legacy DB match prediction evaluator",
        ("backend/ml/eval/eval_match_predictions.py",),
        "MIGRATE",
        "Computes result accuracy/confusion matrix/log loss from persisted match predictions.",
        "pandas, SQLAlchemy, scikit-learn, DATABASE_URL.",
        "GW-range finished fixtures joined to match_predictions by fixture_id and model_name.",
        "CLI callable.",
        (),
        "Formal Match Model POST evaluation requires frozen run lineage plus Brier/goals/exact-score metrics and season safety.",
        "Reuse metric ideas, but migrate formal evaluation to immutable frozen artifacts and explicit season scope.",
    ),
    Asset(
        "model_metadata",
        "Registry / metadata",
        "Legacy model metadata schema/store/seed",
        (
            "backend/app/schemas/model_metadata.py",
            "backend/app/utils/model_metadata_store.py",
            "backend/scripts/seed_model_metadata.py",
        ),
        "MIGRATE",
        "Stores historical model status/metrics/default flags as JSON artifacts.",
        "Pydantic model, local artifacts/model_metadata directory.",
        "Per-model JSON metadata with task/feature version/windows/metrics/status/old default flags.",
        "Seed CLI plus library store; mounted model/evaluation APIs read these artifacts.",
        ("model_registry",),
        "Day76B model registry and the newer rule that software deployment != model activation require a newer authority boundary.",
        "Preserve historical metadata, but old is_production_default flags must not silently activate current live models.",
    ),
    Asset(
        "models_api",
        "API",
        "Mounted /models API",
        ("backend/app/api/routes/models.py",),
        "WRAP",
        "Lists model names from prediction rows plus legacy metadata artifacts.",
        "Prediction ORM, model_metadata_store, current season.",
        "Returns model rows with status/default/metric metadata.",
        "Mounted in backend/app/main.py.",
        (),
        "Should eventually expose Champion/Challenger activation state from the versioned model registry.",
        "Preserve response usefulness while changing authority source.",
    ),
    Asset(
        "evaluation_api",
        "API",
        "Mounted /evaluation API and frontend metadata view",
        (
            "backend/app/api/routes/evaluation.py",
            "frontend/src/app/evaluation/page.tsx",
        ),
        "WRAP",
        "Displays legacy model metadata metrics and production-default labels.",
        "model_metadata_store; Next.js client.",
        "Model metadata summaries, not fresh frozen-GW evaluation results.",
        "Backend route is mounted; frontend page is routable.",
        (),
        "GW1+ POST evaluation should feed a formal track-record surface.",
        "Keep the UX/API idea; replace stale metadata-only authority with evaluated run evidence.",
    ),
    Asset(
        "predictions_api",
        "API",
        "Mounted /predictions read API plus baseline mutation endpoint",
        ("backend/app/api/routes/predictions.py",),
        "WRAP",
        "Lists/sorts player predictions and exposes a baseline run endpoint.",
        "Prediction/Player/Team/Gameweek/PlayerGameweekStat ORM.",
        "Read rows by season/GW/model with filters; /baseline/run mutates predictions.",
        "Mounted in backend/app/main.py.",
        ("adapt_legacy_db_player_prediction",),
        "Day76B explicitly supports adapting legacy DB rows to the standard player prediction contract.",
        "Preserve read contract; separate mutation into canonical PRE/publish jobs.",
    ),
    Asset(
        "recommend_squad",
        "Decision",
        "Legacy /recommendations/squad greedy squad builder",
        ("backend/app/api/routes/recommendations.py",),
        "MIGRATE",
        "Greedy 15-player/starting-XI recommendation from persisted predicted points.",
        "Prediction/Player/Team ORM; hardcoded squad/formation/budget/club rules.",
        "Returns starting_xi, bench, squad_list, cost/bank summaries.",
        "Mounted API.",
        (),
        "Overlaps Day74B versioned rules, Day100A SquadLegalityEngine, Day100B objective, and Day101A/B optimizers.",
        "Preserve approved API shape where useful; migrate decision logic to shared rules and optimizer contracts.",
    ),
    Asset(
        "recommend_transfers",
        "Decision",
        "Legacy /recommendations/transfers single-transfer recommender",
        ("backend/app/api/routes/recommendations.py",),
        "MIGRATE",
        "Selects one outgoing player and ranks same-position replacements.",
        "Prediction/Player/Team ORM; hardcoded max-per-club, bank, free-transfer and -4 hit logic.",
        "Request includes squad ids/bank/free transfers; response returns candidate rows and net gain.",
        "Mounted API.",
        (),
        "Directly overlaps Day74B transfer rules and Day126+ free-transfer ledger/candidate generator/optimizer.",
        "Reuse response semantics and ranking ideas, but migrate all legality/economics to shared transfer state.",
    ),
    Asset(
        "recommend_captain",
        "Decision",
        "Legacy /recommendations/captain recommender",
        ("backend/app/api/routes/recommendations.py",),
        "MIGRATE",
        "Ranks persisted predicted points with recent-form/minutes labels for captain/vice.",
        "Prediction/Player/Team/PlayerGameweekStat ORM.",
        "Returns captain, vice, top_candidates and explanatory fields; future match factors are placeholders.",
        "Mounted API and frontend BFF/page consume it.",
        (),
        "Overlaps Day101B lineup/captain optimizer and future stateful weekly decision engine.",
        "Preserve explanation-oriented API shape; replace independent captain selection with shared lineup/squad state.",
    ),
    Asset(
        "team_form",
        "Feature / API",
        "Mounted team form utility",
        ("backend/app/api/routes/team_form.py",),
        "KEEP",
        "Computes recent W/D/L points, goals for/against and clean-sheet rate from finished fixtures.",
        "Fixture and Team ORM.",
        "Season-scoped read-only summary with optional before timestamp.",
        "Mounted /teams/{team_id}/form route.",
        (),
        "Can continue as an explanatory/feature utility while newer match features evolve.",
        "Useful read-only capability with no competing write semantics.",
    ),
    Asset(
        "h2h",
        "Feature / API",
        "Mounted head-to-head utility",
        ("backend/app/api/routes/h2h.py",),
        "KEEP",
        "Returns season-scoped previous meetings and summary statistics.",
        "Fixture and Team ORM.",
        "Read-only season/team-pair/N fixture contract.",
        "Mounted /h2h route.",
        (),
        "Independent explanatory surface; not duplicated by the current prediction contracts.",
        "Keep as optional product/feature context.",
    ),
    Asset(
        "wildcard_horizon",
        "Decision",
        "Legacy wildcard multi-GW horizon builder",
        ("backend/app/utils/wildcard_horizon.py",),
        "MIGRATE",
        "Aggregates stored future Prediction rows over a requested horizon with minutes stability.",
        "Prediction/Player/Team/PlayerGameweekStat ORM.",
        "DB prediction rows across start_gw..end_gw -> per-player horizon totals and GW predictions.",
        "Library callable; wildcard_builder imports it.",
        (),
        "Overlaps Day79B fixture horizon and Day97A standard player prediction horizon, where missing future predictions remain explicitly missing.",
        "Move wildcard planning to the standard horizon contract rather than rebuilding a separate DB horizon.",
    ),
    Asset(
        "wildcard_builder",
        "Decision",
        "Legacy wildcard greedy squad builder",
        ("backend/app/utils/wildcard_builder.py",),
        "MIGRATE",
        "Builds a locked-player-aware wildcard squad from legacy horizon rows.",
        "wildcard_horizon, Prediction-derived player rows, hardcoded quotas/formation/club/budget rules.",
        "WildcardBuildResponse with squad, XI/bench, captain/vice, rejected candidates and priority transfers.",
        "Library callable; mounted chips route calls it.",
        (),
        "Overlaps Day74B rules and Day100A legality engine; uses duplicated SQUAD_RULES/STARTING_RULES constants.",
        "Preserve lock-player/useful response behavior; move legality and optimizer mechanics to shared engines.",
    ),
    Asset(
        "chip_routes",
        "Decision",
        "Legacy mounted Free Hit / Wildcard chip routes",
        ("backend/app/api/routes/chips.py",),
        "MIGRATE",
        "Builds Free Hit and delegates Wildcard planning.",
        "Prediction/Player/Team ORM, wildcard_builder, hardcoded squad/lineup rules.",
        "FreeHit/Wildcard request-response schemas; reads predictions and builds decision payloads.",
        "Mounted /chips routes.",
        (),
        "Day75A chip registry explicitly scans this route as a legacy source; Day74B rules cover shared transfer legality.",
        "Keep endpoints while migrating chip inventory/window/effect/legality behavior to versioned rules.",
    ),
    Asset(
        "chip_scenarios",
        "Decision",
        "Legacy Bench Boost and Triple Captain scenario calculators",
        (
            "backend/app/utils/bench_boost_scenario.py",
            "backend/app/utils/triple_captain_scenario.py",
            "backend/app/utils/chip_scenario.py",
        ),
        "MIGRATE",
        "Computes projected chip deltas from a SquadSnapshot and Prediction rows.",
        "Prediction ORM, SquadSnapshot, shared chip_scenario result wrapper.",
        "Scenario result with baseline, modified points, delta, explanation/details.",
        "Library callable.",
        (),
        "Day75A provides the authoritative chip policy/inventory/effect registry.",
        "Reuse pure scenario math, but route activation/inventory/exclusivity through shared chip policy.",
    ),
    Asset(
        "squad_snapshot_backtest",
        "Decision state",
        "Legacy SquadSnapshot / comparison / decision backtest",
        (
            "backend/app/schemas/squad_snapshot.py",
            "backend/app/utils/squad_snapshot_compare.py",
            "backend/app/utils/decision_backtest.py",
        ),
        "MIGRATE",
        "Represents before/after squads and compares projected decision gain.",
        "Pydantic SquadSnapshot, Prediction ORM.",
        "15-player ids, captain/vice, bench order, bank, target_gw, model_name; legacy snapshot has no season field.",
        "Library callable.",
        (),
        "Day125B will define the versioned squad-state contract; legacy backtest Prediction query also lacks explicit season filtering.",
        "Reuse comparison concepts but migrate state and queries to a season-aware versioned contract.",
    ),
    Asset(
        "legacy_local_artifact_stores",
        "Ops / artifacts",
        "Legacy local run/decision/feature/model metadata stores",
        (
            "backend/app/utils/decision_run_store.py",
            "backend/app/utils/feature_snapshot_store.py",
            "backend/app/utils/run_snapshot_store.py",
            "backend/app/utils/model_metadata_store.py",
        ),
        "MIGRATE",
        "Stores JSON artifacts under legacy artifacts/* directories.",
        "Local filesystem and Pydantic schemas.",
        "Ad-hoc JSON file naming by run/model; local-only.",
        "Library callable; decision-runs API exposes one store.",
        ("legacy_decision_run_key", "legacy_run_snapshot_key", "legacy_model_metadata_key"),
        "Day76A introduces versioned artifact paths/storage interface and explicit legacy-key compatibility tests.",
        "Keep readable compatibility, but route new writes through the Day76A storage abstraction.",
    ),
    Asset(
        "decision_runs_api",
        "API",
        "Mounted legacy /decision-runs artifact API",
        ("backend/app/api/routes/decision_runs.py",),
        "WRAP",
        "Lists/loads legacy local decision-run JSON artifacts.",
        "decision_run_store.",
        "DecisionRunArtifact list/detail response.",
        "Mounted API.",
        (),
        "Day76A storage abstraction should become the source behind this surface.",
        "Keep useful observability surface, change storage backend through an adapter.",
    ),
    Asset(
        "refresh_checks",
        "Ops",
        "Legacy lightweight refresh validation",
        ("backend/ml/validation/run_refresh_checks.py",),
        "KEEP",
        "Checks null critical IDs, duplicate prediction keys, fixture GW coverage and target prediction counts.",
        "SessionLocal, season-aware SQL.",
        "Structured validation report for season/target_gw/model.",
        "CLI and library callable.",
        (),
        "Can be a reusable validation stage inside Day128+ canonical PRE/POST runner.",
        "Keep focused checks; extend rather than rewrite.",
    ),
    Asset(
        "weekly_refresh",
        "Ops",
        "Legacy one-command weekly refresh prototype",
        ("backend/ml/validation/run_weekly_refresh.py",),
        "MIGRATE",
        "Sequences live ingest, player prediction writers, match writer, validation and optional legacy snapshots.",
        "FastAPI HTTP endpoints, shell subprocesses, DB, legacy snapshot stores.",
        "target_gw/season-driven CLI with direct child commands and optional decision endpoint.",
        "CLI callable.",
        (),
        "Direct predecessor of Day128A/B + Day129A/B canonical PRE/POST/auto/resume runner.",
        "Reuse sequencing knowledge, but migrate to explicit DAG, artifact-first safety, resume/manifests and no silent final freeze.",
    ),
    Asset(
        "frontend_predictions",
        "Frontend",
        "Predictions page and BFF routes",
        (
            "frontend/src/app/predictions/page.tsx",
            "frontend/src/app/api/predictions/route.ts",
            "frontend/src/app/api/models/route.ts",
            "frontend/src/app/api/teams/route.ts",
        ),
        "WRAP",
        "Interactive player prediction explorer with model/team/filter controls.",
        "Next.js App Router; backend /predictions, /models, /teams.",
        "Consumes legacy prediction-row/model APIs.",
        "Routable current frontend surface.",
        (),
        "Future public prediction views should consume standard approved outputs while preserving useful filters.",
        "Keep UX; adapt BFF/backend contract rather than rebuilding the page from scratch.",
    ),
    Asset(
        "frontend_match_predictions",
        "Frontend",
        "Match predictions page and BFF routes",
        (
            "frontend/src/app/match-predictions/page.tsx",
            "frontend/src/app/api/match/models/route.ts",
            "frontend/src/app/api/match/predictions/route.ts",
        ),
        "WRAP",
        "Displays match prediction rows for selected GW/model.",
        "Next.js; backend /match/predictions/list and /match/predictions/models.",
        "Fixture/model rows with H/D/A probabilities/result.",
        "Routable current frontend surface.",
        (),
        "Should eventually point at approved Match Model outputs/frozen track record.",
        "Keep page and BFF shape, migrate authority/source.",
    ),
    Asset(
        "frontend_squad",
        "Frontend",
        "Legacy squad recommendation surfaces",
        (
            "frontend/src/app/page.tsx",
            "frontend/src/app/squad/page.tsx",
            "frontend/src/app/api/squad/route.ts",
        ),
        "MIGRATE",
        "Displays greedy recommendation squad from /recommendations/squad.",
        "Next.js; legacy recommendations API.",
        "Legacy squad response with starting_xi/bench/squad_list and cost summary.",
        "Routable current frontend surface.",
        (),
        "Overlaps Day101 model squad output and future Day125B stateful squad/current-team concepts.",
        "Reuse components/UX where useful; separate Model Team from imported/current user squad and use stateful decision contracts.",
    ),
    Asset(
        "frontend_captain",
        "Frontend",
        "Captain recommendation page and BFF route",
        (
            "frontend/src/app/captain/page.tsx",
            "frontend/src/app/api/recommendations/captain/route.ts",
        ),
        "MIGRATE",
        "Displays captain/vice candidates and explanations.",
        "Next.js; legacy /recommendations/captain endpoint.",
        "CaptainResponse with predicted points, labels, recent form, minutes stability and placeholder future factors.",
        "Routable current frontend surface.",
        (),
        "Overlaps Day101B lineup/captain output and future approved decision surface.",
        "Reuse presentation, switch to shared decision-state output.",
    ),
    Asset(
        "frontend_evaluation",
        "Frontend",
        "Evaluation page",
        ("frontend/src/app/evaluation/page.tsx",),
        "WRAP",
        "Shows model metadata/evaluation summaries.",
        "Next.js; backend /evaluation routes.",
        "Legacy metadata metrics and old production-default labels.",
        "Routable current frontend surface.",
        (),
        "GW1+ formal POST evaluation should provide track-record evidence instead of static seed metrics alone.",
        "Keep the page concept and adapt to live evaluation records.",
    ),
    Asset(
        "stray_backend_frontend_route",
        "Frontend",
        "Stray backend/frontend match prediction route copy",
        ("backend/frontend/app/api/match/predictions/route.ts",),
        "RETIRE_WITH_REASON",
        "Duplicate-looking Next.js route outside the current frontend/src application tree.",
        "Next.js-style source but located under backend/frontend.",
        "Unknown/legacy duplicate of match prediction proxy.",
        "Tracked file, but not part of frontend/src App Router tree.",
        (),
        "Current frontend has frontend/src/app/api/match/predictions/route.ts.",
        "Keep untouched on Day124B; mark for later archival/removal only after confirming no separate build references backend/frontend.",
    ),
)


EXPECTED_FRONTEND_GAPS: Tuple[Tuple[str, str], ...] = (
    ("model-squad", "No current tracked frontend route/page with 'model-squad' in its path."),
    ("transfers/targets", "No current tracked frontend route/page with 'transfers/targets' in its path."),
    ("recommendations page", "No dedicated current frontend recommendations page was found; captain/squad surfaces exist separately."),
    ("ops page", "No current tracked frontend route/page with 'ops' in its path."),
)


OVERLAPS: Tuple[Tuple[str, str, str], ...] = (
    (
        "Player prediction persistence/output",
        "Legacy Prediction rows and direct-writing baseline/ridge runners",
        "Day76B standard prediction contract + explicit legacy DB adapter; Day76D/GW1 artifact-first refresh",
    ),
    (
        "Match prediction persistence/output",
        "Legacy MatchPrediction rows and match_baseline_v0 route writer",
        "Day76B match/scoreline contracts + GW1 frozen Match Model artifacts",
    ),
    (
        "Squad legality",
        "recommendations.py / wildcard_builder.py / chips.py hardcoded quotas, formations, budget and club limits",
        "Day74B versioned squad/transfer rules + Day100A SquadLegalityEngine",
    ),
    (
        "Transfer economics/legality",
        "recommendations.py one-transfer/free-transfer/-4 logic",
        "Day74B app.rules.transfers + Day126A/B/127A roadmap",
    ),
    (
        "Chip policy",
        "chips.py + bench/triple/wildcard scenario-specific assumptions",
        "Day75A versioned chip rules/inventory/effect contract",
    ),
    (
        "Multi-GW planning",
        "wildcard_horizon.py aggregation over Prediction table",
        "Day79B fixture horizon + Day97A player prediction horizon with explicit missing-future semantics",
    ),
    (
        "Opening squad / lineup / captain",
        "legacy recommendations squad/captain and chip builders",
        "Day100B objective + Day101A opening squad optimizer + Day101B lineup/captain optimizer",
    ),
    (
        "Artifact storage",
        "app/utils/*_store.py ad-hoc artifacts directories",
        "Day76A versioned artifact paths/storage interface with legacy-key compatibility",
    ),
    (
        "Model metadata / activation",
        "legacy model_metadata JSON is_active/is_production_default flags",
        "Day76B registry plus v3.2 requirement that software deployment and model Champion activation are separate",
    ),
    (
        "Weekly orchestration",
        "run_weekly_refresh.py HTTP/shell sequence",
        "Day128A/B + Day129A/B planned canonical PRE/POST/auto/resume runner",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the Day124B legacy asset compatibility inventory."
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to auto-detection from this script.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Markdown output path. Defaults to ../private-planning/"
            "FPL_LEGACY_ASSET_COMPATIBILITY.md relative to the Git repo root. "
            "An explicitly supplied relative path is resolved from the repo root."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return non-zero if required current paths are unexpectedly missing or classifications are invalid.",
    )
    return parser.parse_args()


def detect_repo_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    script_path = Path(__file__).resolve()
    for parent in (script_path.parent,) + tuple(script_path.parents):
        if (parent / ".git").exists():
            return parent

    # Expected location: backend/ml/validation/<script>.py
    candidate = script_path.parents[3]
    return candidate.resolve()


def run_git(repo_root: Path, args: Sequence[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git"] + list(args),
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def tracked_files(repo_root: Path) -> List[str]:
    output = run_git(repo_root, ["ls-files"])
    if output is not None:
        return sorted(line for line in output.splitlines() if line.strip())

    rows: List[str] = []
    for path in repo_root.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            rows.append(path.relative_to(repo_root).as_posix())
    return sorted(rows)


def current_head(repo_root: Path) -> str:
    return run_git(repo_root, ["log", "-1", "--oneline"]) or "unavailable"


def git_status(repo_root: Path) -> str:
    output = run_git(repo_root, ["status", "--short"])
    if output is None:
        return "unavailable"
    return output or "clean"


def historical_paths(repo_root: Path) -> List[str]:
    output = run_git(
        repo_root,
        ["log", "--all", "--name-only", "--pretty=format:"],
    )
    if output is None:
        return []
    return sorted(set(line.strip() for line in output.splitlines() if line.strip()))


def read_text(repo_root: Path, rel_path: str) -> str:
    path = repo_root / rel_path
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def mounted_route_modules(repo_root: Path) -> Dict[str, bool]:
    main_text = read_text(repo_root, "backend/app/main.py")
    modules = {
        "predictions": "predictions_router",
        "recommendations": "recommendations_router",
        "match_predictions": "match_predictions_router",
        "team_form": "team_form_router",
        "chips": "chips.router",
        "decision_runs": "decision_runs.router",
        "evaluation": "evaluation.router",
        "ingest": "ingest_router",
        "ingest_gw_stats": "ingest_gw_stats_router",
        "gameweeks": "gameweeks_router",
        "models": "models.router",
        "h2h": "h2h_router",
    }
    return {
        name: ("include_router(%s)" % token) in main_text
        for name, token in modules.items()
    }


def direct_test_references(repo_root: Path, patterns: Sequence[str]) -> List[str]:
    if not patterns:
        return []

    tests_root = repo_root / "backend" / "tests"
    if not tests_root.exists():
        return []

    found: List[str] = []
    for path in tests_root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(pattern in text for pattern in patterns):
            found.append(path.relative_to(repo_root).as_posix())
    return sorted(set(found))


def extract_seed_model_metadata(repo_root: Path) -> List[Dict[str, str]]:
    rel = "backend/scripts/seed_model_metadata.py"
    text = read_text(repo_root, rel)
    if not text:
        return []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    rows: List[Dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name != "ModelMetadataArtifact":
            continue

        values: Dict[str, str] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                value = ast.literal_eval(keyword.value)
            except Exception:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[keyword.arg] = str(value)
            elif isinstance(value, dict):
                values[keyword.arg] = json.dumps(value, sort_keys=True)
        rows.append(values)
    return rows


def serialized_model_files(tracked: Iterable[str]) -> List[str]:
    suffixes = (
        ".pkl",
        ".pickle",
        ".joblib",
        ".onnx",
        ".pt",
        ".pth",
        ".cbm",
        ".bst",
    )
    return sorted(path for path in tracked if path.lower().endswith(suffixes))


def source_serialization_calls(repo_root: Path, tracked: Sequence[str]) -> List[str]:
    findings: List[str] = []
    patterns = (
        "joblib.dump",
        "pickle.dump",
        ".save_model(",
        "torch.save(",
    )
    for rel in tracked:
        if not rel.endswith(".py"):
            continue
        text = read_text(repo_root, rel)
        if any(pattern in text for pattern in patterns):
            findings.append(rel)
    return sorted(findings)


def markdown_escape(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def existence_label(repo_root: Path, paths: Sequence[str]) -> Tuple[str, List[str]]:
    missing = [path for path in paths if not (repo_root / path).is_file()]
    if not missing:
        return "present", []
    if len(missing) == len(paths):
        return "missing", missing
    return "partial", missing


def route_mount_note(asset: Asset, mounts: Dict[str, bool]) -> str:
    path_blob = " ".join(asset.paths)
    mapping = (
        ("backend/app/api/routes/predictions.py", "predictions"),
        ("backend/app/api/routes/recommendations.py", "recommendations"),
        ("backend/app/api/routes/match_predictions.py", "match_predictions"),
        ("backend/app/api/routes/team_form.py", "team_form"),
        ("backend/app/api/routes/chips.py", "chips"),
        ("backend/app/api/routes/decision_runs.py", "decision_runs"),
        ("backend/app/api/routes/evaluation.py", "evaluation"),
        ("backend/app/api/routes/ingest.py", "ingest"),
        ("backend/app/api/routes/ingest_gw_stats.py", "ingest_gw_stats"),
        ("backend/app/api/routes/gameweeks.py", "gameweeks"),
        ("backend/app/api/routes/models.py", "models"),
        ("backend/app/api/routes/h2h.py", "h2h"),
    )
    states = []
    for path_token, mount_key in mapping:
        if path_token in path_blob:
            states.append("%s=%s" % (mount_key, mounts.get(mount_key, False)))
    return ", ".join(states)


def frontend_gap_status(
    tracked: Sequence[str],
    history: Sequence[str],
    token: str,
) -> Tuple[str, List[str], List[str]]:
    """Check dedicated Next.js page continuity, not generic frontend/API paths."""

    token_lower = token.lower()
    current = sorted(
        path for path in tracked
        if path.startswith("frontend/src/app/")
        and path.endswith("/page.tsx")
        and token_lower in path.lower()
    )
    historical = sorted(
        path for path in history
        if "frontend" in path.lower()
        and path.lower().endswith("/page.tsx")
        and token_lower in path.lower()
    )
    if current:
        status = "present_currently"
    elif historical:
        status = "historical_page_only"
    else:
        status = "not_found_in_current_or_git_page_history"
    return status, current, historical


def predictions_page_behavior_signals(repo_root: Path) -> Tuple[List[str], str]:
    """Return conservative static clues about the current /predictions page."""

    rel = "frontend/src/app/predictions/page.tsx"
    text = read_text(repo_root, rel)
    if not text:
        return [], "source_unavailable"

    signals: List[str] = []

    if re.search(r"\.slice\(\s*0\s*,\s*20\s*\)", text):
        signals.append("explicit slice(0, 20)")

    top20_patterns = (
        r"top\s*20",
        r"top[_-]?20",
        r"limit\s*[:=]\s*20",
        r"pageSize\s*[:=]\s*20",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in top20_patterns):
        signals.append("explicit Top-20/limit-20 text or constant")

    position_markers = [
        marker
        for marker in ("GKP", "DEF", "MID", "FWD")
        if re.search(r"['\"]%s['\"]" % marker, text)
    ]
    if position_markers:
        signals.append("position markers: " + ", ".join(position_markers))

    if (
        re.search(r"predicted[_A-Za-z]*points", text, flags=re.IGNORECASE)
        and ".sort(" in text
    ):
        signals.append("predicted-points sorting logic")

    if (
        any("20" in signal for signal in signals)
        and len(position_markers) >= 3
    ):
        interpretation = "static evidence is consistent with a per-position Top-20 ranking view"
    elif signals:
        interpretation = (
            "ranking/filtering clues found, but static evidence is not sufficient to prove "
            "the exact per-position Top-20 behavior"
        )
    else:
        interpretation = (
            "no decisive Top-20-by-position signal found by conservative static scan; "
            "do not infer absence of the behavior from the route name alone"
        )

    return signals, interpretation


def validate_inventory(repo_root: Path) -> List[str]:
    errors: List[str] = []
    seen = set()
    for asset in ASSETS:
        if asset.asset_id in seen:
            errors.append("duplicate asset_id=%s" % asset.asset_id)
        seen.add(asset.asset_id)
        if asset.classification not in CLASSIFICATIONS:
            errors.append(
                "asset %s has invalid classification=%s"
                % (asset.asset_id, asset.classification)
            )

        state, _missing = existence_label(repo_root, asset.paths)
        if state == "missing":
            # Some historical/stray items may legitimately be gone. Do not
            # fail the whole export unless all paths of a foundational asset
            # expected to be current are missing.
            if asset.classification in ("KEEP", "WRAP", "MIGRATE"):
                errors.append(
                    "expected current asset %s has no present paths" % asset.asset_id
                )
    return errors


def build_report(repo_root: Path) -> str:
    tracked = tracked_files(repo_root)
    history = historical_paths(repo_root)
    mounts = mounted_route_modules(repo_root)
    head = current_head(repo_root)
    status = git_status(repo_root)
    model_rows = extract_seed_model_metadata(repo_root)
    model_files = serialized_model_files(tracked)
    serialization_calls = source_serialization_calls(repo_root, tracked)

    counts: Dict[str, int] = {key: 0 for key in CLASSIFICATIONS}
    for asset in ASSETS:
        counts[asset.classification] += 1

    lines: List[str] = []
    lines.extend(
        [
            "# FPL Legacy Asset Compatibility Inventory",
            "",
            "**Milestone:** Day124B — Day1–61 Legacy Asset Compatibility Inventory  ",
            "**Scope:** factual compatibility map; no legacy implementation is modified or deleted by this exporter.",
            "",
            "## Executive decision",
            "",
            "The Day1–61 codebase remains a substantial reusable foundation. The compatibility strategy is not a rewrite: preserve canonical data/schema and useful read surfaces; wrap stable diagnostics and UI contracts; migrate legacy direct writers and decision logic behind the Day71+ contracts/rules/artifact boundaries; retire only clearly superseded or unsafe duplicate paths with an explicit reason.",
            "",
            "## Source snapshot",
            "",
            "- Repository HEAD: `%s`" % head,
            "- Working tree at export time: `%s`" % markdown_escape(status),
            "- Tracked file count inspected: `%s`" % len(tracked),
            "- Static inspection only: `True`",
            "- Database access: `False`",
            "- Prediction/recommendation writes: `False`",
            "- Frozen artifact mutation: `False`",
            "",
            "## Classification summary",
            "",
            "| Classification | Count | Meaning |",
            "| --- | ---: | --- |",
            "| KEEP | %s | Reuse the current asset substantially as-is. |" % counts["KEEP"],
            "| WRAP | %s | Preserve the asset/API/algorithm but put a newer contract, service, storage, or activation boundary around it. |" % counts["WRAP"],
            "| MIGRATE | %s | Reuse important behavior/data, but move it to the Day71+ shared rules/contracts/state/runner architecture before making it authoritative. |" % counts["MIGRATE"],
            "| RETIRE_WITH_REASON | %s | Keep untouched for history on Day124B, but do not use as the future operational path. |" % counts["RETIRE_WITH_REASON"],
            "",
            "## Inventory",
            "",
            "| ID | Category | Asset | Current paths | Callable | Direct tracked-test evidence | Classification |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    detailed_rows: List[Tuple[Asset, str, List[str], List[str], str]] = []
    for asset in ASSETS:
        state, missing = existence_label(repo_root, asset.paths)
        tests = direct_test_references(repo_root, asset.test_patterns)
        mount_note = route_mount_note(asset, mounts)

        callable_text = asset.callability
        if mount_note:
            callable_text += "; mount check: " + mount_note

        path_text = "<br>".join("`%s`" % path for path in asset.paths)
        if missing:
            path_text += "<br>missing: " + ", ".join("`%s`" % p for p in missing)

        tests_text = "<br>".join("`%s`" % path for path in tests) if tests else "No direct tracked test reference found"

        lines.append(
            "| %s | %s | %s | %s | %s | %s | **%s** |"
            % (
                markdown_escape(asset.asset_id),
                markdown_escape(asset.category),
                markdown_escape(asset.name),
                path_text,
                markdown_escape(callable_text),
                tests_text,
                asset.classification,
            )
        )
        detailed_rows.append((asset, state, missing, tests, callable_text))

    lines.extend(
        [
            "",
            "## Detailed compatibility notes",
            "",
        ]
    )

    for asset, state, missing, tests, callable_text in detailed_rows:
        lines.extend(
            [
                "### %s — %s" % (asset.asset_id, asset.name),
                "",
                "- Classification: **%s**" % asset.classification,
                "- Current presence: `%s`" % state,
                "- Role: %s" % asset.role,
                "- Dependencies: %s" % asset.dependencies,
                "- Current data contract: %s" % asset.data_contract,
                "- Callability: %s" % callable_text,
                "- Direct tracked-test evidence: %s"
                % (
                    ", ".join("`%s`" % path for path in tests)
                    if tests
                    else "none found by direct static reference scan"
                ),
                "- Day71+ overlap: %s" % (asset.overlap or "none identified"),
                "- Decision rationale: %s" % asset.rationale,
            ]
        )
        if asset.notes:
            lines.append("- Notes: %s" % asset.notes)
        if missing:
            lines.append(
                "- Missing expected path(s): %s"
                % ", ".join("`%s`" % path for path in missing)
            )
        lines.append("")

    lines.extend(
        [
            "## Saved model metadata inventory",
            "",
            "The existing `seed_model_metadata.py` is historical metadata, not a current activation authority. Its `is_production_default` flags describe the prior Day56/57 system and must not silently override the frozen-live/Champion process.",
            "",
            "| Model | Task | Feature version | Status | Old production default | Metrics summary |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if model_rows:
        for row in model_rows:
            lines.append(
                "| `%s` | %s | %s | %s | %s | `%s` |"
                % (
                    markdown_escape(row.get("model_name", "")),
                    markdown_escape(row.get("task_type", "")),
                    markdown_escape(row.get("feature_version", "")),
                    markdown_escape(row.get("status", "")),
                    markdown_escape(row.get("is_production_default", "")),
                    markdown_escape(row.get("metrics_summary", "")),
                )
            )
    else:
        lines.append("| _No seed metadata rows parsed_ | | | | | |")

    lines.extend(
        [
            "",
            "### Serialized estimator check",
            "",
            "- Tracked serialized estimator files (`.pkl/.pickle/.joblib/.onnx/.pt/.pth/.cbm/.bst`): %s"
            % (
                ", ".join("`%s`" % p for p in model_files)
                if model_files
                else "**none found**"
            ),
            "- Python sources containing common model-serialization calls: %s"
            % (
                ", ".join("`%s`" % p for p in serialization_calls)
                if serialization_calls
                else "**none found**"
            ),
            "",
            "Interpretation: the v2/v2_1 training scripts in the current tree are primarily reproducible training/evaluation code and metadata evidence, not a complete persisted-estimator serving system. Future model work should preserve these baselines while adding explicit versioned train/infer artifacts instead of pretending a metadata JSON file is a deployable estimator.",
            "",
            "## Day1–61 ↔ Day71+ overlap map",
            "",
            "| Concept | Day1–61 / legacy implementation | Day71+ authority or overlap | Required compatibility action |",
            "| --- | --- | --- | --- |",
        ]
    )
    for concept, legacy, newer in OVERLAPS:
        lines.append(
            "| %s | %s | %s | Preserve approved behavior/evidence; route future authoritative use through the newer shared contract rather than duplicating the concept. |"
            % (
                markdown_escape(concept),
                markdown_escape(legacy),
                markdown_escape(newer),
            )
        )

    lines.extend(
        [
            "",
            "## Frontend continuity / missing-surface audit",
            "",
            "The current Next.js tree contains player predictions, match predictions, squad, captain and evaluation pages. The roadmap also names older/desired dedicated pages that are not visible in the current tracked page tree; these are recorded as route/page reconciliation gaps rather than silently assumed to exist. API/BFF routes do not by themselves count as a page.",
            "",
            "| Expected surface | Current/git-history status | Current matches | Historical path matches |",
            "| --- | --- | --- | --- |",
        ]
    )
    for token, _description in EXPECTED_FRONTEND_GAPS:
        search_token = token.split()[0]
        gap_status, current_matches, history_matches = frontend_gap_status(
            tracked, history, search_token
        )
        lines.append(
            "| %s | `%s` | %s | %s |"
            % (
                markdown_escape(token),
                gap_status,
                "<br>".join("`%s`" % p for p in current_matches) if current_matches else "none",
                "<br>".join("`%s`" % p for p in history_matches[:20]) if history_matches else "none",
            )
        )

    lines.extend(
        [
            "",
            "### Current frontend surfaces detected",
            "",
        ]
    )
    current_frontend_pages = sorted(
        path for path in tracked
        if path.startswith("frontend/src/app/") and path.endswith("/page.tsx")
    )
    for path in current_frontend_pages:
        lines.append("- `%s`" % path)

    prediction_signals, prediction_interpretation = predictions_page_behavior_signals(
        repo_root
    )
    lines.extend(
        [
            "",
            "### `/predictions` behavior reconciliation",
            "",
            "A missing dedicated `/recommendations` page does **not** imply that "
            "recommendation/ranking functionality is absent. The current `/predictions` "
            "page is inventoried separately and should be classified by its source behavior, "
            "not by route naming alone.",
            "",
            "- Source: `frontend/src/app/predictions/page.tsx`",
            "- Static behavior signals: %s"
            % (
                ", ".join("`%s`" % signal for signal in prediction_signals)
                if prediction_signals
                else "none conclusively detected"
            ),
            "- Interpretation: %s." % prediction_interpretation,
            "",
            "## Test-coverage conclusions",
            "",
            "1. The Day71+ contract/artifact/optimizer layer has focused `backend/tests/ml/...` coverage.",
            "2. Most mounted Day1–61 API routes and legacy model runners have no direct tracked unit-test reference in the current tree; they should therefore be treated as compatibility-sensitive during migration.",
            "3. `test_legacy_prediction_adapters.py` and Day76A artifact path/storage tests are important bridges: they prove the new layer already anticipates legacy compatibility instead of requiring a rewrite.",
            "4. Frontend unit/component tests were not found in the current tracked test inventory; production build checks should remain the minimum fast-CI frontend gate until focused UI tests are added where valuable.",
            "",
            "## Explicit migration boundaries",
            "",
            "- Do not delete or rewrite legacy code during Day124B.",
            "- Do not make legacy metadata `is_production_default` flags the current Champion authority.",
            "- Do not use direct-writing player/match scripts as the implicit live PRE/FREEZE pipeline.",
            "- Preserve `predictions` and `match_predictions` DB compatibility while immutable frozen artifacts remain the evaluation source of truth.",
            "- Preserve useful mounted API response shapes where practical, but move squad/transfer/chip legality to versioned rules and shared engines.",
            "- Preserve team-form/H2H and historical-ingest utilities unless a concrete later milestone proves them redundant.",
            "- Migrate old local artifact writers to Day76A storage through adapters rather than breaking old artifact readability.",
            "- Use Day125B+ versioned squad state before extending decision backtesting/current-team flows.",
            "",
            "## Day124B stop-point review",
            "",
            "- [x] Player-model families, match-model families, training/evaluation paths inventoried.",
            "- [x] Ingest/refresh jobs, DB writers/tables, recommendation/team-form/transfer/chip/ops utilities inventoried.",
            "- [x] Current frontend prediction/match/squad/captain/evaluation routes inventoried; missing named legacy/roadmap surfaces explicitly flagged.",
            "- [x] Dependencies, current contracts, callability and direct test evidence recorded.",
            "- [x] Every current inventory item classified as KEEP / WRAP / MIGRATE / RETIRE_WITH_REASON.",
            "- [x] Day1–61 vs Day71+ overlaps explicitly mapped.",
            "- [x] No legacy implementation or frozen artifact modified by the inventory process.",
            "",
            "> **Stop point:** the project has a reviewed compatibility map showing what prior work should be reused, wrapped, migrated, or intentionally retired. Classification is a future-action decision, not a Day124B code deletion instruction.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = detect_repo_root(args.repo_root)

    errors = validate_inventory(repo_root)
    report = build_report(repo_root)

    if args.out is None:
        out = repo_root.parent / "private-planning" / "FPL_LEGACY_ASSET_COMPATIBILITY.md"
    else:
        out = Path(args.out).expanduser()
        if not out.is_absolute():
            out = repo_root / out
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print("=== Day124B Legacy Asset Compatibility Inventory ===")
    print("repo_root:", repo_root)
    print("output:", out)
    print("asset_count:", len(ASSETS))
    for classification in CLASSIFICATIONS:
        count = sum(1 for asset in ASSETS if asset.classification == classification)
        print("%s: %s" % (classification, count))
    print("validation_errors:", len(errors))
    if errors:
        for error in errors:
            print("- %s" % error)

    if args.check and errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
