from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ml.validation.resolve_prediction_mode import resolve_prediction_mode


TEAM_PRIOR_STAT_COLUMNS = [
    "prev_season_clean_sheets",
    "prev_season_draws",
    "prev_season_goal_difference",
    "prev_season_goals_against",
    "prev_season_goals_for",
    "prev_season_losses",
    "prev_season_matches",
    "prev_season_total_points",
    "prev_season_wins",
    "prev_season_points_per_match",
    "prev_season_goals_for_per_match",
    "prev_season_goals_against_per_match",
    "prev_season_scheduled_fixtures",
    "prev_season_home_matches",
    "prev_season_away_matches",
    "prev_season_clean_sheet_rate",
]

REQUIRED_EFFECTIVE_STAT_COLUMNS = [
    "prev_season_matches",
    "prev_season_total_points",
    "prev_season_points_per_match",
    "prev_season_goals_for_per_match",
    "prev_season_goals_against_per_match",
    "prev_season_goal_difference",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Define and apply a conservative promoted-team fallback policy for Pre-GW1 match features. "
            "This is read-only and does not generate predictions or write to the database."
        )
    )
    parser.add_argument("--source-season", required=True, help="Prior/source season, for example 2024_25.")
    parser.add_argument("--target-season", required=True, help="Target season, for example 2025_26.")
    parser.add_argument("--target-gw", type=int, default=1)
    parser.add_argument(
        "--prediction-mode",
        default="auto",
        choices=["auto", "pre_gw1_prior", "early_season_blend", "normal_weekly"],
    )
    parser.add_argument("--pre-gw1-match-features-csv", required=True, help="Day69A match feature preview CSV.")
    parser.add_argument("--team-prior-csv", required=True, help="Day68A team prior CSV.")
    parser.add_argument("--team-mapping-csv", required=True, help="Day68B team mapping candidate CSV.")
    parser.add_argument(
        "--fallback-policy",
        default="relegated_historical_only_mean_v0",
        choices=["relegated_historical_only_mean_v0"],
        help="Fallback policy for target-season teams without accepted previous-season EPL priors.",
    )
    parser.add_argument("--out-csv", required=True, help="Output match feature CSV with effective fallback features.")
    parser.add_argument("--out-json", required=True, help="Output JSON report.")
    parser.add_argument("--out-md", required=True, help="Output Markdown report.")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_raw_id(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    if text_value.endswith(".0"):
        text_value = text_value[:-2]
    return text_value


def nullable_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    text_value = str(value).strip().lower()
    return text_value in {"true", "1", "yes", "y"}


def load_csv(path_value: str, required_columns: Sequence[str], label: str) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError("%s CSV does not exist: %s" % (label, path))

    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("%s CSV is empty: %s" % (label, path))

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise RuntimeError("%s CSV missing required columns: %s" % (label, missing))

    return df


def filter_source_target(df: pd.DataFrame, source_season: str, target_season: str, label: str) -> pd.DataFrame:
    required = ["source_season", "target_season"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError("%s missing required season columns: %s" % (label, missing))

    out = df[
        (df["source_season"].astype(str) == str(source_season))
        & (df["target_season"].astype(str) == str(target_season))
    ].copy()

    if out.empty:
        raise RuntimeError(
            "%s has no rows for source_season=%s target_season=%s."
            % (label, source_season, target_season)
        )

    return out


def load_inputs(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = load_csv(
        args.pre_gw1_match_features_csv,
        required_columns=[
            "source_season",
            "target_season",
            "target_gw",
            "prediction_mode",
            "fixture_id",
            "home_team_short_name",
            "away_team_short_name",
            "home_team_prior_status",
            "away_team_prior_status",
            "home_has_prev_season_team_prior",
            "away_has_prev_season_team_prior",
        ],
        label="Pre-GW1 match features",
    )
    features = filter_source_target(features, args.source_season, args.target_season, "Pre-GW1 match features")
    features = features[features["target_gw"].astype(int) == int(args.target_gw)].copy()
    if features.empty:
        raise RuntimeError("No Pre-GW1 feature rows found for target_gw=%s." % args.target_gw)

    team_priors = load_csv(
        args.team_prior_csv,
        required_columns=[
            "source_season",
            "target_season",
            "raw_team_id",
            "raw_team_name",
            "raw_team_short_name",
            "prior_source",
            "prior_identity_scope",
        ],
        label="Team priors",
    )
    team_priors = filter_source_target(team_priors, args.source_season, args.target_season, "Team priors")
    team_priors["raw_team_id"] = team_priors["raw_team_id"].apply(normalize_raw_id)

    team_mapping = load_csv(
        args.team_mapping_csv,
        required_columns=[
            "source_season",
            "target_season",
            "raw_team_id",
            "raw_team_name",
            "raw_team_short_name",
            "candidate_rank",
            "candidate_team_id",
            "candidate_team_name",
            "candidate_team_short_name",
            "match_status",
            "is_auto_approved",
            "needs_manual_review",
            "safe_team_match_for_auto_approval",
        ],
        label="Team mapping",
    )
    team_mapping = filter_source_target(team_mapping, args.source_season, args.target_season, "Team mapping")
    team_mapping["raw_team_id"] = team_mapping["raw_team_id"].apply(normalize_raw_id)

    return features, team_priors, team_mapping


def top_mapping_rows(team_mapping: pd.DataFrame) -> pd.DataFrame:
    return team_mapping[
        (team_mapping["candidate_rank"].isna())
        | (team_mapping["candidate_rank"] == 1)
    ].copy()


def build_fallback_basis(team_priors: pd.DataFrame, team_mapping: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    top = top_mapping_rows(team_mapping)
    historical_only = top[top["match_status"] == "historical_only_unmatched"].copy()

    basis_raw_ids = set(historical_only["raw_team_id"].dropna().astype(str).tolist())

    basis = team_priors[team_priors["raw_team_id"].astype(str).isin(basis_raw_ids)].copy()

    diagnostics = {
        "basis_raw_team_ids": sorted(basis["raw_team_id"].dropna().astype(str).tolist()),
        "basis_raw_team_names": sorted(basis["raw_team_name"].dropna().astype(str).tolist()),
        "basis_raw_team_short_names": sorted(basis["raw_team_short_name"].dropna().astype(str).tolist()),
        "basis_team_count": int(len(basis)),
        "historical_only_mapping_count": int(len(historical_only)),
        "missing_basis_prior_count": int(len(basis_raw_ids) - len(basis)),
    }

    return basis, diagnostics


def build_fallback_values(fallback_basis: pd.DataFrame) -> Dict[str, Any]:
    values: Dict[str, Any] = {}

    for stat_col in TEAM_PRIOR_STAT_COLUMNS:
        if stat_col not in fallback_basis.columns:
            values[stat_col] = None
            continue

        series = pd.to_numeric(fallback_basis[stat_col], errors="coerce")
        if series.notna().any():
            values[stat_col] = round(float(series.mean()), 4)
        else:
            values[stat_col] = None

    return values


def target_teams_requiring_fallback(features: pd.DataFrame) -> List[str]:
    names = set()

    for side in ["home", "away"]:
        status_col = side + "_team_prior_status"
        short_col = side + "_team_short_name"
        has_prior_col = side + "_has_prev_season_team_prior"

        missing_mask = (
            (features[status_col].astype(str) == "target_only_or_unmapped_prior_missing")
            | (~features[has_prior_col].apply(bool_value))
        )
        names.update(features.loc[missing_mask, short_col].dropna().astype(str).tolist())

    return sorted(names)


def apply_effective_features(
    features: pd.DataFrame,
    fallback_values: Dict[str, Any],
    fallback_basis_diagnostics: Dict[str, Any],
    fallback_policy: str,
) -> pd.DataFrame:
    result = features.copy()
    fallback_basis_short_names = ",".join(fallback_basis_diagnostics["basis_raw_team_short_names"])

    for side in ["home", "away"]:
        has_prior_col = side + "_has_prev_season_team_prior"
        prior_status_col = side + "_team_prior_status"

        fallback_applied_values: List[bool] = []
        feature_status_values: List[str] = []
        feature_source_values: List[str] = []
        feature_values_present: List[bool] = []
        fallback_basis_values: List[Optional[str]] = []

        for _, row in result.iterrows():
            has_prior = bool_value(row.get(has_prior_col))
            prior_status = str(row.get(prior_status_col))

            should_apply_fallback = (not has_prior) or prior_status == "target_only_or_unmapped_prior_missing"

            fallback_applied_values.append(bool(should_apply_fallback))
            if should_apply_fallback:
                feature_status_values.append("promoted_team_fallback_v0")
                feature_source_values.append(fallback_policy)
                fallback_basis_values.append(fallback_basis_short_names)
            else:
                feature_status_values.append("reliable_previous_season_prior")
                feature_source_values.append("historical_staging")
                fallback_basis_values.append(None)

        result[side + "_team_fallback_applied"] = fallback_applied_values
        result[side + "_effective_team_feature_status"] = feature_status_values
        result[side + "_effective_team_feature_source"] = feature_source_values
        result[side + "_fallback_policy"] = [
            fallback_policy if applied else None for applied in fallback_applied_values
        ]
        result[side + "_fallback_basis_raw_team_short_names"] = fallback_basis_values

        for stat_col in TEAM_PRIOR_STAT_COLUMNS:
            original_col = side + "_" + stat_col
            effective_col = side + "_effective_" + stat_col

            if original_col not in result.columns:
                result[effective_col] = fallback_values.get(stat_col)
                continue

            values = []
            for _, row in result.iterrows():
                if bool_value(row.get(side + "_team_fallback_applied")):
                    values.append(fallback_values.get(stat_col))
                else:
                    values.append(row.get(original_col))
            result[effective_col] = values

        required_present = []
        for _, row in result.iterrows():
            has_all_required = True
            for stat_col in REQUIRED_EFFECTIVE_STAT_COLUMNS:
                value = row.get(side + "_effective_" + stat_col)
                if value is None or pd.isna(value):
                    has_all_required = False
                    break
            required_present.append(has_all_required)

        result[side + "_has_effective_team_feature_values"] = required_present

    result["any_team_fallback_applied"] = (
        result["home_team_fallback_applied"].apply(bool_value)
        | result["away_team_fallback_applied"].apply(bool_value)
    )
    result["both_teams_have_effective_team_features"] = (
        result["home_has_effective_team_feature_values"].apply(bool_value)
        & result["away_has_effective_team_feature_values"].apply(bool_value)
    )

    add_effective_diff_columns(result)
    return result


def add_effective_diff_columns(result: pd.DataFrame) -> None:
    diff_specs = [
        ("prev_season_points_per_match", "effective_prior_points_per_match_diff_home_minus_away"),
        ("prev_season_goals_for_per_match", "effective_prior_goals_for_per_match_diff_home_minus_away"),
        ("prev_season_goals_against_per_match", "effective_prior_goals_against_per_match_diff_home_minus_away"),
        ("prev_season_clean_sheet_rate", "effective_prior_clean_sheet_rate_diff_home_minus_away"),
        ("prev_season_goal_difference", "effective_prior_goal_difference_diff_home_minus_away"),
        ("prev_season_total_points", "effective_prior_total_points_diff_home_minus_away"),
    ]

    for stat_col, out_col in diff_specs:
        home_col = "home_effective_" + stat_col
        away_col = "away_effective_" + stat_col

        values = []
        for _, row in result.iterrows():
            home_value = nullable_float(row.get(home_col))
            away_value = nullable_float(row.get(away_col))
            if home_value is None or away_value is None:
                values.append(None)
            else:
                values.append(round(home_value - away_value, 4))
        result[out_col] = values


def build_report(
    args: argparse.Namespace,
    features_before: pd.DataFrame,
    features_after: pd.DataFrame,
    fallback_basis: pd.DataFrame,
    fallback_basis_diagnostics: Dict[str, Any],
    fallback_values: Dict[str, Any],
    target_fallback_teams: List[str],
    mode_result: Dict[str, Any],
    blockers: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    if len(features_before) != len(features_after):
        blockers.append("Feature row count changed after applying fallback policy.")

    if int(features_after["fixture_id"].dropna().duplicated().sum()) > 0:
        blockers.append("Duplicate fixture_id values found after fallback policy.")

    if fallback_basis_diagnostics["basis_team_count"] == 0:
        blockers.append("Fallback basis is empty; cannot compute promoted-team fallback values.")

    if fallback_basis_diagnostics["missing_basis_prior_count"] != 0:
        blockers.append("Some historical-only mapping rows are missing team prior rows.")

    if fallback_basis_diagnostics["basis_team_count"] != 3:
        warnings.append(
            "Fallback basis team count is %s, expected 3 for a standard Premier League season transition."
            % fallback_basis_diagnostics["basis_team_count"]
        )

    missing_required_effective = {}
    for side in ["home", "away"]:
        side_missing = {}
        for stat_col in REQUIRED_EFFECTIVE_STAT_COLUMNS:
            col = side + "_effective_" + stat_col
            if col in features_after.columns:
                side_missing[stat_col] = int(features_after[col].isna().sum())
            else:
                side_missing[stat_col] = len(features_after)
        missing_required_effective[side] = side_missing

    for side, side_missing in missing_required_effective.items():
        for stat_col, missing_count in side_missing.items():
            if missing_count != 0:
                blockers.append("%s effective %s has %s missing values." % (side, stat_col, missing_count))

    fallback_fixture_count = int(features_after["any_team_fallback_applied"].sum())
    both_effective_count = int(features_after["both_teams_have_effective_team_features"].sum())

    if both_effective_count != len(features_after):
        blockers.append("Not all fixtures have effective team feature values after fallback policy.")

    if not target_fallback_teams:
        warnings.append("No target-season teams required fallback. Policy output is still valid but no fallback was applied.")

    return {
        "created_at": utc_now(),
        "source_season": args.source_season,
        "target_season": args.target_season,
        "target_gw": args.target_gw,
        "requested_prediction_mode": args.prediction_mode,
        "resolved_prediction_mode": mode_result["resolved_prediction_mode"],
        "fallback_policy": args.fallback_policy,
        "fallback_policy_version": "day69b_v0",
        "audit_only": True,
        "writes_database": False,
        "passed": len(blockers) == 0,
        "ready_for_pre_gw1_match_features_with_fallback": len(blockers) == 0,
        "ready_for_full_pre_gw1_match_prediction": False,
        "ready_for_prediction_write": False,
        "reason_prediction_not_ready": (
            "Day69B defines and applies a transparent fallback feature policy only. "
            "Match probability scoring, model selection, guardrails, calibration, and prediction manifest are not implemented here."
        ),
        "inputs": {
            "pre_gw1_match_features_csv": args.pre_gw1_match_features_csv,
            "team_prior_csv": args.team_prior_csv,
            "team_mapping_csv": args.team_mapping_csv,
            "out_csv": args.out_csv,
            "out_json": args.out_json,
            "out_md": args.out_md,
        },
        "mode_resolution": mode_result,
        "row_counts": {
            "features_before_rows": int(len(features_before)),
            "features_after_rows": int(len(features_after)),
            "fallback_basis_team_rows": int(len(fallback_basis)),
        },
        "fallback_basis": fallback_basis_diagnostics,
        "fallback_values": fallback_values,
        "fallback_application_summary": {
            "target_teams_requiring_fallback": target_fallback_teams,
            "target_teams_requiring_fallback_count": int(len(target_fallback_teams)),
            "fixtures_with_any_fallback_applied": fallback_fixture_count,
            "fixtures_with_both_effective_team_features": both_effective_count,
            "home_fallback_count": int(features_after["home_team_fallback_applied"].sum()),
            "away_fallback_count": int(features_after["away_team_fallback_applied"].sum()),
            "missing_required_effective_values": missing_required_effective,
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "Reliable unchanged teams keep their historical staging previous-season team priors.",
            "Target-season teams without accepted EPL previous-season priors receive a transparent v0 fallback.",
            "The v0 fallback is the mean of historical-only source-season teams, i.e. the teams that left the target-season league set.",
            "The fallback is not a learned promoted-team strength model and should be replaced or calibrated later when better promoted-team data exists.",
            "Original has_prev_season_team_prior flags remain false for fallback teams; effective_* columns carry the fallback values.",
        ],
    }


def write_json(report: Dict[str, Any], out_json: str) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_markdown(report: Dict[str, Any], out_md: str) -> None:
    lines: List[str] = []
    lines.append("# Day69B — Promoted-Team Fallback Policy for Pre-GW1 Match Features")
    lines.append("")
    lines.append("Created at: `%s`" % report["created_at"])
    lines.append("")
    lines.append("Source season: `%s`" % report["source_season"])
    lines.append("Target season: `%s`" % report["target_season"])
    lines.append("Target GW: `%s`" % report["target_gw"])
    lines.append("Resolved prediction mode: `%s`" % report["resolved_prediction_mode"])
    lines.append("Fallback policy: `%s`" % report["fallback_policy"])
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Passed: `%s`" % report["passed"])
    lines.append("- Audit only: `%s`" % report["audit_only"])
    lines.append("- Writes database: `%s`" % report["writes_database"])
    lines.append("- Ready for Pre-GW1 match features with fallback: `%s`" % report["ready_for_pre_gw1_match_features_with_fallback"])
    lines.append("- Ready for full Pre-GW1 match prediction: `%s`" % report["ready_for_full_pre_gw1_match_prediction"])
    lines.append("- Ready for prediction write: `%s`" % report["ready_for_prediction_write"])
    lines.append("")
    lines.append("## Row Counts")
    lines.append("")
    for key, value in report["row_counts"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Fallback Basis")
    lines.append("")
    for key, value in report["fallback_basis"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Fallback Values")
    lines.append("")
    for key, value in report["fallback_values"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Fallback Application Summary")
    lines.append("")
    for key, value in report["fallback_application_summary"].items():
        lines.append("- %s: `%s`" % (key, value))
    lines.append("")
    lines.append("## Blockers")
    lines.append("")
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append("- %s" % blocker)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append("- %s" % warning)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in report["notes"]:
        lines.append("- %s" % note)
    lines.append("")

    path = Path(out_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(report: Dict[str, Any]) -> None:
    print("=== Day69B Promoted-Team Fallback Policy ===")
    print("source_season:", report["source_season"])
    print("target_season:", report["target_season"])
    print("target_gw:", report["target_gw"])
    print("resolved_prediction_mode:", report["resolved_prediction_mode"])
    print("fallback_policy:", report["fallback_policy"])
    print("fallback_policy_version:", report["fallback_policy_version"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print("ready_for_pre_gw1_match_features_with_fallback:", report["ready_for_pre_gw1_match_features_with_fallback"])
    print("ready_for_full_pre_gw1_match_prediction:", report["ready_for_full_pre_gw1_match_prediction"])
    print("ready_for_prediction_write:", report["ready_for_prediction_write"])
    print("saved_csv:", report["inputs"]["out_csv"])
    print("saved_json:", report["inputs"]["out_json"])
    print("saved_md:", report["inputs"]["out_md"])
    print()
    print("Row counts:")
    for key, value in report["row_counts"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Fallback basis:")
    for key, value in report["fallback_basis"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Fallback values:")
    for key, value in report["fallback_values"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Fallback application summary:")
    for key, value in report["fallback_application_summary"].items():
        print("- %s: %s" % (key, value))
    print()
    print("Blockers:", report["blockers"] or "none")
    print("Warnings:", report["warnings"] or "none")


def main() -> None:
    args = parse_args()

    blockers: List[str] = []
    warnings: List[str] = []

    mode_result = resolve_prediction_mode(
        season=args.target_season,
        target_gw=args.target_gw,
        requested_prediction_mode=args.prediction_mode,
        prior_season=args.source_season,
        stabilization_gw=6,
        allow_experimental_mode=False,
    )

    if not mode_result["valid"]:
        blockers.append("Prediction mode resolver returned invalid: %s" % mode_result["errors"])
    if mode_result["resolved_prediction_mode"] != "pre_gw1_prior":
        blockers.append(
            "Day69B expects resolved_prediction_mode=pre_gw1_prior, got %s."
            % mode_result["resolved_prediction_mode"]
        )
    if args.target_gw != 1:
        blockers.append("Day69B expects target_gw=1, got %s." % args.target_gw)

    features, team_priors, team_mapping = load_inputs(args)

    fallback_basis, fallback_basis_diagnostics = build_fallback_basis(team_priors, team_mapping)
    fallback_values = build_fallback_values(fallback_basis)
    target_fallback_teams = target_teams_requiring_fallback(features)

    features_after = apply_effective_features(
        features=features,
        fallback_values=fallback_values,
        fallback_basis_diagnostics=fallback_basis_diagnostics,
        fallback_policy=args.fallback_policy,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    features_after.to_csv(out_csv, index=False)

    report = build_report(
        args=args,
        features_before=features,
        features_after=features_after,
        fallback_basis=fallback_basis,
        fallback_basis_diagnostics=fallback_basis_diagnostics,
        fallback_values=fallback_values,
        target_fallback_teams=target_fallback_teams,
        mode_result=mode_result,
        blockers=blockers,
        warnings=warnings,
    )

    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_report(report)

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
