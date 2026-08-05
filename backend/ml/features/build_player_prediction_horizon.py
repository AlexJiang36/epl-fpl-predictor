from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


ARTIFACT_TYPE = "player_prediction_horizon"
HORIZON_VERSION = "day97a_v1"
HORIZON_SCHEMA_VERSION = "fpl_player_prediction_horizon_v1"
DEFAULT_TARGET_SEASON = "2026_27"
DEFAULT_START_GW = 1
DEFAULT_HORIZON = 5
DEFAULT_DISCOUNTS = {
    1: 1.00,
    2: 0.85,
    3: 0.70,
    4: 0.55,
    5: 0.40,
}
OBJECTIVE_GW1_ONLY = "gw1_only_fallback"
OBJECTIVE_MULTI_GW = "gw1_gw5_preview"
FUTURE_CONTEXT_MANUAL = "manual_review_only"
RECOMMENDATION_STATUS = "preview_only"


class PlayerPredictionHorizonError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def nonempty_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise PlayerPredictionHorizonError("%s must be non-empty." % label)
    return text


def nullable_int(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def nullable_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_risk_flags(value: Any) -> Tuple[str, ...]:
    if value is None or (not isinstance(value, (list, tuple, set, dict)) and pd.isna(value)):
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = [item.strip() for item in text.split(",") if item.strip()]
        value = parsed
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


def normalize_scope(
    target_season: str,
    start_gw: int,
    horizon: int,
) -> Tuple[str, int, int, int]:
    season = nonempty_text(target_season, "target_season")
    start = int(start_gw)
    length = int(horizon)
    if start < 1 or start > 38:
        raise PlayerPredictionHorizonError("start_gw must be from 1 to 38.")
    if length < 1:
        raise PlayerPredictionHorizonError("horizon must be at least 1.")
    end = start + length - 1
    if end > 38:
        raise PlayerPredictionHorizonError(
            "start_gw + horizon - 1 must not exceed Gameweek 38."
        )
    return season, start, length, end


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path.expanduser().resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PlayerPredictionHorizonError("%s must contain a JSON object." % path)
    return payload


def require_columns(
    dataframe: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(required) - set(dataframe.columns))
    if missing:
        raise PlayerPredictionHorizonError(
            "%s is missing required columns: %s" % (label, missing)
        )


def validate_source_reports(
    prediction_report: Mapping[str, Any],
    fixture_report: Mapping[str, Any],
    target_season: str,
    start_gw: int,
    horizon: int,
) -> None:
    if str(prediction_report.get("target_season")) != target_season:
        raise PlayerPredictionHorizonError("Day76D target season does not match.")
    if int(prediction_report.get("target_gw", 0)) != start_gw:
        raise PlayerPredictionHorizonError("Day76D target GW does not match.")
    if prediction_report.get("stop_point_satisfied") is not True:
        raise PlayerPredictionHorizonError("Day76D stop point is not satisfied.")
    if prediction_report.get("writes_database") is not False:
        raise PlayerPredictionHorizonError("Day76D must remain read-only.")
    if prediction_report.get("production_approved") is not False:
        raise PlayerPredictionHorizonError(
            "Day76D preview cannot be production approved."
        )

    if str(fixture_report.get("target_season")) != target_season:
        raise PlayerPredictionHorizonError("Day79B target season does not match.")
    if int(fixture_report.get("start_gw", 0)) != start_gw:
        raise PlayerPredictionHorizonError("Day79B start GW does not match.")
    if int(fixture_report.get("horizon", 0)) != horizon:
        raise PlayerPredictionHorizonError("Day79B horizon does not match.")
    if fixture_report.get("ready_for_day97a") is not True:
        raise PlayerPredictionHorizonError("Day79B is not ready for Day97A.")
    if fixture_report.get("stop_point_satisfied") is not True:
        raise PlayerPredictionHorizonError("Day79B stop point is not satisfied.")
    if fixture_report.get("writes_database") is not False:
        raise PlayerPredictionHorizonError("Day79B must remain read-only.")


def load_prediction_rows(path: Path, target_season: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path, low_memory=False)
    require_columns(
        dataframe,
        [
            "target_season",
            "target_gw",
            "player_id",
            "team_id",
            "position",
            "now_cost",
            "has_fixture",
            "selection_eligible",
            "appearance_probability",
            "start_probability",
            "expected_minutes",
            "predicted_points",
            "fallback_used",
            "fallback_level",
            "risk_flags",
            "readiness_status",
            "production_ready",
            "prediction_write_allowed",
        ],
        "standard player predictions",
    )
    dataframe = dataframe[
        dataframe["target_season"].astype(str) == target_season
    ].copy()
    if dataframe.empty:
        raise PlayerPredictionHorizonError(
            "No standard prediction rows found for target season."
        )
    dataframe["player_id"] = dataframe["player_id"].apply(nullable_int)
    dataframe["team_id"] = dataframe["team_id"].apply(nullable_int)
    if dataframe["player_id"].isna().any():
        raise PlayerPredictionHorizonError("Prediction player IDs must be present.")
    if dataframe["player_id"].duplicated().any():
        raise PlayerPredictionHorizonError(
            "Standard prediction rows contain duplicate player IDs."
        )
    if any(bool_value(value) for value in dataframe["production_ready"]):
        raise PlayerPredictionHorizonError(
            "Preview source cannot contain production-ready rows."
        )
    if any(bool_value(value) for value in dataframe["prediction_write_allowed"]):
        raise PlayerPredictionHorizonError(
            "Preview source cannot enable prediction writes."
        )
    return dataframe


def load_fixture_context(
    path: Path,
    target_season: str,
    start_gw: int,
    end_gw: int,
) -> pd.DataFrame:
    dataframe = pd.read_csv(path, low_memory=False)
    require_columns(
        dataframe,
        [
            "target_season",
            "gameweek",
            "player_id",
            "team_id",
            "position",
            "price_units",
            "fixture_id",
            "fixture_count_for_team_gw",
            "has_fixture",
            "blank_gw_flag",
            "double_gw_flag",
            "opponent_team_id",
            "opponent_team_name",
            "opponent_team_short_name",
            "is_home",
            "kickoff_time_utc",
            "kickoff_time_status",
            "kickoff_change_status",
            "player_fixture_eligible",
            "eligibility_reason",
            "manual_review_required",
        ],
        "player fixture eligibility",
    )
    dataframe = dataframe[
        (dataframe["target_season"].astype(str) == target_season)
        & (pd.to_numeric(dataframe["gameweek"], errors="coerce") >= start_gw)
        & (pd.to_numeric(dataframe["gameweek"], errors="coerce") <= end_gw)
    ].copy()
    dataframe["player_id"] = dataframe["player_id"].apply(nullable_int)
    dataframe["team_id"] = dataframe["team_id"].apply(nullable_int)
    dataframe["gameweek"] = pd.to_numeric(
        dataframe["gameweek"], errors="raise"
    ).astype(int)
    if dataframe.duplicated(["player_id", "gameweek"]).any():
        raise PlayerPredictionHorizonError(
            "Fixture context contains duplicate player-Gameweek rows. "
            "Accelerated Day97A supports at most one fixture per team-Gameweek."
        )
    return dataframe


def prediction_lookup(
    predictions: pd.DataFrame,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    result: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in predictions.to_dict(orient="records"):
        key = (int(row["player_id"]), int(row["target_gw"]))
        if key in result:
            raise PlayerPredictionHorizonError(
                "Duplicate prediction row for player_id=%s target_gw=%s." % key
            )
        result[key] = row
    return result


def discount_schedule(
    start_gw: int,
    horizon: int,
    provided: Optional[Mapping[int, float]] = None,
) -> Dict[int, float]:
    if provided is not None:
        schedule = {int(key): float(value) for key, value in provided.items()}
    else:
        schedule = {
            gw: DEFAULT_DISCOUNTS.get(index + 1, max(0.0, 1.0 - 0.15 * index))
            for index, gw in enumerate(range(start_gw, start_gw + horizon))
        }
    required = set(range(start_gw, start_gw + horizon))
    if set(schedule) != required:
        raise PlayerPredictionHorizonError(
            "Discount schedule must contain exactly the requested Gameweeks."
        )
    if any(value < 0.0 for value in schedule.values()):
        raise PlayerPredictionHorizonError("Discounts must be non-negative.")
    return schedule


def build_long_horizon(
    predictions: pd.DataFrame,
    fixture_context: pd.DataFrame,
    start_gw: int,
    horizon: int,
    discounts: Mapping[int, float],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    lookup = prediction_lookup(predictions)
    rows: List[Dict[str, Any]] = []
    missing_prediction_count = 0
    future_missing_count = 0
    available_prediction_count = 0

    for fixture_row in fixture_context.to_dict(orient="records"):
        player_id = int(fixture_row["player_id"])
        gameweek = int(fixture_row["gameweek"])
        source = lookup.get((player_id, gameweek))
        source_is_available = source is not None
        has_fixture = bool_value(fixture_row["has_fixture"])
        fixture_eligible = bool_value(fixture_row["player_fixture_eligible"])
        current_eligible = bool_value(fixture_row.get("selection_eligible"))

        if source_is_available:
            predicted_points = nullable_float(source.get("predicted_points"))
            expected_minutes = nullable_float(source.get("expected_minutes"))
            start_probability = nullable_float(source.get("start_probability"))
            appearance_probability = nullable_float(
                source.get("appearance_probability")
            )
            uncertainty_lower = nullable_float(source.get("uncertainty_lower"))
            uncertainty_upper = nullable_float(source.get("uncertainty_upper"))
            fallback_used = bool_value(source.get("fallback_used"))
            fallback_level = nullable_int(source.get("fallback_level"))
            risk_flags = normalize_risk_flags(source.get("risk_flags"))
            readiness_status = str(
                source.get("readiness_status") or RECOMMENDATION_STATUS
            )
            source_prediction_run_id = str(source.get("run_id") or "")
            prediction_status = "available_preview"
            available_prediction_count += 1
        else:
            predicted_points = None
            expected_minutes = None
            start_probability = None
            appearance_probability = None
            uncertainty_lower = None
            uncertainty_upper = None
            fallback_used = True
            fallback_level = 4
            risk_flags = ("missing_future_prediction",)
            readiness_status = RECOMMENDATION_STATUS
            source_prediction_run_id = ""
            prediction_status = (
                "missing_future_prediction"
                if gameweek > start_gw
                else "missing_required_gw1_prediction"
            )
            missing_prediction_count += 1
            if gameweek > start_gw:
                future_missing_count += 1

        if not has_fixture:
            row_status = "blank_fixture_context"
        elif not fixture_eligible:
            row_status = "player_fixture_ineligible"
        else:
            row_status = prediction_status

        discounted_points = (
            None
            if predicted_points is None
            else float(predicted_points) * float(discounts[gameweek])
        )
        manual_review = (
            bool_value(fixture_row.get("manual_review_required"))
            or predicted_points is None
            or not has_fixture
        )

        rows.append(
            {
                "contract_version": HORIZON_SCHEMA_VERSION,
                "artifact_type": ARTIFACT_TYPE,
                "horizon_version": HORIZON_VERSION,
                "target_season": str(fixture_row["target_season"]),
                "start_gw": start_gw,
                "horizon": horizon,
                "end_gw": start_gw + horizon - 1,
                "target_gw": gameweek,
                "gameweek_discount": float(discounts[gameweek]),
                "player_id": player_id,
                "fpl_player_id": nullable_int(
                    source.get("fpl_player_id") if source else fixture_row.get("player_code")
                ),
                "player_name": str(
                    source.get("player_name") if source else fixture_row.get("player_name")
                ),
                "web_name": str(
                    source.get("web_name") if source else fixture_row.get("web_name")
                ),
                "team_id": int(fixture_row["team_id"]),
                "team_name": str(fixture_row.get("team_name") or ""),
                "team_short_name": str(fixture_row.get("team_short_name") or ""),
                "position": str(fixture_row.get("position") or "").upper(),
                "now_cost": int(float(fixture_row.get("price_units") or 0)),
                "price": nullable_float(fixture_row.get("price")),
                "fixture_id": nullable_int(fixture_row.get("fixture_id")),
                "fixture_count": int(
                    float(fixture_row.get("fixture_count_for_team_gw") or 0)
                ),
                "has_fixture": has_fixture,
                "blank_gw_flag": bool_value(fixture_row.get("blank_gw_flag")),
                "double_gw_flag": bool_value(fixture_row.get("double_gw_flag")),
                "opponent_team_id": nullable_int(
                    fixture_row.get("opponent_team_id")
                ),
                "opponent_team_name": str(
                    fixture_row.get("opponent_team_name") or ""
                ),
                "opponent_team_short_name": str(
                    fixture_row.get("opponent_team_short_name") or ""
                ),
                "is_home": (
                    None
                    if fixture_row.get("is_home") is None
                    or pd.isna(fixture_row.get("is_home"))
                    else bool_value(fixture_row.get("is_home"))
                ),
                "kickoff_time_utc": (
                    None
                    if pd.isna(fixture_row.get("kickoff_time_utc"))
                    else str(fixture_row.get("kickoff_time_utc"))
                ),
                "kickoff_time_status": str(
                    fixture_row.get("kickoff_time_status") or ""
                ),
                "kickoff_change_status": str(
                    fixture_row.get("kickoff_change_status") or ""
                ),
                "selection_eligible": current_eligible and fixture_eligible,
                "fixture_eligibility_reason": str(
                    fixture_row.get("eligibility_reason") or ""
                ),
                "prediction_available": source_is_available,
                "prediction_status": prediction_status,
                "row_status": row_status,
                "predicted_points": predicted_points,
                "discounted_predicted_points": discounted_points,
                "expected_minutes": expected_minutes,
                "appearance_probability": appearance_probability,
                "start_probability": start_probability,
                "uncertainty_lower": uncertainty_lower,
                "uncertainty_upper": uncertainty_upper,
                "confidence_score": nullable_float(
                    source.get("confidence_score") if source else None
                ),
                "confidence_label": (
                    str(source.get("confidence_label") or "")
                    if source
                    else "missing"
                ),
                "fallback_used": fallback_used,
                "fallback_level": fallback_level,
                "fallback_reason": (
                    "future_gameweek_prediction_not_generated"
                    if source is None and gameweek > start_gw
                    else (
                        "required_gw1_prediction_missing"
                        if source is None
                        else ""
                    )
                ),
                "risk_flags": list(risk_flags),
                "readiness_status": readiness_status,
                "manual_review_required": manual_review,
                "source_prediction_run_id": source_prediction_run_id,
                "source_fixture_horizon_run_id": str(
                    fixture_row.get("run_id") or ""
                ),
                "prediction_source": (
                    str(source.get("prediction_source") or "")
                    if source
                    else "missing"
                ),
                "recommendation_status": RECOMMENDATION_STATUS,
                "production_ready": False,
                "production_approved": False,
                "prediction_write_allowed": False,
            }
        )

    long_frame = pd.DataFrame(rows)
    expected_rows = len(predictions) * horizon
    diagnostics = {
        "expected_player_gameweek_rows": expected_rows,
        "actual_player_gameweek_rows": len(long_frame),
        "unique_players": int(long_frame["player_id"].nunique()),
        "available_prediction_row_count": available_prediction_count,
        "missing_prediction_row_count": missing_prediction_count,
        "future_missing_prediction_row_count": future_missing_count,
    }
    return long_frame, diagnostics


def determine_mode(
    long_frame: pd.DataFrame,
    start_gw: int,
    horizon: int,
) -> Dict[str, Any]:
    required_gws = list(range(start_gw, start_gw + horizon))
    available_by_gw = {
        gw: int(
            (
                (long_frame["target_gw"] == gw)
                & long_frame["prediction_available"].apply(bool_value)
            ).sum()
        )
        for gw in required_gws
    }
    players = int(long_frame["player_id"].nunique())
    complete_gws = [
        gw for gw, count in available_by_gw.items() if count == players
    ]
    if complete_gws == required_gws:
        return {
            "objective_mode": OBJECTIVE_MULTI_GW,
            "effective_horizon": horizon,
            "future_fixture_context": "prediction_available",
            "complete_prediction_gameweeks": complete_gws,
            "missing_prediction_gameweeks": [],
        }
    if available_by_gw.get(start_gw) == players:
        return {
            "objective_mode": OBJECTIVE_GW1_ONLY,
            "effective_horizon": 1,
            "future_fixture_context": FUTURE_CONTEXT_MANUAL,
            "complete_prediction_gameweeks": complete_gws,
            "missing_prediction_gameweeks": [
                gw for gw in required_gws if gw not in complete_gws
            ],
        }
    raise PlayerPredictionHorizonError(
        "GW1 predictions are incomplete; the fallback cannot safely proceed."
    )


def build_optimizer_rows(
    long_frame: pd.DataFrame,
    mode: Mapping[str, Any],
) -> pd.DataFrame:
    effective_horizon = int(mode["effective_horizon"])
    start_gw = int(long_frame["start_gw"].iloc[0])
    end_gw = start_gw + effective_horizon - 1
    rows = long_frame[
        (long_frame["target_gw"] >= start_gw)
        & (long_frame["target_gw"] <= end_gw)
        & long_frame["selection_eligible"].apply(bool_value)
        & long_frame["prediction_available"].apply(bool_value)
    ].copy()
    columns = [
        "player_id",
        "target_gw",
        "predicted_points",
        "expected_minutes",
        "start_probability",
        "appearance_probability",
        "has_fixture",
        "fallback_used",
        "fallback_level",
        "uncertainty_lower",
        "uncertainty_upper",
        "now_cost",
        "position",
        "risk_flags",
        "readiness_status",
        "production_ready",
        "selection_eligible",
        "fixture_id",
        "opponent_team_id",
        "is_home",
        "kickoff_time_utc",
        "manual_review_required",
        "source_prediction_run_id",
        "source_fixture_horizon_run_id",
    ]
    return rows[columns].sort_values(["player_id", "target_gw"]).reset_index(drop=True)


def build_player_summary(
    long_frame: pd.DataFrame,
    mode: Mapping[str, Any],
) -> pd.DataFrame:
    summaries: List[Dict[str, Any]] = []
    for player_id, group in long_frame.groupby("player_id", sort=True):
        ordered = group.sort_values("target_gw")
        available = ordered[ordered["prediction_available"].apply(bool_value)]
        fixture_gws = ordered[ordered["has_fixture"].apply(bool_value)]
        missing_gws = ordered.loc[
            ~ordered["prediction_available"].apply(bool_value),
            "target_gw",
        ].astype(int).tolist()
        summaries.append(
            {
                "artifact_type": "player_prediction_horizon_summary",
                "horizon_version": HORIZON_VERSION,
                "target_season": ordered["target_season"].iloc[0],
                "start_gw": int(ordered["start_gw"].iloc[0]),
                "horizon": int(ordered["horizon"].iloc[0]),
                "effective_horizon": int(mode["effective_horizon"]),
                "objective_mode": str(mode["objective_mode"]),
                "player_id": int(player_id),
                "player_name": ordered["player_name"].iloc[0],
                "web_name": ordered["web_name"].iloc[0],
                "team_id": int(ordered["team_id"].iloc[0]),
                "team_name": ordered["team_name"].iloc[0],
                "team_short_name": ordered["team_short_name"].iloc[0],
                "position": ordered["position"].iloc[0],
                "now_cost": int(ordered["now_cost"].iloc[0]),
                "selection_eligible": bool(
                    ordered["selection_eligible"].apply(bool_value).any()
                ),
                "fixture_gameweek_count": int(len(fixture_gws)),
                "prediction_gameweek_count": int(len(available)),
                "missing_prediction_gameweek_count": int(len(missing_gws)),
                "missing_prediction_gameweeks": ",".join(
                    str(value) for value in missing_gws
                ),
                "available_undiscounted_points": float(
                    available["predicted_points"].fillna(0.0).sum()
                ),
                "available_discounted_points": float(
                    available["discounted_predicted_points"].fillna(0.0).sum()
                ),
                "full_horizon_prediction_complete": len(available)
                == int(ordered["horizon"].iloc[0]),
                "future_fixture_context": str(mode["future_fixture_context"]),
                "manual_review_required": bool(
                    ordered["manual_review_required"].apply(bool_value).any()
                ),
                "fallback_used": bool(
                    ordered["fallback_used"].apply(bool_value).any()
                ),
                "max_fallback_level": int(
                    pd.to_numeric(
                        ordered["fallback_level"], errors="coerce"
                    ).fillna(0).max()
                ),
                "recommendation_status": RECOMMENDATION_STATUS,
                "production_approved": False,
            }
        )
    return pd.DataFrame(summaries)


def build_validation(
    long_frame: pd.DataFrame,
    optimizer_rows: pd.DataFrame,
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    start_gw: int,
    horizon: int,
    mode: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = len(predictions) * horizon
    blockers: List[str] = []
    if len(long_frame) != expected:
        blockers.append("Long-format row count does not equal players × horizon.")
    if long_frame.duplicated(["player_id", "target_gw"]).any():
        blockers.append("Duplicate player-Gameweek rows exist.")
    gw1_rows = long_frame[long_frame["target_gw"] == start_gw]
    if len(gw1_rows) != len(predictions):
        blockers.append("GW1 does not contain one row per source player.")
    if not gw1_rows["prediction_available"].apply(bool_value).all():
        blockers.append("GW1 contains missing predictions.")
    if optimizer_rows.empty:
        blockers.append("Optimizer projection rows are empty.")
    if optimizer_rows["predicted_points"].isna().any():
        blockers.append("Optimizer rows contain missing predicted points.")
    if any(bool_value(value) for value in optimizer_rows["production_ready"]):
        blockers.append("Optimizer rows cannot be production ready.")

    warnings: List[str] = []
    missing_future = int(
        (
            (long_frame["target_gw"] > start_gw)
            & ~long_frame["prediction_available"].apply(bool_value)
        ).sum()
    )
    if missing_future:
        warnings.append(
            "%s future player-Gameweek predictions are missing and remain explicit; "
            "GW1-only fallback is active." % missing_future
        )
    return {
        "passed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "expected_long_row_count": expected,
        "long_row_count": len(long_frame),
        "summary_row_count": len(summary),
        "optimizer_row_count": len(optimizer_rows),
        "unique_player_count": int(long_frame["player_id"].nunique()),
        "duplicate_player_gameweek_count": int(
            long_frame.duplicated(["player_id", "target_gw"]).sum()
        ),
        "prediction_rows_by_gw": {
            str(gw): int(
                (
                    (long_frame["target_gw"] == gw)
                    & long_frame["prediction_available"].apply(bool_value)
                ).sum()
            )
            for gw in range(start_gw, start_gw + horizon)
        },
        "missing_prediction_rows_by_gw": {
            str(gw): int(
                (
                    (long_frame["target_gw"] == gw)
                    & ~long_frame["prediction_available"].apply(bool_value)
                ).sum()
            )
            for gw in range(start_gw, start_gw + horizon)
        },
        "objective_mode": mode["objective_mode"],
        "effective_horizon": mode["effective_horizon"],
        "future_fixture_context": mode["future_fixture_context"],
    }


def build_run_metadata_payload(
    *,
    target_season: str,
    start_gw: int,
    horizon: int,
    as_of_time: str,
    created_at: str,
    prediction_report: Mapping[str, Any],
    fixture_report: Mapping[str, Any],
    inputs: Mapping[str, Dict[str, Any]],
    objective_mode: str,
) -> Dict[str, Any]:
    from ml.contracts.run_metadata import (
        build_run_metadata,
        provenance_inputs_from_file_metadata,
    )

    parent_ids = [
        str((prediction_report.get("run_metadata") or {}).get("run_id") or ""),
        str((fixture_report.get("run_metadata") or {}).get("run_id") or ""),
    ]
    parent_ids = [value for value in parent_ids if value]
    rules = (
        ((fixture_report.get("run_metadata") or {}).get("versions") or {}).get(
            "rules_versions"
        )
        or ((prediction_report.get("run_metadata") or {}).get("versions") or {}).get(
            "rules_versions"
        )
        or {"scoring": "fpl_2026_27_scoring_v1"}
    )
    return build_run_metadata(
        run_id=None,
        run_type="prediction",
        artifact_type=ARTIFACT_TYPE,
        source_seasons=["2025_26"],
        target_season=target_season,
        target_gw=start_gw,
        horizon=horizon,
        as_of_time=as_of_time,
        prediction_mode="pre_gw1_prior",
        created_at=created_at,
        feature_version=HORIZON_VERSION,
        model_version="pre_gw1_heuristic_preview",
        rules_versions=rules,
        manifest_version=HORIZON_VERSION,
        artifact_version=HORIZON_SCHEMA_VERSION,
        additional_versions={
            "source_prediction_refresh_version": str(
                prediction_report.get("refresh_version") or "day76d_v1"
            ),
            "source_fixture_horizon_version": str(
                fixture_report.get("horizon_version") or "day79b_v1"
            ),
            "objective_mode": objective_mode,
        },
        provenance={
            "producer": "ml.features.build_player_prediction_horizon",
            "inputs": provenance_inputs_from_file_metadata(inputs),
            "parent_run_ids": parent_ids,
            "notes": [
                "Artifact-first Day97A preview horizon.",
                "Missing future predictions remain explicit and are never trusted zeroes.",
                "The optimizer source falls back to GW1-only when GW2-GW5 predictions are unavailable.",
                "No database, prediction-table, recommendation, or squad-state writes are performed.",
            ],
        },
    ).to_dict()


def build_report(
    *,
    run_metadata: Mapping[str, Any],
    target_season: str,
    start_gw: int,
    horizon: int,
    as_of_time: str,
    mode: Mapping[str, Any],
    validation: Mapping[str, Any],
    inputs: Mapping[str, Dict[str, Any]],
    prediction_report: Mapping[str, Any],
    fixture_report: Mapping[str, Any],
) -> Dict[str, Any]:
    passed = bool(validation["passed"])
    stop = passed and int(mode["effective_horizon"]) >= 1
    return {
        "created_at_utc": utc_now(),
        "artifact_type": ARTIFACT_TYPE,
        "horizon_version": HORIZON_VERSION,
        "horizon_schema_version": HORIZON_SCHEMA_VERSION,
        "target_season": target_season,
        "start_gw": start_gw,
        "end_gw": start_gw + horizon - 1,
        "requested_horizon": horizon,
        "effective_horizon": int(mode["effective_horizon"]),
        "objective_mode": str(mode["objective_mode"]),
        "future_fixture_context": str(mode["future_fixture_context"]),
        "as_of_time_utc": as_of_time,
        "prediction_source": "pre_gw1_heuristic_preview",
        "recommendation_status": RECOMMENDATION_STATUS,
        "preview_only": True,
        "audit_only": True,
        "production_approved": False,
        "historical_multi_season_backtest_complete": False,
        "component_model_stack_complete": False,
        "writes_database": False,
        "writes_predictions_table": False,
        "writes_recommendations": False,
        "writes_squad_state": False,
        "run_metadata": dict(run_metadata),
        "source_artifacts": dict(inputs),
        "source_status": {
            "day76d_stop_point_satisfied": prediction_report.get(
                "stop_point_satisfied"
            ),
            "day79b_stop_point_satisfied": fixture_report.get(
                "stop_point_satisfied"
            ),
            "day79b_ready_for_day97a": fixture_report.get("ready_for_day97a"),
        },
        "mode_resolution": dict(mode),
        "validation": dict(validation),
        "passed": passed,
        "ready_for_day100b_objective": stop,
        "ready_for_day101a": stop,
        "stop_point_satisfied": stop,
        "blockers": list(validation["blockers"]),
        "warnings": list(validation["warnings"]),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "# Day97A — Multi-GW Prediction Horizon Artifact",
        "",
        "- Run ID: `%s`" % report["run_metadata"]["run_id"],
        "- Target season: `%s`" % report["target_season"],
        "- Requested horizon: `GW%s-GW%s`"
        % (report["start_gw"], report["end_gw"]),
        "- Effective horizon: `%s`" % report["effective_horizon"],
        "- Objective mode: `%s`" % report["objective_mode"],
        "- Future fixture context: `%s`" % report["future_fixture_context"],
        "- Recommendation status: `%s`" % report["recommendation_status"],
        "",
        "## Row counts",
        "",
        "```text",
        "long rows: %s" % validation["long_row_count"],
        "player summaries: %s" % validation["summary_row_count"],
        "optimizer rows: %s" % validation["optimizer_row_count"],
        "unique players: %s" % validation["unique_player_count"],
        "```",
        "",
        "## Prediction availability",
        "",
        "```text",
    ]
    for gw, count in validation["prediction_rows_by_gw"].items():
        lines.append("GW%s available predictions: %s" % (gw, count))
    for gw, count in validation["missing_prediction_rows_by_gw"].items():
        lines.append("GW%s missing predictions: %s" % (gw, count))
    lines.extend(
        [
            "```",
            "",
            "## Safety",
            "",
            "```text",
            "preview_only: True",
            "production_approved: False",
            "writes_database: False",
            "writes_predictions_table: False",
            "writes_recommendations: False",
            "writes_squad_state: False",
            "```",
            "",
            "Missing predictions are represented as missing values with explicit "
            "`prediction_status`, `fallback_reason`, and `manual_review_required` fields. "
            "They are never converted into trusted zero-point predictions.",
            "",
            "## Warnings",
            "",
        ]
    )
    if report["warnings"]:
        lines.extend("- %s" % item for item in report["warnings"])
    else:
        lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend("- %s" % item for item in report["blockers"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Stop point",
            "",
            "> The optimizer has one validated player-GW source, with either "
            "GW1-GW5 preview points or an explicit GW1-only fallback.",
            "",
            "Stop point satisfied: `%s`" % report["stop_point_satisfied"],
            "",
        ]
    )
    return "\n".join(lines)


def artifact_definitions() -> Dict[str, Tuple[str, str]]:
    return {
        "player_prediction_horizon_csv": ("player_prediction_horizon", "csv"),
        "player_horizon_summary_csv": ("player_horizon_summary", "csv"),
        "optimizer_projection_rows_csv": ("optimizer_projection_rows", "csv"),
        "run_metadata_json": ("run_metadata", "json"),
        "player_prediction_horizon_report_json": (
            "player_prediction_horizon_report",
            "json",
        ),
        "player_prediction_horizon_report_md": (
            "player_prediction_horizon_report",
            "md",
        ),
    }


def write_immutable_outputs(
    artifact_root: Path,
    target_season: str,
    start_gw: int,
    as_of_time: str,
    run_id: str,
    long_frame: pd.DataFrame,
    summary: pd.DataFrame,
    optimizer_rows: pd.DataFrame,
    run_metadata: Mapping[str, Any],
    report: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    from ml.artifacts.paths import build_immutable_artifact_key
    from ml.artifacts.storage import LocalArtifactStorage

    storage = LocalArtifactStorage(artifact_root)
    keys = {
        name: build_immutable_artifact_key(
            artifact_type=ARTIFACT_TYPE,
            season=target_season,
            target_gw=start_gw,
            as_of_time=as_of_time,
            run_id=run_id,
            version=HORIZON_VERSION,
            filename=filename,
            extension=extension,
        )
        for name, (filename, extension) in artifact_definitions().items()
    }
    payloads = {
        "player_prediction_horizon_csv": long_frame.to_csv(index=False),
        "player_horizon_summary_csv": summary.to_csv(index=False),
        "optimizer_projection_rows_csv": optimizer_rows.to_csv(index=False),
        "run_metadata_json": json.dumps(
            run_metadata, indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
    }
    stored: Dict[str, Dict[str, Any]] = {}
    for name, content in payloads.items():
        stored[name] = storage.write_immutable_text(keys[name], content).to_dict()

    report["artifacts"] = {
        "root": str(Path(artifact_root).expanduser().resolve()),
        "keys": keys,
        "stored_before_report": dict(stored),
    }
    report_payloads = {
        "player_prediction_horizon_report_json": json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        "player_prediction_horizon_report_md": build_markdown_report(report),
    }
    for name, content in report_payloads.items():
        stored[name] = storage.write_immutable_text(keys[name], content).to_dict()
    return stored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Day97A preview player-prediction horizon from Day76D "
            "standard predictions and the Day79B player-fixture horizon."
        )
    )
    parser.add_argument(
        "--prediction-refresh-report-json",
        required=True,
        help="Day76D refresh_report.json path.",
    )
    parser.add_argument(
        "--fixture-horizon-report-json",
        required=True,
        help="Day79B fixture_horizon_report.json path.",
    )
    parser.add_argument("--target-season", default=DEFAULT_TARGET_SEASON)
    parser.add_argument("--start-gw", type=int, default=DEFAULT_START_GW)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--artifact-root",
        default="/private/tmp/fpl-artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_season, start_gw, horizon, end_gw = normalize_scope(
        args.target_season,
        args.start_gw,
        args.horizon,
    )

    prediction_report_path = Path(
        args.prediction_refresh_report_json
    ).expanduser().resolve()
    fixture_report_path = Path(
        args.fixture_horizon_report_json
    ).expanduser().resolve()
    prediction_report = read_json(prediction_report_path)
    fixture_report = read_json(fixture_report_path)
    validate_source_reports(
        prediction_report,
        fixture_report,
        target_season,
        start_gw,
        horizon,
    )

    prediction_csv = (
        prediction_report_path.parent / "standard_player_predictions_csv.csv"
    )
    fixture_context_csv = (
        fixture_report_path.parent / "player_fixture_eligibility.csv"
    )
    if not prediction_csv.is_file():
        raise PlayerPredictionHorizonError(
            "Missing Day76D standard predictions: %s" % prediction_csv
        )
    if not fixture_context_csv.is_file():
        raise PlayerPredictionHorizonError(
            "Missing Day79B player fixture context: %s" % fixture_context_csv
        )

    predictions = load_prediction_rows(prediction_csv, target_season)
    fixture_context = load_fixture_context(
        fixture_context_csv,
        target_season,
        start_gw,
        end_gw,
    )
    if int(fixture_context["player_id"].nunique()) != len(predictions):
        raise PlayerPredictionHorizonError(
            "Prediction and fixture-context player counts do not match."
        )

    discounts = discount_schedule(start_gw, horizon)
    long_frame, _ = build_long_horizon(
        predictions,
        fixture_context,
        start_gw,
        horizon,
        discounts,
    )
    mode = determine_mode(long_frame, start_gw, horizon)
    optimizer_rows = build_optimizer_rows(long_frame, mode)
    summary = build_player_summary(long_frame, mode)
    validation = build_validation(
        long_frame,
        optimizer_rows,
        summary,
        predictions,
        start_gw,
        horizon,
        mode,
    )

    inputs = {
        "prediction_refresh_report": file_metadata(prediction_report_path),
        "standard_player_predictions": file_metadata(prediction_csv),
        "fixture_horizon_report": file_metadata(fixture_report_path),
        "player_fixture_eligibility": file_metadata(fixture_context_csv),
    }
    as_of_time = min(
        str(prediction_report.get("as_of_time_utc") or ""),
        str(fixture_report.get("as_of_time_utc") or ""),
    )
    if not as_of_time:
        raise PlayerPredictionHorizonError("Source as-of time is missing.")
    created_at = utc_now()
    run_metadata = build_run_metadata_payload(
        target_season=target_season,
        start_gw=start_gw,
        horizon=horizon,
        as_of_time=as_of_time,
        created_at=created_at,
        prediction_report=prediction_report,
        fixture_report=fixture_report,
        inputs=inputs,
        objective_mode=str(mode["objective_mode"]),
    )
    report = build_report(
        run_metadata=run_metadata,
        target_season=target_season,
        start_gw=start_gw,
        horizon=horizon,
        as_of_time=as_of_time,
        mode=mode,
        validation=validation,
        inputs=inputs,
        prediction_report=prediction_report,
        fixture_report=fixture_report,
    )
    if not report["passed"]:
        raise PlayerPredictionHorizonError(
            "Day97A validation failed: %s" % report["blockers"]
        )

    stored = write_immutable_outputs(
        artifact_root=Path(args.artifact_root),
        target_season=target_season,
        start_gw=start_gw,
        as_of_time=as_of_time,
        run_id=str(run_metadata["run_id"]),
        long_frame=long_frame,
        summary=summary,
        optimizer_rows=optimizer_rows,
        run_metadata=run_metadata,
        report=report,
    )
    print("Day97A player prediction horizon complete.")
    print("run_id:", run_metadata["run_id"])
    print("target_season:", target_season)
    print("start_gw:", start_gw)
    print("end_gw:", end_gw)
    print("requested_horizon:", horizon)
    print("effective_horizon:", mode["effective_horizon"])
    print("objective_mode:", mode["objective_mode"])
    print("future_fixture_context:", mode["future_fixture_context"])
    print("long_rows:", len(long_frame))
    print("summary_rows:", len(summary))
    print("optimizer_rows:", len(optimizer_rows))
    print("immutable_artifacts:", len(stored))
    print("preview_only: true")
    print("writes_database: false")
    print("production_approved: false")
    print("stop_point_satisfied:", str(report["stop_point_satisfied"]).lower())


if __name__ == "__main__":
    main()
