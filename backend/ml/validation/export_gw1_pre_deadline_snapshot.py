from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from app.rules.squad import load_squad_transfer_rules
from ml.artifacts.paths import build_immutable_artifact_key
from ml.artifacts.storage import LocalArtifactStorage
from ml.contracts.run_metadata import (
    build_run_metadata,
    provenance_inputs_from_file_metadata,
)
from ml.decision.optimize_opening_squad import file_metadata, sha256_file
from ml.decision.squad_rules import SquadLegalityEngine


SNAPSHOT_VERSION = "day101c_v1"
ARTIFACT_TYPE = "gw1_pre_deadline_snapshot"
ARTIFACT_VERSION = "fpl_gw1_pre_deadline_snapshot_v1"
RECOMMENDATION_STATUS = "preview_only"

SNAPSHOT_KIND_CANDIDATE = "pre_deadline_candidate"
SNAPSHOT_KIND_FINAL = "final_pre_deadline"

VARIANTS = ("primary", "alternative_a", "alternative_b")
POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}

GLOBAL_REQUIRED_COLUMNS = {
    "target_season",
    "target_gw",
    "player_id",
    "player_name",
    "web_name",
    "team_id",
    "team_name",
    "team_short_name",
    "position",
    "now_cost",
    "selection_eligible",
    "prediction_available",
    "prediction_status",
    "predicted_points",
    "expected_minutes",
    "appearance_probability",
    "start_probability",
    "fallback_used",
    "fallback_level",
    "risk_flags",
    "manual_review_required",
    "source_prediction_run_id",
    "source_fixture_horizon_run_id",
    "recommendation_status",
    "production_ready",
    "production_approved",
    "prediction_write_allowed",
}

FINAL_REFRESH_REQUIRED_COLUMNS = {
    "player_id",
    "player_name",
    "web_name",
    "team_id",
    "team_name",
    "team_short_name",
    "position",
    "now_cost",
    "status",
    "chance_of_playing_next_round",
    "news",
    "news_added",
    "official_availability_probability",
    "official_availability_workload_factor",
    "official_availability_adjustment_applied",
    "appearance_probability",
    "start_probability",
    "expected_minutes",
    "predicted_points",
    "prediction_confidence",
    "fallback_policy_used",
    "fallback_level",
    "fallback_reason",
    "risk_flags",
    "status_cutoff_valid",
    "status_hard_guardrail_applied",
    "prediction_write_allowed",
    "production_ready",
}


class GW1SnapshotError(RuntimeError):
    """Raised when a Day101C local GW1 deliverable cannot be trusted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any, label: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise GW1SnapshotError("%s is required." % label)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise GW1SnapshotError("%s must be valid ISO-8601." % label) from exc
    if parsed.tzinfo is None:
        raise GW1SnapshotError("%s must include a timezone offset." % label)
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def nullable_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def nullable_int(value: Any) -> Optional[int]:
    number = nullable_float(value)
    return None if number is None else int(number)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def list_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if parsed is None:
        return []
    return [str(parsed).strip()] if str(parsed).strip() else []


def same_float(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    lvalue = nullable_float(left)
    rvalue = nullable_float(right)
    if lvalue is None or rvalue is None:
        return lvalue is None and rvalue is None
    return abs(lvalue - rvalue) <= tolerance


def load_json(path: Path, label: str) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise GW1SnapshotError("%s does not exist: %s" % (label, resolved))
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GW1SnapshotError("Invalid JSON in %s: %s" % (resolved, exc))
    if not isinstance(payload, dict):
        raise GW1SnapshotError("%s must contain a JSON object." % label)
    return payload


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise GW1SnapshotError("%s missing columns: %s" % (label, missing))


def resolve_report_artifact(
    report: Mapping[str, Any],
    key_name: str,
    label: str,
) -> Path:
    artifacts = report.get("artifacts") or {}
    root = artifacts.get("root")
    keys = artifacts.get("keys") or {}
    key = keys.get(key_name)
    if not root or not key:
        raise GW1SnapshotError("%s artifact metadata is missing %s." % (label, key_name))
    path = Path(str(root)).expanduser().resolve() / Path(str(key))
    if not path.is_file():
        raise GW1SnapshotError("%s artifact does not exist: %s" % (label, path))
    return path


def resolve_source_metadata_path(metadata: Mapping[str, Any], label: str) -> Path:
    raw_path = metadata.get("path")
    if not raw_path:
        raise GW1SnapshotError("%s source path is missing." % label)
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.is_file():
        raise GW1SnapshotError("%s source does not exist: %s" % (label, path))
    expected_sha = str(metadata.get("sha256") or "")
    if expected_sha and sha256_file(path) != expected_sha:
        raise GW1SnapshotError("%s source fingerprint changed." % label)
    return path


def validate_day101b_report(report: Mapping[str, Any]) -> None:
    blockers: List[str] = []
    checks = [
        (
            report.get("artifact_type") == "opening_lineup_optimizer",
            "Day101B artifact_type must be opening_lineup_optimizer.",
        ),
        (report.get("passed") is True, "Day101B report must have passed=true."),
        (
            report.get("ready_for_day101c") is True,
            "Day101B report must have ready_for_day101c=true.",
        ),
        (
            report.get("stop_point_satisfied") is True,
            "Day101B report must have stop_point_satisfied=true.",
        ),
        (
            report.get("preview_only") is True,
            "Day101B report must remain preview_only=true.",
        ),
        (
            report.get("production_approved") is False,
            "Day101B report must have production_approved=false.",
        ),
        (
            report.get("writes_database") is False,
            "Day101B report must have writes_database=false.",
        ),
        (
            report.get("writes_predictions_table") is False,
            "Day101B report must have writes_predictions_table=false.",
        ),
        (
            report.get("writes_recommendations") is False,
            "Day101B report must have writes_recommendations=false.",
        ),
        (
            report.get("writes_squad_state") is False,
            "Day101B report must have writes_squad_state=false.",
        ),
        (not report.get("blockers"), "Day101B report must not contain blockers."),
        (
            int(report.get("target_gw") or 0) == 1,
            "Day101C requires target_gw=1.",
        ),
    ]
    for valid, message in checks:
        if not valid:
            blockers.append(message)

    primary = (report.get("plans") or {}).get("primary") or {}
    legality = primary.get("legality") or {}
    if legality.get("valid") is not True:
        blockers.append("Day101B Primary legality must be valid.")
    reconciliation = primary.get("objective_reconciliation") or {}
    if reconciliation.get("passed") is not True:
        blockers.append("Day101B Primary objective reconciliation must have passed.")

    availability = report.get("availability_scope") or {}
    if availability.get("official_chance_of_playing_next_round_propagated") is not True:
        blockers.append(
            "Day101B must record official_chance_of_playing_next_round_propagated=true."
        )
    if availability.get(
        "official_availability_consumed_via_adjusted_projection_inputs"
    ) is not True:
        blockers.append(
            "Day101B must record official availability consumption via adjusted inputs."
        )
    if availability.get(
        "day101b_additional_official_availability_penalty_applied"
    ) is not False:
        blockers.append("Day101B must not apply a second availability penalty.")
    if availability.get("required_follow_up") not in (None, ""):
        blockers.append("Day101B availability required_follow_up must be empty.")

    if blockers:
        raise GW1SnapshotError("Unsafe Day101B report: %s" % " | ".join(blockers))


def load_day101a_source(
    day101b_report: Mapping[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    metadata = (day101b_report.get("source_artifacts") or {}).get("day101a_report") or {}
    path = resolve_source_metadata_path(metadata, "Day101A report")
    report = load_json(path, "Day101A report")

    expected_run = str(day101b_report.get("source_day101a_run_id") or "")
    actual_run = str((report.get("run_metadata") or {}).get("run_id") or "")
    if expected_run and expected_run != actual_run:
        raise GW1SnapshotError(
            "Day101A run mismatch: Day101B expects %s, loaded %s."
            % (expected_run, actual_run)
        )

    checks = [
        (
            report.get("artifact_type") == "opening_squad_optimizer",
            "Day101A artifact_type must be opening_squad_optimizer.",
        ),
        (report.get("passed") is True, "Day101A report must have passed=true."),
        (
            report.get("ready_for_day101b") is True,
            "Day101A report must have ready_for_day101b=true.",
        ),
        (
            report.get("preview_only") is True,
            "Day101A report must remain preview_only=true.",
        ),
        (
            report.get("production_approved") is False,
            "Day101A report must have production_approved=false.",
        ),
        (
            report.get("writes_database") is False,
            "Day101A report must have writes_database=false.",
        ),
        (
            report.get("writes_squad_state") is False,
            "Day101A report must have writes_squad_state=false.",
        ),
    ]
    blockers = [message for valid, message in checks if not valid]
    for variant in VARIANTS:
        summary = (report.get("variants") or {}).get(variant) or {}
        if (summary.get("squad_legality") or {}).get("valid") is not True:
            blockers.append("Day101A %s squad legality must be valid." % variant)
        if (summary.get("objective_reconciliation") or {}).get("passed") is not True:
            blockers.append("Day101A %s objective reconciliation must pass." % variant)
    if blockers:
        raise GW1SnapshotError("Unsafe Day101A report: %s" % " | ".join(blockers))
    return path, report


def load_day97a_source(
    day101a_report: Mapping[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    metadata = (day101a_report.get("source_artifacts") or {}).get("day97a_report") or {}
    path = resolve_source_metadata_path(metadata, "Day97A report")
    report = load_json(path, "Day97A report")
    if report.get("passed") is not True or report.get("stop_point_satisfied") is not True:
        raise GW1SnapshotError("Day97A report is not a passing source.")
    if report.get("preview_only") is not True or report.get("production_approved") is not False:
        raise GW1SnapshotError("Day97A source safety flags are invalid.")
    return path, report


def load_day76d_source(
    day97a_report: Mapping[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    metadata = (day97a_report.get("source_artifacts") or {}).get(
        "prediction_refresh_report"
    ) or {}
    path = resolve_source_metadata_path(metadata, "Day76D prediction refresh report")
    report = load_json(path, "Day76D prediction refresh report")
    if report.get("passed") is not True or report.get("stop_point_satisfied") is not True:
        raise GW1SnapshotError("Day76D refresh report is not a passing source.")
    if report.get("writes_database") is not False:
        raise GW1SnapshotError("Day76D refresh must remain read-only.")
    return path, report


def normalize_all_squads(
    frame: pd.DataFrame,
    day101a_report: Mapping[str, Any],
) -> pd.DataFrame:
    required = {
        "variant",
        "variant_label",
        "player_id",
        "player_name",
        "web_name",
        "team_id",
        "team_name",
        "team_short_name",
        "position",
        "now_cost",
        "gw1_predicted_points",
        "expected_minutes",
        "start_probability",
        "appearance_probability",
        "fallback_used",
        "fallback_level",
        "risk_flags",
        "manual_review_required",
        "variant_total_cost_units",
        "variant_bank_units",
        "variant_objective_value",
    }
    require_columns(frame, required, "Day101A opening_squad.csv")

    normalized = frame.copy()
    normalized["player_id"] = normalized["player_id"].astype(int)
    normalized["team_id"] = normalized["team_id"].astype(int)
    normalized["now_cost"] = normalized["now_cost"].astype(int)
    for variant in VARIANTS:
        subset = normalized[normalized["variant"].astype(str) == variant]
        if len(subset) != 15:
            raise GW1SnapshotError("Day101A %s must contain exactly 15 rows." % variant)
        if subset["player_id"].duplicated().any():
            raise GW1SnapshotError("Day101A %s contains duplicate player IDs." % variant)
        expected = set(
            int(v)
            for v in (
                ((day101a_report.get("variants") or {}).get(variant) or {}).get(
                    "selected_player_ids"
                )
                or []
            )
        )
        actual = set(subset["player_id"].astype(int))
        if expected != actual:
            raise GW1SnapshotError("Day101A %s CSV does not match report IDs." % variant)
    return normalized


def normalize_primary_lineup(
    frame: pd.DataFrame,
    day101b_report: Mapping[str, Any],
) -> pd.DataFrame:
    required = {
        "plan",
        "player_id",
        "player_name",
        "web_name",
        "team_short_name",
        "position",
        "now_cost",
        "role",
        "is_starter",
        "is_captain",
        "is_vice_captain",
        "formation",
        "predicted_points",
        "expected_minutes",
        "start_probability",
        "appearance_probability",
        "fallback_used",
        "fallback_level",
        "risk_flags",
        "manual_review_required",
    }
    require_columns(frame, required, "Day101B opening_lineup.csv")
    primary = frame[frame["plan"].astype(str) == "primary"].copy()
    if len(primary) != 15:
        raise GW1SnapshotError("Day101B Primary lineup must contain exactly 15 rows.")
    primary["player_id"] = primary["player_id"].astype(int)
    if primary["player_id"].duplicated().any():
        raise GW1SnapshotError("Day101B Primary lineup contains duplicate player IDs.")

    plan = ((day101b_report.get("plans") or {}).get("primary") or {})
    expected_ids = set(
        int(v)
        for v in list(plan.get("starting_player_ids") or [])
        + list(plan.get("bench_order") or [])
    )
    actual_ids = set(primary["player_id"])
    if expected_ids != actual_ids or len(expected_ids) != 15:
        raise GW1SnapshotError("Day101B Primary lineup CSV does not match report IDs.")
    return primary


def load_final_refresh_frames(
    day76d_report_path: Path,
) -> Tuple[Path, pd.DataFrame, Path, pd.DataFrame]:
    run_dir = day76d_report_path.resolve().parent
    preview_path = run_dir / "prediction_preview_csv.csv"
    standard_path = run_dir / "standard_player_predictions_csv.csv"
    if not preview_path.is_file():
        raise GW1SnapshotError("Missing Day76D prediction preview: %s" % preview_path)
    if not standard_path.is_file():
        raise GW1SnapshotError("Missing Day76D standard predictions: %s" % standard_path)
    preview = pd.read_csv(preview_path, low_memory=False)
    standard = pd.read_csv(standard_path, low_memory=False)
    require_columns(preview, FINAL_REFRESH_REQUIRED_COLUMNS, "Day76D prediction preview")
    require_columns(
        standard,
        {
            "player_id",
            "position",
            "now_cost",
            "selection_eligible",
            "appearance_probability",
            "start_probability",
            "expected_minutes",
            "predicted_points",
            "risk_flags",
            "prediction_write_allowed",
            "production_ready",
        },
        "Day76D standard predictions",
    )
    return preview_path, preview, standard_path, standard


def build_horizon_projection_map(
    long_frame: pd.DataFrame,
    player_ids: Sequence[int],
    requested_horizon: int,
) -> Dict[int, Dict[str, Any]]:
    require_columns(
        long_frame,
        {
            "player_id",
            "target_gw",
            "prediction_available",
            "predicted_points",
            "opponent_team_short_name",
            "is_home",
            "row_status",
        },
        "Day97A player prediction horizon",
    )
    result: Dict[int, Dict[str, Any]] = {}
    wanted = set(int(v) for v in player_ids)
    scoped = long_frame[long_frame["player_id"].astype(int).isin(wanted)].copy()
    for player_id in sorted(wanted):
        group = scoped[scoped["player_id"].astype(int) == player_id]
        if len(group) != requested_horizon:
            raise GW1SnapshotError(
                "Day97A horizon row count mismatch for player_id=%s." % player_id
            )
        by_gw = {int(row["target_gw"]): row for row in group.to_dict(orient="records")}
        payload: Dict[str, Any] = {}
        for gw in range(1, requested_horizon + 1):
            row = by_gw.get(gw)
            if row is None:
                raise GW1SnapshotError(
                    "Day97A is missing GW%s for player_id=%s." % (gw, player_id)
                )
            available = bool_value(row.get("prediction_available"))
            opponent = str(row.get("opponent_team_short_name") or "")
            home_away = ""
            if opponent:
                home_away = "H" if bool_value(row.get("is_home")) else "A"
            payload["gw%s" % gw] = {
                "prediction_available": available,
                "predicted_points": (
                    nullable_float(row.get("predicted_points")) if available else None
                ),
                "opponent": opponent,
                "home_away": home_away,
                "row_status": str(row.get("row_status") or ""),
            }
        result[player_id] = payload
    return result


def build_global_prediction_snapshot(
    long_frame: pd.DataFrame,
    target_season: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    require_columns(long_frame, GLOBAL_REQUIRED_COLUMNS, "Day97A player horizon")
    gw1 = long_frame[
        (long_frame["target_season"].astype(str) == str(target_season))
        & (long_frame["target_gw"].astype(int) == 1)
    ].copy()
    if gw1.empty:
        raise GW1SnapshotError("Global GW1 prediction snapshot is empty.")
    if gw1["player_id"].astype(int).duplicated().any():
        raise GW1SnapshotError("Global GW1 prediction snapshot has duplicate IDs.")
    if not gw1["prediction_available"].apply(bool_value).all():
        raise GW1SnapshotError("Global GW1 snapshot contains unavailable predictions.")
    for column in (
        "predicted_points",
        "expected_minutes",
        "appearance_probability",
        "start_probability",
    ):
        if gw1[column].isna().any():
            raise GW1SnapshotError("Global GW1 snapshot has missing %s." % column)
    for column in (
        "production_ready",
        "production_approved",
        "prediction_write_allowed",
    ):
        if gw1[column].apply(bool_value).any():
            raise GW1SnapshotError("Global GW1 snapshot contains unsafe %s=true." % column)
    audit = {
        "player_count": int(len(gw1)),
        "eligible_player_count": int(gw1["selection_eligible"].apply(bool_value).sum()),
        "ineligible_player_count": int((~gw1["selection_eligible"].apply(bool_value)).sum()),
        "prediction_available_count": int(
            gw1["prediction_available"].apply(bool_value).sum()
        ),
        "duplicate_player_ids": 0,
        "missing_predicted_points": 0,
        "missing_expected_minutes": 0,
        "missing_appearance_probability": 0,
        "missing_start_probability": 0,
        "prediction_write_allowed_true": 0,
        "production_approved_true": 0,
        "production_ready_true": 0,
    }
    return gw1.sort_values("player_id").reset_index(drop=True), audit


def reconcile_primary_with_final_refresh(
    primary_squad: pd.DataFrame,
    lineup: pd.DataFrame,
    preview: pd.DataFrame,
    standard: pd.DataFrame,
    horizon_map: Mapping[int, Mapping[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    preview_by_id = preview.copy()
    preview_by_id["player_id"] = preview_by_id["player_id"].astype(int)
    preview_by_id = preview_by_id.set_index("player_id", drop=False)
    standard_by_id = standard.copy()
    standard_by_id["player_id"] = standard_by_id["player_id"].astype(int)
    standard_by_id = standard_by_id.set_index("player_id", drop=False)
    lineup_by_id = lineup.set_index(lineup["player_id"].astype(int), drop=False)

    mismatches: List[str] = []
    rows: List[Dict[str, Any]] = []
    for squad_row in primary_squad.sort_values("player_id").to_dict(orient="records"):
        player_id = int(squad_row["player_id"])
        if player_id not in lineup_by_id.index:
            mismatches.append("player_id=%s missing from Day101B lineup" % player_id)
            continue
        if player_id not in preview_by_id.index:
            mismatches.append("player_id=%s missing from Day76D preview" % player_id)
            continue
        if player_id not in standard_by_id.index:
            mismatches.append("player_id=%s missing from Day76D standard rows" % player_id)
            continue

        lineup_row = lineup_by_id.loc[player_id]
        refresh_row = preview_by_id.loc[player_id]
        standard_row = standard_by_id.loc[player_id]

        comparisons = [
            (int(squad_row["team_id"]) == int(refresh_row["team_id"]), "team_id"),
            (str(squad_row["team_name"]) == str(refresh_row["team_name"]), "team_name"),
            (str(squad_row["position"]) == str(refresh_row["position"]), "position"),
            (int(squad_row["now_cost"]) == int(refresh_row["now_cost"]), "now_cost"),
            (
                same_float(lineup_row["appearance_probability"], refresh_row["appearance_probability"]),
                "appearance_probability",
            ),
            (
                same_float(lineup_row["start_probability"], refresh_row["start_probability"]),
                "start_probability",
            ),
            (
                same_float(lineup_row["expected_minutes"], refresh_row["expected_minutes"], 1e-4),
                "expected_minutes",
            ),
            (
                same_float(lineup_row["predicted_points"], refresh_row["predicted_points"], 1e-6),
                "predicted_points",
            ),
        ]
        for valid, field in comparisons:
            if not valid:
                mismatches.append("player_id=%s mismatch:%s" % (player_id, field))

        if not bool_value(standard_row.get("selection_eligible")):
            mismatches.append("player_id=%s is not selection_eligible in final refresh" % player_id)
        if bool_value(standard_row.get("prediction_write_allowed")):
            mismatches.append("player_id=%s has unsafe prediction_write_allowed=true" % player_id)
        if bool_value(standard_row.get("production_ready")):
            mismatches.append("player_id=%s has unsafe production_ready=true" % player_id)

        lineup_risks = sorted(set(list_value(lineup_row.get("risk_flags"))))
        refresh_risks = sorted(set(list_value(refresh_row.get("risk_flags"))))
        if lineup_risks != refresh_risks:
            mismatches.append("player_id=%s mismatch:risk_flags" % player_id)

        horizon = horizon_map[player_id]
        row: Dict[str, Any] = {
            "player_id": player_id,
            "fpl_player_id": nullable_int(squad_row.get("fpl_player_id")),
            "player_name": str(squad_row.get("player_name") or ""),
            "web_name": str(squad_row.get("web_name") or ""),
            "team_id": int(squad_row["team_id"]),
            "team_name": str(squad_row.get("team_name") or ""),
            "team_short_name": str(squad_row.get("team_short_name") or ""),
            "position": str(squad_row["position"]),
            "now_cost": int(squad_row["now_cost"]),
            "role": str(lineup_row["role"]),
            "is_starter": bool_value(lineup_row["is_starter"]),
            "is_captain": bool_value(lineup_row["is_captain"]),
            "is_vice_captain": bool_value(lineup_row["is_vice_captain"]),
            "formation": str(lineup_row["formation"]),
            "gw1_predicted_points": float(lineup_row["predicted_points"]),
            "expected_minutes": float(lineup_row["expected_minutes"]),
            "start_probability": float(lineup_row["start_probability"]),
            "appearance_probability": float(lineup_row["appearance_probability"]),
            "status": str(refresh_row.get("status") or ""),
            "chance_of_playing_next_round": nullable_float(
                refresh_row.get("chance_of_playing_next_round")
            ),
            "news": str(refresh_row.get("news") or ""),
            "news_added": str(refresh_row.get("news_added") or ""),
            "official_availability_probability": nullable_float(
                refresh_row.get("official_availability_probability")
            ),
            "official_availability_workload_factor": float(
                refresh_row.get("official_availability_workload_factor") or 1.0
            ),
            "official_availability_adjustment_applied": bool_value(
                refresh_row.get("official_availability_adjustment_applied")
            ),
            "prediction_confidence": str(
                refresh_row.get("prediction_confidence") or ""
            ),
            "fallback_used": bool_value(lineup_row.get("fallback_used")),
            "fallback_policy_used": str(
                refresh_row.get("fallback_policy_used") or ""
            ),
            "fallback_level": int(float(lineup_row.get("fallback_level") or 0)),
            "fallback_reason": str(refresh_row.get("fallback_reason") or ""),
            "risk_flags": json.dumps(lineup_risks, separators=(",", ":")),
            "manual_review_required": bool_value(
                lineup_row.get("manual_review_required")
            ),
            "status_cutoff_valid": bool_value(refresh_row.get("status_cutoff_valid")),
            "status_hard_guardrail_applied": bool_value(
                refresh_row.get("status_hard_guardrail_applied")
            ),
            "selection_eligible_final_refresh": bool_value(
                standard_row.get("selection_eligible")
            ),
        }
        for gw, projection in horizon.items():
            row["%s_predicted_points" % gw] = projection.get("predicted_points")
            row["%s_prediction_available" % gw] = bool(
                projection.get("prediction_available")
            )
            row["%s_opponent" % gw] = str(projection.get("opponent") or "")
            row["%s_home_away" % gw] = str(projection.get("home_away") or "")
        rows.append(row)

    if len(rows) != 15:
        raise GW1SnapshotError("Final-refresh reconciliation did not retain 15 players.")
    export = pd.DataFrame(rows)
    return export, {
        "passed": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "checked_player_count": len(rows),
        "fields_checked": [
            "player_id",
            "team_id",
            "team_name",
            "position",
            "now_cost",
            "selection_eligible",
            "appearance_probability",
            "start_probability",
            "expected_minutes",
            "predicted_points",
            "risk_flags",
            "status/chance/news availability metadata preserved from final refresh",
        ],
    }


def build_model_team_audit(
    primary_export: pd.DataFrame,
    day101a_report: Mapping[str, Any],
    day101b_report: Mapping[str, Any],
    engine: SquadLegalityEngine,
) -> Dict[str, Any]:
    primary_plan = (day101b_report.get("plans") or {}).get("primary") or {}
    primary_squad_summary = (day101a_report.get("variants") or {}).get("primary") or {}
    bank_units = int(primary_squad_summary.get("bank_units") or 0)

    legality_players = [
        {
            "player_id": int(row["player_id"]),
            "player_name": str(row.get("player_name") or row.get("web_name") or ""),
            "position": str(row["position"]),
            "club_id": int(row["team_id"]),
            "price_units": int(row["now_cost"]),
            "selection_eligible": bool_value(row.get("selection_eligible_final_refresh")),
            "eligibility_reason": "final_refresh_reconciled",
        }
        for row in primary_export.to_dict(orient="records")
    ]
    legality = engine.validate_plan(
        legality_players,
        starting_player_ids=primary_plan["starting_player_ids"],
        bench_order=primary_plan["bench_order"],
        captain_player_id=primary_plan["captain_player_id"],
        vice_captain_player_id=primary_plan["vice_captain_player_id"],
        declared_bank_units=bank_units,
    )

    position_counts = {
        position: int((primary_export["position"] == position).sum())
        for position in ("GKP", "DEF", "MID", "FWD")
    }
    expected_position_quotas = {
        position: int(engine.rules.position_quotas[position])
        for position in ("GKP", "DEF", "MID", "FWD")
    }
    position_quotas_pass = position_counts == expected_position_quotas

    club_counts = {
        str(int(team_id)): int(count)
        for team_id, count in primary_export.groupby("team_id").size().items()
    }
    max_per_club = int(engine.rules.squad["max_players_per_club"])
    club_limits_pass = all(count <= max_per_club for count in club_counts.values())

    total_cost_units = int(primary_export["now_cost"].sum())
    report_cost_units = int(primary_squad_summary.get("total_cost_units") or 0)
    budget_units = int(engine.rules.initial_budget_units)
    budget_reconciles = (
        total_cost_units == report_cost_units
        and total_cost_units + bank_units == budget_units
    )

    starters = set(int(v) for v in primary_plan.get("starting_player_ids") or [])
    bench_order = [int(v) for v in primary_plan.get("bench_order") or []]
    captain_id = int(primary_plan["captain_player_id"])
    vice_id = int(primary_plan["vice_captain_player_id"])
    captaincy_pass = (
        captain_id != vice_id and captain_id in starters and vice_id in starters
    )
    duplicates = int(primary_export["player_id"].astype(int).duplicated().sum())

    return {
        "selected_player_count": int(len(primary_export)),
        "duplicate_player_count": duplicates,
        "position_counts": position_counts,
        "expected_position_quotas": expected_position_quotas,
        "position_quotas_pass": position_quotas_pass,
        "club_counts": club_counts,
        "max_players_per_club": max_per_club,
        "club_limits_pass": club_limits_pass,
        "total_cost_units": total_cost_units,
        "bank_units": bank_units,
        "initial_budget_units": budget_units,
        "budget_and_bank_reconcile": budget_reconciles,
        "starting_xi_count": len(starters),
        "formation": str(primary_plan["formation"]),
        "bench_order": bench_order,
        "bench_count": len(bench_order),
        "captain_player_id": captain_id,
        "vice_captain_player_id": vice_id,
        "captain_and_vice_distinct_starting_members": captaincy_pass,
        "legality": legality,
        "objective_value": float(primary_plan["objective_value"]),
        "objective_reconciliation": dict(
            primary_plan.get("objective_reconciliation") or {}
        ),
    }


def build_version_audit(
    day76d_report: Mapping[str, Any],
    day97a_report: Mapping[str, Any],
    day101a_report: Mapping[str, Any],
    day101b_report: Mapping[str, Any],
) -> Dict[str, Any]:
    d76_versions = (day76d_report.get("run_metadata") or {}).get("versions") or {}
    d97_versions = (day97a_report.get("run_metadata") or {}).get("versions") or {}
    d101b_versions = (day101b_report.get("run_metadata") or {}).get("versions") or {}
    objective_policy = (
        ((day101a_report.get("variants") or {}).get("primary") or {}).get(
            "objective_policy"
        )
        or {}
    )
    result = {
        "rules": dict(d101b_versions.get("rules_versions") or {}),
        "feature": {
            "day76d_player_feature_version": str(
                d76_versions.get("feature_version") or ""
            ),
            "day97a_horizon_version": str(
                day97a_report.get("horizon_version") or ""
            ),
        },
        "prediction": {
            "day76d_refresh_version": str(day76d_report.get("refresh_version") or ""),
            "day76d_model_version": str(d76_versions.get("model_version") or ""),
            "day97a_horizon_schema_version": str(
                day97a_report.get("horizon_schema_version") or ""
            ),
            "day97a_objective_mode": str(day97a_report.get("objective_mode") or ""),
        },
        "objective": {
            "contract_version": str(objective_policy.get("contract_version") or ""),
            "policy_version": str(objective_policy.get("policy_version") or ""),
        },
        "artifact": {
            "day101a_optimizer_version": str(
                day101a_report.get("optimizer_version") or ""
            ),
            "day101b_optimizer_version": str(
                day101b_report.get("optimizer_version") or ""
            ),
            "day101c_snapshot_version": SNAPSHOT_VERSION,
            "day101c_artifact_version": ARTIFACT_VERSION,
        },
    }
    required_groups_complete = all(
        bool(group)
        and all(str(value).strip() for value in group.values())
        for name, group in result.items()
        if name in {"feature", "prediction", "objective", "artifact"}
    ) and bool(result["rules"])
    result["all_required_version_groups_recorded"] = required_groups_complete
    return result


def build_exclusion_audit(
    candidate_frame: pd.DataFrame,
    day101a_report: Mapping[str, Any],
) -> Dict[str, Any]:
    require_columns(
        candidate_frame,
        {
            "player_id",
            "player_name",
            "web_name",
            "team_short_name",
            "position",
            "now_cost",
            "candidate_eligible",
            "exclusion_reasons",
        },
        "Day101A candidate_pool.csv",
    )
    excluded = candidate_frame[
        ~candidate_frame["candidate_eligible"].apply(bool_value)
    ].copy()
    entries: List[Dict[str, Any]] = []
    for row in excluded.sort_values("player_id").to_dict(orient="records"):
        entries.append(
            {
                "player_id": int(row["player_id"]),
                "player_name": str(row.get("player_name") or row.get("web_name") or ""),
                "team_short_name": str(row.get("team_short_name") or ""),
                "position": str(row.get("position") or ""),
                "now_cost": int(float(row.get("now_cost") or 0)),
                "reasons": list_value(row.get("exclusion_reasons")),
            }
        )
    report_counts = ((day101a_report.get("candidate_audit") or {}).get(
        "exclusion_reason_counts"
    ) or {})
    return {
        "candidate_count": int(candidate_frame["candidate_eligible"].apply(bool_value).sum()),
        "excluded_count": int(len(excluded)),
        "exclusion_reason_counts": dict(report_counts),
        "excluded_players": entries,
    }


def build_alternative_payload(
    all_squads: pd.DataFrame,
    day101a_report: Mapping[str, Any],
    variant: str,
) -> Dict[str, Any]:
    subset = all_squads[all_squads["variant"].astype(str) == variant].copy()
    summary = ((day101a_report.get("variants") or {}).get(variant) or {})
    players = []
    for row in subset.sort_values(
        ["position", "now_cost", "player_id"],
        key=lambda series: series.map(POSITION_ORDER) if series.name == "position" else series,
        ascending=[True, False, True],
    ).to_dict(orient="records"):
        players.append(
            {
                "player_id": int(row["player_id"]),
                "player_name": str(row.get("player_name") or row.get("web_name") or ""),
                "team_short_name": str(row.get("team_short_name") or ""),
                "position": str(row["position"]),
                "now_cost": int(row["now_cost"]),
                "gw1_predicted_points": float(row["gw1_predicted_points"]),
                "expected_minutes": float(row["expected_minutes"]),
                "start_probability": float(row["start_probability"]),
                "appearance_probability": float(row["appearance_probability"]),
                "fallback_used": bool_value(row["fallback_used"]),
                "fallback_level": int(float(row["fallback_level"])),
                "risk_flags": list_value(row.get("risk_flags")),
            }
        )
    return {
        "variant": variant,
        "variant_label": str(subset["variant_label"].iloc[0]),
        "total_cost_units": int(summary.get("total_cost_units") or 0),
        "bank_units": int(summary.get("bank_units") or 0),
        "objective_value": float(summary.get("objective_value") or 0.0),
        "squad_legality": dict(summary.get("squad_legality") or {}),
        "objective_reconciliation": dict(
            summary.get("objective_reconciliation") or {}
        ),
        "alternative_design": summary.get("alternative_design"),
        "players": players,
        "lineup_scope_note": (
            "Day101A alternative-squad roles are provisional objective assignments; "
            "the final Day101B lineup applies only to the Primary squad."
        ),
    }


def evaluation_contract() -> Dict[str, Any]:
    return {
        "contract_version": "gw1_player_evaluation_v1",
        "scope": "post_gw1_leakage_safe_evaluation",
        "source_rule": (
            "Use only the immutable final pre-deadline global player prediction "
            "snapshot and Model Team snapshot. Do not regenerate predictions after "
            "GW1 outcomes are known."
        ),
        "player_prediction_metrics": {
            "1_point_accuracy": {
                "metrics": ["MAE", "RMSE", "mean_error_bias"],
                "cohorts": [
                    "all_eligible_players",
                    "actually_played_players",
                    "60_plus_minutes_players",
                ],
            },
            "2_top_k_ranking_hits": {
                "k_values": [10, 20, 50],
                "metrics": ["hits", "precision_at_k", "recall_at_k"],
                "headline_metric": "top_20_hits",
            },
            "3_top_k_points_capture": {
                "primary_k": 20,
                "metrics": [
                    "actual_points_of_predicted_top_20",
                    "actual_points_of_actual_top_20",
                    "top_20_points_capture_ratio",
                ],
            },
            "4_ranking_quality": {
                "metrics": ["spearman_rank_correlation", "ndcg_at_20"]
            },
            "5_position_level": {
                "positions": ["GKP", "DEF", "MID", "FWD"],
                "metrics": ["MAE", "top_5_hits_by_position"],
            },
            "6_availability_and_minutes": {
                "metrics": ["appearance_accuracy", "start_accuracy", "minutes_MAE"],
                "inputs": [
                    "appearance_probability",
                    "start_probability",
                    "expected_minutes",
                ],
            },
        },
        "model_team_metrics_separate": True,
        "model_team_metrics": [
            "selected_15_actual_points",
            "starting_xi_actual_points",
            "captain_result",
            "vice_captain_contingency",
            "bench_points",
            "bench_order_consequences",
        ],
        "do_not_collapse_into_one_score": True,
    }


def build_validation_report(
    *,
    model_audit: Mapping[str, Any],
    final_refresh_audit: Mapping[str, Any],
    version_audit: Mapping[str, Any],
    source_metadata: Mapping[str, Mapping[str, Any]],
    as_of_time: str,
    fpl_deadline_time: str,
    day76d_report: Mapping[str, Any],
    day97a_report: Mapping[str, Any],
    day101a_report: Mapping[str, Any],
    day101b_report: Mapping[str, Any],
    final_freeze: bool,
) -> Dict[str, Any]:
    as_of = parse_utc(as_of_time, "as_of_time")
    deadline = parse_utc(fpl_deadline_time, "fpl_deadline_time")
    deadline_pass = as_of < deadline

    fingerprint_missing = [
        name
        for name, metadata in source_metadata.items()
        if not str(metadata.get("sha256") or "").strip()
    ]
    fingerprints_pass = not fingerprint_missing

    legality = model_audit.get("legality") or {}
    checks = {
        "player_count_15": {
            "passed": int(model_audit["selected_player_count"]) == 15,
            "detail": "selected_player_count=%s" % model_audit["selected_player_count"],
        },
        "position_quotas_pass": {
            "passed": bool(model_audit["position_quotas_pass"]),
            "detail": "actual=%s expected=%s"
            % (model_audit["position_counts"], model_audit["expected_position_quotas"]),
        },
        "club_limits_pass": {
            "passed": bool(model_audit["club_limits_pass"]),
            "detail": "club_counts=%s max=%s"
            % (model_audit["club_counts"], model_audit["max_players_per_club"]),
        },
        "budget_and_bank_reconcile": {
            "passed": bool(model_audit["budget_and_bank_reconcile"]),
            "detail": "cost=%s bank=%s budget=%s"
            % (
                model_audit["total_cost_units"],
                model_audit["bank_units"],
                model_audit["initial_budget_units"],
            ),
        },
        "starting_xi_and_formation_pass": {
            "passed": bool(legality.get("valid"))
            and int(model_audit["starting_xi_count"]) == 11,
            "detail": "formation=%s legality=%s"
            % (model_audit["formation"], legality.get("valid")),
        },
        "bench_order_pass": {
            "passed": bool(legality.get("valid"))
            and int(model_audit["bench_count"]) == 4,
            "detail": "bench_order=%s" % model_audit["bench_order"],
        },
        "captain_and_vice_pass": {
            "passed": bool(
                model_audit["captain_and_vice_distinct_starting_members"]
            ),
            "detail": "captain=%s vice=%s"
            % (
                model_audit["captain_player_id"],
                model_audit["vice_captain_player_id"],
            ),
        },
        "no_duplicate_players": {
            "passed": int(model_audit["duplicate_player_count"]) == 0,
            "detail": "duplicate_player_count=%s"
            % model_audit["duplicate_player_count"],
        },
        "player_identity_price_availability_match_final_refresh": {
            "passed": bool(final_refresh_audit["passed"]),
            "detail": "mismatch_count=%s" % final_refresh_audit["mismatch_count"],
        },
        "objective_total_reconciles_to_player_components": {
            "passed": bool(
                (model_audit.get("objective_reconciliation") or {}).get("passed")
            ),
            "detail": "objective_value=%s" % model_audit["objective_value"],
        },
        "required_versions_recorded": {
            "passed": bool(version_audit["all_required_version_groups_recorded"]),
            "detail": "rules/feature/prediction/objective/artifact versions recorded",
        },
        "input_fingerprints_recorded": {
            "passed": fingerprints_pass,
            "detail": "missing=%s" % fingerprint_missing,
        },
        "as_of_time_before_fpl_deadline": {
            "passed": deadline_pass,
            "detail": "as_of=%s deadline=%s" % (format_utc(as_of), format_utc(deadline)),
        },
        "status_preview_only": {
            "passed": (
                day101b_report.get("preview_only") is True
                and day101a_report.get("preview_only") is True
                and day97a_report.get("preview_only") is True
                and str(day101b_report.get("recommendation_status")) == RECOMMENDATION_STATUS
            ),
            "detail": "recommendation_status=preview_only",
        },
        "production_and_manager_state_writes_disabled": {
            "passed": all(
                value is False
                for value in (
                    day101b_report.get("production_approved"),
                    day101b_report.get("writes_database"),
                    day101b_report.get("writes_predictions_table"),
                    day101b_report.get("writes_recommendations"),
                    day101b_report.get("writes_squad_state"),
                    day101a_report.get("production_approved"),
                    day101a_report.get("writes_database"),
                    day101a_report.get("writes_squad_state"),
                    day97a_report.get("production_approved"),
                    day97a_report.get("writes_database"),
                    day76d_report.get("writes_database"),
                    day76d_report.get("writes_predictions_table"),
                )
            ),
            "detail": "all production/database/prediction/recommendation/squad-state writes disabled",
        },
    }
    blockers = [name for name, payload in checks.items() if not payload["passed"]]
    warnings = [
        "GW2-GW5 player predictions remain unavailable in gw1_only_fallback mode."
    ]
    if not final_freeze:
        warnings.insert(
            0,
            "This is a pre-deadline candidate snapshot, not the final leakage-safe GW1 freeze.",
        )
    historical_approval_incomplete = not bool(
        day76d_report.get("historical_multi_season_backtest_complete")
    )
    return {
        "validation_version": SNAPSHOT_VERSION,
        "target_season": str(day101b_report["target_season"]),
        "target_gw": 1,
        "as_of_time_utc": format_utc(as_of),
        "fpl_deadline_time_utc": format_utc(deadline),
        "snapshot_kind": SNAPSHOT_KIND_FINAL if final_freeze else SNAPSHOT_KIND_CANDIDATE,
        "final_pre_deadline_snapshot_frozen": bool(final_freeze),
        "checks": checks,
        "final_refresh_reconciliation": dict(final_refresh_audit),
        "version_audit": dict(version_audit),
        "input_fingerprint_count": len(source_metadata),
        "historical_multi_season_backtest_complete": bool(
            day76d_report.get("historical_multi_season_backtest_complete")
        ),
        "historical_approval_incomplete": historical_approval_incomplete,
        "production_approved": False,
        "preview_only": True,
        "writes_database": False,
        "writes_predictions_table": False,
        "writes_recommendations": False,
        "writes_squad_state": False,
        "passed": not blockers,
        "ready_for_manual_fpl_entry_review": not blockers,
        "ready_for_final_freeze": not blockers,
        "stop_point_satisfied": not blockers,
        "blockers": blockers,
        "warnings": warnings,
    }


def build_run_metadata_payload(
    day101b_report: Mapping[str, Any],
    source_metadata: Mapping[str, Mapping[str, Any]],
    version_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    source_run = day101b_report.get("run_metadata") or {}
    source_versions = source_run.get("versions") or {}
    return build_run_metadata(
        run_id=None,
        run_type="evaluation",
        artifact_type=ARTIFACT_TYPE,
        source_seasons=list(source_run.get("source_seasons") or ["2025_26"]),
        target_season=str(day101b_report["target_season"]),
        target_gw=1,
        horizon=1,
        as_of_time=str(day101b_report["as_of_time_utc"]),
        prediction_mode="pre_gw1_prior",
        created_at=utc_now(),
        model_version=str(source_versions.get("model_version") or "pre_gw1_heuristic_preview"),
        rules_versions=dict(version_audit.get("rules") or {}),
        manifest_version=SNAPSHOT_VERSION,
        artifact_version=ARTIFACT_VERSION,
        additional_versions={
            "snapshot_version": SNAPSHOT_VERSION,
            "source_day101b_optimizer_version": str(
                day101b_report.get("optimizer_version") or "day101b_v1"
            ),
            "objective_contract_version": str(
                (version_audit.get("objective") or {}).get("contract_version") or ""
            ),
            "objective_policy_version": str(
                (version_audit.get("objective") or {}).get("policy_version") or ""
            ),
        },
        provenance={
            "producer": "ml.validation.export_gw1_pre_deadline_snapshot",
            "inputs": provenance_inputs_from_file_metadata(source_metadata),
            "parent_run_ids": [str(source_run.get("run_id") or "")],
            "notes": [
                "Day101C local GW1 validation and immutable export.",
                "Consumes the fixed Day101A Primary squad and Day101B final weekly setup without manual Model Team substitution.",
                "Reconciles selected player identity, team, position, price, availability probabilities, and risk flags to the Day76D refresh used by Day97A.",
                "Exports the plan-required local squad deliverables plus the global player prediction baseline for later evaluation.",
                "All outputs remain preview_only and write-disabled.",
            ],
        },
    ).to_dict()


def player_name_lookup(primary_export: pd.DataFrame) -> Dict[int, str]:
    return {
        int(row["player_id"]): str(row.get("web_name") or row.get("player_name") or "")
        for row in primary_export.to_dict(orient="records")
    }


def primary_player_records(primary_export: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    ordered = primary_export.copy()
    ordered["position_order"] = ordered["position"].map(POSITION_ORDER)
    ordered = ordered.sort_values(["position_order", "now_cost", "player_id"], ascending=[True, False, True])
    for row in ordered.to_dict(orient="records"):
        payload = dict(row)
        payload.pop("position_order", None)
        payload["risk_flags"] = list_value(payload.get("risk_flags"))
        for key, value in list(payload.items()):
            if isinstance(value, float) and pd.isna(value):
                payload[key] = None
        records.append(payload)
    return records


def build_opening_squad_payload(
    *,
    primary_export: pd.DataFrame,
    all_squads: pd.DataFrame,
    day101a_report: Mapping[str, Any],
    day101b_report: Mapping[str, Any],
    day97a_report: Mapping[str, Any],
    day76d_report: Mapping[str, Any],
    model_audit: Mapping[str, Any],
    validation: Mapping[str, Any],
    exclusions: Mapping[str, Any],
    version_audit: Mapping[str, Any],
    source_metadata: Mapping[str, Mapping[str, Any]],
    run_metadata: Mapping[str, Any],
    final_freeze: bool,
) -> Dict[str, Any]:
    primary_plan = ((day101b_report.get("plans") or {}).get("primary") or {})
    names = player_name_lookup(primary_export)
    bench_order = [int(v) for v in primary_plan["bench_order"]]
    return {
        "artifact_type": "gw1_opening_squad",
        "artifact_version": ARTIFACT_VERSION,
        "snapshot_version": SNAPSHOT_VERSION,
        "target_season": str(day101b_report["target_season"]),
        "target_gw": 1,
        "as_of_time_utc": str(day101b_report["as_of_time_utc"]),
        "fpl_deadline_time_utc": validation["fpl_deadline_time_utc"],
        "snapshot_kind": validation["snapshot_kind"],
        "final_pre_deadline_snapshot_frozen": bool(final_freeze),
        "recommendation_status": RECOMMENDATION_STATUS,
        "preview_only": True,
        "production_approved": False,
        "historical_multi_season_backtest_complete": False,
        "historical_approval_incomplete": True,
        "writes_database": False,
        "writes_predictions_table": False,
        "writes_recommendations": False,
        "writes_squad_state": False,
        "primary": {
            "players": primary_player_records(primary_export),
            "total_cost_units": int(model_audit["total_cost_units"]),
            "bank_units": int(model_audit["bank_units"]),
            "starting_player_ids": [int(v) for v in primary_plan["starting_player_ids"]],
            "formation": str(primary_plan["formation"]),
            "captain_player_id": int(primary_plan["captain_player_id"]),
            "captain_name": names[int(primary_plan["captain_player_id"])],
            "vice_captain_player_id": int(primary_plan["vice_captain_player_id"]),
            "vice_captain_name": names[int(primary_plan["vice_captain_player_id"])],
            "bench_order": bench_order,
            "bench_names": [names[player_id] for player_id in bench_order],
            "objective_value": float(primary_plan["objective_value"]),
            "objective_reconciliation": dict(
                primary_plan.get("objective_reconciliation") or {}
            ),
            "legality": dict(primary_plan.get("legality") or {}),
        },
        "alternative_a": build_alternative_payload(
            all_squads, day101a_report, "alternative_a"
        ),
        "alternative_b": build_alternative_payload(
            all_squads, day101a_report, "alternative_b"
        ),
        "horizon": {
            "requested_horizon": int(day97a_report["requested_horizon"]),
            "effective_horizon": int(day97a_report["effective_horizon"]),
            "objective_mode": str(day97a_report["objective_mode"]),
            "future_fixture_context": str(day97a_report["future_fixture_context"]),
            "missing_future_predictions_are_zero": False,
        },
        "key_exclusions": dict(exclusions),
        "version_audit": dict(version_audit),
        "input_artifacts": dict(source_metadata),
        "run_metadata": dict(run_metadata),
        "validation_summary": {
            "passed": validation["passed"],
            "stop_point_satisfied": validation["stop_point_satisfied"],
            "blockers": list(validation["blockers"]),
            "warnings": list(validation["warnings"]),
        },
        "source_status": {
            "day76d_historical_multi_season_backtest_complete": bool(
                day76d_report.get("historical_multi_season_backtest_complete")
            ),
            "day76d_component_model_stack_complete": bool(
                day76d_report.get("component_model_stack_complete")
            ),
            "day76d_production_approved": bool(
                day76d_report.get("production_approved")
            ),
        },
        "manual_review_checklist": [
            "Confirm no material team news, injury, suspension, transfer, or availability change has occurred since the source as_of_time.",
            "If material news changed, rerun the standardized refresh chain before using this snapshot.",
            "Confirm official FPL entry matches the Primary 15, XI, formation, captain, vice, and bench order.",
            "Confirm the official GW1 deadline has not changed.",
            "Use --final-freeze only after the last material-news refresh and manual entry review.",
        ],
        "team_alex_included": False,
        "team_alex_note": "Team Alex / Gliding Tiger remains a separate human-in-the-loop track and must be frozen separately.",
    }


def fmt_price(units: Any) -> str:
    return "£%.1fm" % (float(units) / 10.0)


def fmt_points(value: Any) -> str:
    number = nullable_float(value)
    return "—" if number is None else "%.2f" % number


def fmt_prob(value: Any) -> str:
    number = nullable_float(value)
    return "—" if number is None else "%.0f%%" % (100.0 * number)


def role_label(row: Mapping[str, Any]) -> str:
    if bool_value(row.get("is_captain")):
        return "Starter (C)"
    if bool_value(row.get("is_vice_captain")):
        return "Starter (VC)"
    role = str(row.get("role") or "")
    if role == "starter":
        return "Starter"
    if role == "bench_gk":
        return "Bench GK"
    if role.startswith("bench_"):
        return "Bench %s" % role.split("_", 1)[1]
    return role


def build_opening_squad_markdown(
    payload: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> str:
    primary = payload["primary"]
    players = primary["players"]
    lines: List[str] = [
        "# Day101C — Local GW1 Opening Squad Validation and Export",
        "",
        "- Season: `%s`" % payload["target_season"],
        "- Gameweek: `GW1`",
        "- Snapshot kind: `%s`" % payload["snapshot_kind"],
        "- Source as-of time: `%s`" % payload["as_of_time_utc"],
        "- FPL deadline: `%s`" % payload["fpl_deadline_time_utc"],
        "- Status: `preview_only`",
        "- Production approved: `False`",
        "- Database / prediction / recommendation / squad-state writes: `False`",
        "",
        "## Primary Model Team — 15-player squad",
        "",
    ]
    for position in ("GKP", "DEF", "MID", "FWD"):
        lines.extend(
            [
                "### %s" % position,
                "",
                "| Player | Team | Price | Role | GW1 | GW2 | GW3 | GW4 | GW5 | Exp min | Start | Appear | Status / official chance | Confidence | Fallback | Risks |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|",
            ]
        )
        for player in players:
            if player["position"] != position:
                continue
            chance = nullable_float(player.get("chance_of_playing_next_round"))
            status_text = str(player.get("status") or "")
            if chance is not None:
                status_text = "%s / %.0f%%" % (status_text or "?", chance)
            risks = ", ".join(list_value(player.get("risk_flags"))) or "none"
            fallback = "%s / L%s" % (
                str(player.get("fallback_policy_used") or "fallback"),
                int(player.get("fallback_level") or 0),
            )
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %.1f | %s | %s | %s | %s | %s | %s |"
                % (
                    player["web_name"],
                    player["team_short_name"],
                    fmt_price(player["now_cost"]),
                    role_label(player),
                    fmt_points(player.get("gw1_predicted_points")),
                    fmt_points(player.get("gw2_predicted_points")),
                    fmt_points(player.get("gw3_predicted_points")),
                    fmt_points(player.get("gw4_predicted_points")),
                    fmt_points(player.get("gw5_predicted_points")),
                    float(player["expected_minutes"]),
                    fmt_prob(player.get("start_probability")),
                    fmt_prob(player.get("appearance_probability")),
                    status_text or "—",
                    str(player.get("prediction_confidence") or "—"),
                    fallback,
                    risks,
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Cost, bank, XI, captaincy, and bench",
            "",
            "- Total cost: **%s**" % fmt_price(primary["total_cost_units"]),
            "- Bank: **%s**" % fmt_price(primary["bank_units"]),
            "- Formation: **%s**" % primary["formation"],
            "- Captain: **%s**" % primary["captain_name"],
            "- Vice-captain: **%s**" % primary["vice_captain_name"],
            "",
            "### Starting XI",
            "",
        ]
    )
    starter_ids = set(int(v) for v in primary["starting_player_ids"])
    for position in ("GKP", "DEF", "MID", "FWD"):
        names = [
            player["web_name"]
            for player in players
            if player["position"] == position and int(player["player_id"]) in starter_ids
        ]
        if names:
            lines.append("- **%s:** %s" % (position, ", ".join(names)))
    lines.extend(["", "### Ordered bench", ""])
    for index, name in enumerate(primary["bench_names"]):
        label = "GK" if index == 0 else str(index)
        lines.append("- Bench %s: **%s**" % (label, name))

    horizon = payload["horizon"]
    lines.extend(
        [
            "",
            "## GW1 and horizon projection status",
            "",
            "- Requested horizon: **GW1–GW%s**" % horizon["requested_horizon"],
            "- Effective prediction horizon: **%s Gameweek(s)**" % horizon["effective_horizon"],
            "- Objective mode: `%s`" % horizon["objective_mode"],
            "- Future fixture context: `%s`" % horizon["future_fixture_context"],
            "- Missing GW2–GW5 predictions are **missing**, not zero-point predictions.",
            "",
            "## Key risks",
            "",
        ]
    )
    risk_lines = []
    for player in players:
        risks = list_value(player.get("risk_flags"))
        if risks or bool_value(player.get("manual_review_required")):
            risk_lines.append(
                "- **%s:** %s"
                % (player["web_name"], ", ".join(risks) if risks else "manual review")
            )
    lines.extend(risk_lines or ["- No selected-player risk flags beyond the preserved fallback/uncertainty labels."])

    exclusions = payload["key_exclusions"]
    lines.extend(
        [
            "",
            "## Key exclusions",
            "",
            "- Optimizer-safe candidates: **%s**" % exclusions["candidate_count"],
            "- Excluded players: **%s**" % exclusions["excluded_count"],
        ]
    )
    reason_counts = exclusions.get("exclusion_reason_counts") or {}
    if reason_counts:
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            lines.append("- `%s`: %s" % (reason, count))
    else:
        lines.append("- No structured exclusion reasons recorded.")

    lines.extend(["", "## Two alternative squads", ""])
    for variant_key, title in (
        ("alternative_a", "Alternative A — stronger starting XI / cheaper bench"),
        ("alternative_b", "Alternative B — lower minutes / availability risk"),
    ):
        alt = payload[variant_key]
        lines.extend(
            [
                "### %s" % title,
                "",
                "- Cost: **%s**" % fmt_price(alt["total_cost_units"]),
                "- Bank: **%s**" % fmt_price(alt["bank_units"]),
                "- Objective: **%.6f**" % float(alt["objective_value"]),
            ]
        )
        design = alt.get("alternative_design") or {}
        if design.get("description"):
            lines.append("- Design: %s" % design["description"])
        for position in ("GKP", "DEF", "MID", "FWD"):
            names = [
                player["player_name"] or player.get("web_name", "")
                for player in alt["players"]
                if player["position"] == position
            ]
            lines.append("- **%s:** %s" % (position, ", ".join(names)))
        lines.extend(["", "> %s" % alt["lineup_scope_note"], ""])

    lines.extend(
        [
            "## Final manual-review checklist",
            "",
            "- [x] Day101C automated legality / budget / lineup / captaincy checks pass.",
            "- [x] Primary player identity, team, position, price, and availability values reconcile to the final Day76D refresh used by Day97A.",
            "- [x] Input fingerprints and required versions are recorded.",
            "- [x] Source `as_of_time` is before the recorded FPL deadline.",
            "- [ ] Recheck late injury, suspension, transfer, and lineup news immediately before the deadline.",
            "- [ ] If material news changes, rerun the standardized refresh chain rather than manually editing the Model Team.",
            "- [ ] Confirm the official FPL entry matches the Primary 15, XI, formation, captain, vice, and bench order.",
            "- [ ] Use `--final-freeze` only after the final material-news refresh and entry review.",
            "",
            "## Historical approval / production boundary",
            "",
            "> **Historical approval is incomplete.** The current prediction path is a `pre_gw1_heuristic_preview`; multi-season historical backtesting and the complete component-model stack are not approved, and production approval remains false.",
            "",
            "## Validation result",
            "",
            "- Passed: `%s`" % validation["passed"],
            "- Ready for manual FPL entry review: `%s`" % validation["ready_for_manual_fpl_entry_review"],
            "- Ready for final freeze: `%s`" % validation["ready_for_final_freeze"],
            "- Stop point satisfied: `%s`" % validation["stop_point_satisfied"],
            "",
            "## Stop point",
            "",
            "> A private GW1 squad can be reviewed and manually entered into official FPL without requiring an API, frontend, or database write.",
            "",
        ]
    )
    return "\n".join(lines)


def build_validation_markdown(validation: Mapping[str, Any]) -> str:
    label_map = {
        "player_count_15": "Player count = 15",
        "position_quotas_pass": "Position quotas pass",
        "club_limits_pass": "Club limits pass",
        "budget_and_bank_reconcile": "Budget and bank reconcile",
        "starting_xi_and_formation_pass": "Starting XI and formation pass",
        "bench_order_pass": "Bench order passes",
        "captain_and_vice_pass": "Captain and vice are distinct starting-XI members",
        "no_duplicate_players": "No duplicate players",
        "player_identity_price_availability_match_final_refresh": "IDs, teams, positions, prices, and availability match final refresh",
        "objective_total_reconciles_to_player_components": "Objective total reconciles to player components",
        "required_versions_recorded": "Rules, feature, prediction, objective, and artifact versions recorded",
        "input_fingerprints_recorded": "Input fingerprints recorded",
        "as_of_time_before_fpl_deadline": "as_of_time is before FPL deadline",
        "status_preview_only": "Status is preview_only",
        "production_and_manager_state_writes_disabled": "Production and manager-state writes disabled",
    }
    lines = [
        "# Day101C — GW1 Opening Squad Validation",
        "",
        "- Snapshot kind: `%s`" % validation["snapshot_kind"],
        "- Source as-of: `%s`" % validation["as_of_time_utc"],
        "- FPL deadline: `%s`" % validation["fpl_deadline_time_utc"],
        "",
        "## Required validation",
        "",
    ]
    for key, payload in validation["checks"].items():
        mark = "x" if payload["passed"] else " "
        lines.append("- [%s] **%s** — %s" % (mark, label_map[key], payload["detail"]))
    lines.extend(["", "## Warnings", ""])
    if validation["warnings"]:
        lines.extend("- %s" % item for item in validation["warnings"])
    else:
        lines.append("- none")
    lines.extend(["", "## Blockers", ""])
    if validation["blockers"]:
        lines.extend("- %s" % item for item in validation["blockers"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Historical approval",
            "",
            "- Historical multi-season backtest complete: `%s`"
            % validation["historical_multi_season_backtest_complete"],
            "- Historical approval incomplete: `%s`"
            % validation["historical_approval_incomplete"],
            "- Production approved: `False`",
            "",
            "## Stop point",
            "",
            "> A private GW1 squad can be reviewed and manually entered into official FPL without requiring an API, frontend, or database write.",
            "",
            "Stop point satisfied: `%s`" % validation["stop_point_satisfied"],
            "",
        ]
    )
    return "\n".join(lines)


def artifact_definitions() -> Dict[str, Tuple[str, str]]:
    # Keep the earlier candidate/evaluation artifacts and add every primary output
    # required by the Day101C plan. No plan-required deliverable is omitted.
    return {
        "gw1_opening_squad_csv": ("gw1_opening_squad", "csv"),
        "gw1_opening_squad_json": ("gw1_opening_squad", "json"),
        "gw1_opening_squad_md": ("gw1_opening_squad", "md"),
        "gw1_opening_squad_validation_json": (
            "gw1_opening_squad_validation",
            "json",
        ),
        "gw1_opening_squad_validation_md": (
            "gw1_opening_squad_validation",
            "md",
        ),
        "model_team_snapshot_csv": ("model_team_snapshot", "csv"),
        "global_player_prediction_snapshot_csv": (
            "global_player_prediction_snapshot",
            "csv",
        ),
        "evaluation_contract_json": ("evaluation_contract", "json"),
        "run_metadata_json": ("run_metadata", "json"),
        "gw1_snapshot_report_json": ("gw1_snapshot_report", "json"),
        "gw1_snapshot_report_md": ("gw1_snapshot_report", "md"),
    }


def write_outputs(
    *,
    artifact_root: Path,
    target_season: str,
    as_of_time: str,
    run_metadata: Mapping[str, Any],
    primary_export: pd.DataFrame,
    global_snapshot: pd.DataFrame,
    opening_payload: Dict[str, Any],
    validation: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    storage = LocalArtifactStorage(artifact_root)
    keys = {
        name: build_immutable_artifact_key(
            artifact_type=ARTIFACT_TYPE,
            season=target_season,
            target_gw=1,
            as_of_time=as_of_time,
            run_id=str(run_metadata["run_id"]),
            version=SNAPSHOT_VERSION,
            filename=filename,
            extension=extension,
        )
        for name, (filename, extension) in artifact_definitions().items()
    }
    artifact_manifest = {
        "root": str(Path(artifact_root).expanduser().resolve()),
        "keys": keys,
    }
    opening_payload["artifacts"] = artifact_manifest
    validation["artifacts"] = artifact_manifest

    opening_md = build_opening_squad_markdown(opening_payload, validation)
    validation_md = build_validation_markdown(validation)
    technical_report = {
        "artifact_type": ARTIFACT_TYPE,
        "snapshot_version": SNAPSHOT_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "target_season": target_season,
        "target_gw": 1,
        "snapshot_kind": validation["snapshot_kind"],
        "final_pre_deadline_snapshot_frozen": validation[
            "final_pre_deadline_snapshot_frozen"
        ],
        "as_of_time_utc": validation["as_of_time_utc"],
        "fpl_deadline_time_utc": validation["fpl_deadline_time_utc"],
        "preview_only": True,
        "production_approved": False,
        "writes_database": False,
        "writes_predictions_table": False,
        "writes_recommendations": False,
        "writes_squad_state": False,
        "run_metadata": dict(run_metadata),
        "validation": validation,
        "evaluation_contract": evaluation_contract(),
        "artifacts": artifact_manifest,
        "passed": validation["passed"],
        "ready_for_final_freeze": validation["ready_for_final_freeze"],
        "stop_point_satisfied": validation["stop_point_satisfied"],
        "blockers": list(validation["blockers"]),
        "warnings": list(validation["warnings"]),
    }

    payloads = {
        "gw1_opening_squad_csv": primary_export.to_csv(index=False),
        "gw1_opening_squad_json": json.dumps(
            opening_payload, indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        + "\n",
        "gw1_opening_squad_md": opening_md,
        "gw1_opening_squad_validation_json": json.dumps(
            validation, indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        + "\n",
        "gw1_opening_squad_validation_md": validation_md,
        "model_team_snapshot_csv": primary_export.to_csv(index=False),
        "global_player_prediction_snapshot_csv": global_snapshot.to_csv(index=False),
        "evaluation_contract_json": json.dumps(
            evaluation_contract(), indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n",
        "run_metadata_json": json.dumps(
            run_metadata, indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        + "\n",
        "gw1_snapshot_report_json": json.dumps(
            technical_report, indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        + "\n",
        "gw1_snapshot_report_md": validation_md,
    }

    stored: Dict[str, Dict[str, Any]] = {}
    for name, content in payloads.items():
        stored[name] = storage.write_immutable_text(keys[name], content).to_dict()
    return stored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Day101C local GW1 validation and immutable export. Produces the exact "
            "plan-required gw1_opening_squad CSV/JSON/Markdown and validation "
            "JSON/Markdown outputs, plus the global GW1 evaluation baseline."
        )
    )
    parser.add_argument("--opening-lineup-report-json", required=True)
    parser.add_argument(
        "--fpl-deadline-time",
        required=True,
        help="Official GW1 FPL deadline as timezone-aware ISO-8601.",
    )
    parser.add_argument("--target-season", default="2026_27")
    parser.add_argument("--artifact-root", default="/private/tmp/fpl-artifacts")
    parser.add_argument("--squad-rules-config", default=None)
    parser.add_argument(
        "--final-freeze",
        action="store_true",
        help=(
            "Explicitly mark this run as the final pre-deadline Model Team and global "
            "player-prediction evaluation freeze. Use only after the last material-news "
            "refresh and official-FPL manual review."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deadline = format_utc(parse_utc(args.fpl_deadline_time, "fpl_deadline_time"))

    day101b_path = Path(args.opening_lineup_report_json).expanduser().resolve()
    day101b_report = load_json(day101b_path, "Day101B opening lineup report")
    validate_day101b_report(day101b_report)
    if str(day101b_report.get("target_season")) != str(args.target_season):
        raise GW1SnapshotError("Target season mismatch.")

    day101a_path, day101a_report = load_day101a_source(day101b_report)
    day97a_path, day97a_report = load_day97a_source(day101a_report)
    day76d_path, day76d_report = load_day76d_source(day97a_report)

    squad_path = resolve_report_artifact(day101a_report, "opening_squad_csv", "Day101A")
    candidate_path = resolve_report_artifact(day101a_report, "candidate_pool_csv", "Day101A")
    lineup_path = resolve_report_artifact(day101b_report, "opening_lineup_csv", "Day101B")
    long_horizon_path = resolve_report_artifact(
        day97a_report, "player_prediction_horizon_csv", "Day97A"
    )

    preview_path, preview, standard_path, standard = load_final_refresh_frames(day76d_path)

    all_squads = normalize_all_squads(
        pd.read_csv(squad_path, low_memory=False), day101a_report
    )
    primary_squad = all_squads[all_squads["variant"].astype(str) == "primary"].copy()
    lineup = normalize_primary_lineup(
        pd.read_csv(lineup_path, low_memory=False), day101b_report
    )
    long_horizon = pd.read_csv(long_horizon_path, low_memory=False)
    requested_horizon = int(day97a_report["requested_horizon"])
    horizon_map = build_horizon_projection_map(
        long_horizon,
        primary_squad["player_id"].astype(int).tolist(),
        requested_horizon,
    )

    primary_export, final_refresh_audit = reconcile_primary_with_final_refresh(
        primary_squad,
        lineup,
        preview,
        standard,
        horizon_map,
    )

    rules = load_squad_transfer_rules(
        args.target_season,
        config_path=(
            None
            if args.squad_rules_config is None
            else Path(args.squad_rules_config).expanduser().resolve()
        ),
    )
    engine = SquadLegalityEngine(rules)
    model_audit = build_model_team_audit(
        primary_export,
        day101a_report,
        day101b_report,
        engine,
    )
    global_snapshot, global_audit = build_global_prediction_snapshot(
        long_horizon, args.target_season
    )
    candidate_frame = pd.read_csv(candidate_path, low_memory=False)
    exclusion_audit = build_exclusion_audit(candidate_frame, day101a_report)
    version_audit = build_version_audit(
        day76d_report,
        day97a_report,
        day101a_report,
        day101b_report,
    )

    source_metadata = {
        "day101b_report": file_metadata(
            day101b_path,
            artifact_type="opening_lineup_optimizer",
            run_id=str((day101b_report.get("run_metadata") or {}).get("run_id") or ""),
            version=str(day101b_report.get("optimizer_version") or "day101b_v1"),
        ),
        "day101b_opening_lineup": file_metadata(
            lineup_path,
            artifact_type="opening_lineup_optimizer",
            run_id=str((day101b_report.get("run_metadata") or {}).get("run_id") or ""),
            version=str(day101b_report.get("optimizer_version") or "day101b_v1"),
        ),
        "day101a_report": file_metadata(
            day101a_path,
            artifact_type="opening_squad_optimizer",
            run_id=str((day101a_report.get("run_metadata") or {}).get("run_id") or ""),
            version=str(day101a_report.get("optimizer_version") or "day101a_v1"),
        ),
        "day101a_opening_squad": file_metadata(
            squad_path,
            artifact_type="opening_squad_optimizer",
            run_id=str((day101a_report.get("run_metadata") or {}).get("run_id") or ""),
            version=str(day101a_report.get("optimizer_version") or "day101a_v1"),
        ),
        "day101a_candidate_pool": file_metadata(
            candidate_path,
            artifact_type="opening_squad_optimizer",
            run_id=str((day101a_report.get("run_metadata") or {}).get("run_id") or ""),
            version=str(day101a_report.get("optimizer_version") or "day101a_v1"),
        ),
        "day97a_report": file_metadata(
            day97a_path,
            artifact_type="player_prediction_horizon",
            run_id=str((day97a_report.get("run_metadata") or {}).get("run_id") or ""),
            version=str(day97a_report.get("horizon_version") or "day97a_v1"),
        ),
        "day97a_player_prediction_horizon": file_metadata(
            long_horizon_path,
            artifact_type="player_prediction_horizon",
            run_id=str((day97a_report.get("run_metadata") or {}).get("run_id") or ""),
            version=str(day97a_report.get("horizon_version") or "day97a_v1"),
        ),
        "day76d_refresh_report": file_metadata(
            day76d_path,
            artifact_type="pre_gw1_player_prediction_refresh",
            run_id=str((day76d_report.get("run_metadata") or {}).get("run_id") or ""),
            version=str(day76d_report.get("refresh_version") or "day76d_v1"),
        ),
        "day76d_prediction_preview": file_metadata(
            preview_path,
            artifact_type="pre_gw1_player_prediction_refresh",
            run_id=str((day76d_report.get("run_metadata") or {}).get("run_id") or ""),
            version=str(day76d_report.get("refresh_version") or "day76d_v1"),
        ),
        "day76d_standard_player_predictions": file_metadata(
            standard_path,
            artifact_type="pre_gw1_player_prediction_refresh",
            run_id=str((day76d_report.get("run_metadata") or {}).get("run_id") or ""),
            version=str(day76d_report.get("refresh_version") or "day76d_v1"),
        ),
    }

    run_metadata = build_run_metadata_payload(
        day101b_report,
        source_metadata,
        version_audit,
    )
    validation = build_validation_report(
        model_audit=model_audit,
        final_refresh_audit=final_refresh_audit,
        version_audit=version_audit,
        source_metadata=source_metadata,
        as_of_time=str(day101b_report["as_of_time_utc"]),
        fpl_deadline_time=deadline,
        day76d_report=day76d_report,
        day97a_report=day97a_report,
        day101a_report=day101a_report,
        day101b_report=day101b_report,
        final_freeze=bool(args.final_freeze),
    )
    if not validation["passed"]:
        raise GW1SnapshotError("Day101C validation failed: %s" % validation["blockers"])

    opening_payload = build_opening_squad_payload(
        primary_export=primary_export,
        all_squads=all_squads,
        day101a_report=day101a_report,
        day101b_report=day101b_report,
        day97a_report=day97a_report,
        day76d_report=day76d_report,
        model_audit=model_audit,
        validation=validation,
        exclusions=exclusion_audit,
        version_audit=version_audit,
        source_metadata=source_metadata,
        run_metadata=run_metadata,
        final_freeze=bool(args.final_freeze),
    )

    stored = write_outputs(
        artifact_root=Path(args.artifact_root),
        target_season=args.target_season,
        as_of_time=str(day101b_report["as_of_time_utc"]),
        run_metadata=run_metadata,
        primary_export=primary_export,
        global_snapshot=global_snapshot,
        opening_payload=opening_payload,
        validation=validation,
    )

    print("Day101C local GW1 validation and export complete.")
    print("run_id:", run_metadata["run_id"])
    print("snapshot_kind:", validation["snapshot_kind"])
    print(
        "final_pre_deadline_snapshot_frozen:",
        str(validation["final_pre_deadline_snapshot_frozen"]).lower(),
    )
    print("player_count:", model_audit["selected_player_count"])
    print("position_quotas_pass:", str(model_audit["position_quotas_pass"]).lower())
    print("club_limits_pass:", str(model_audit["club_limits_pass"]).lower())
    print("budget_and_bank_reconcile:", str(model_audit["budget_and_bank_reconcile"]).lower())
    print("formation:", model_audit["formation"])
    print("captain_player_id:", model_audit["captain_player_id"])
    print("vice_captain_player_id:", model_audit["vice_captain_player_id"])
    print("bench_order:", model_audit["bench_order"])
    print("final_refresh_reconciliation:", str(final_refresh_audit["passed"]).lower())
    print("required_versions_recorded:", str(version_audit["all_required_version_groups_recorded"]).lower())
    print("as_of_before_deadline:", str(validation["checks"]["as_of_time_before_fpl_deadline"]["passed"]).lower())
    print("global_player_predictions:", global_audit["player_count"])
    print("global_predictions_available:", global_audit["prediction_available_count"])
    print("immutable_artifacts:", len(stored))
    print("preview_only: true")
    print("writes_database: false")
    print("writes_squad_state: false")
    print("ready_for_manual_fpl_entry_review:", str(validation["ready_for_manual_fpl_entry_review"]).lower())
    print("ready_for_final_freeze:", str(validation["ready_for_final_freeze"]).lower())
    print("stop_point_satisfied:", str(validation["stop_point_satisfied"]).lower())


if __name__ == "__main__":
    main()
