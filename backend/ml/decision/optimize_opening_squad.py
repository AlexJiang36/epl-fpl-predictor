from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from app.rules.squad import load_squad_transfer_rules
from ml.artifacts.paths import build_immutable_artifact_key
from ml.artifacts.storage import LocalArtifactStorage
from ml.contracts.opening_squad import (
    OpeningSquadObjectivePolicy,
    RiskPenaltyPolicy,
    RoleWeightPolicy,
    build_default_opening_squad_objective_policy,
    evaluate_opening_squad_objective,
    reconcile_objective_evaluation,
)
from ml.contracts.run_metadata import (
    build_run_metadata,
    provenance_inputs_from_file_metadata,
)
from ml.decision.squad_rules import SquadLegalityEngine


OPTIMIZER_VERSION = "day101a_v1"
ARTIFACT_TYPE = "opening_squad_optimizer"
ARTIFACT_VERSION = "fpl_opening_squad_optimizer_v1"
RECOMMENDATION_STATUS = "preview_only"
EXPECTED_OBJECTIVE_MODES = {"gw1_only_fallback", "multi_gw"}
VALID_POSITIONS = ("GKP", "DEF", "MID", "FWD")
ROLE_NAMES = ("starter", "bench_gk", "bench_1", "bench_2", "bench_3")
TIE_BREAK_EPSILON = 1e-9


class OpeningSquadOptimizerError(RuntimeError):
    """Raised when Day101A inputs or optimization outputs are unsafe."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, float):
        if np.isnan(value):
            return False
        return value != 0.0
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", "", "none", "nan"}:
        return False
    raise OpeningSquadOptimizerError("Cannot parse boolean value: %r" % value)


def nullable_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(parsed):
        return None
    return parsed


def nullable_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(parsed):
        return None
    if abs(parsed - round(parsed)) > 1e-9:
        return None
    return int(round(parsed))


def parse_list_cell(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except (TypeError, ValueError):
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except (SyntaxError, ValueError):
        pass
    return [text]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(
    path: Path,
    *,
    artifact_type: Optional[str] = None,
    run_id: Optional[str] = None,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise OpeningSquadOptimizerError("Required source file does not exist: %s" % resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "exists": True,
        "artifact_type": artifact_type,
        "run_id": run_id,
        "version": version,
    }


def load_json(path: Path, label: str) -> Dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise OpeningSquadOptimizerError("%s does not exist: %s" % (label, resolved))
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OpeningSquadOptimizerError("Invalid JSON in %s: %s" % (resolved, exc))
    if not isinstance(payload, dict):
        raise OpeningSquadOptimizerError("%s must contain a JSON object." % label)
    return payload


def validate_source_report(report: Mapping[str, Any]) -> None:
    blockers: List[str] = []
    checks = [
        (report.get("passed") is True, "Day97A report must have passed=true."),
        (
            report.get("ready_for_day101a") is True,
            "Day97A report must have ready_for_day101a=true.",
        ),
        (
            report.get("stop_point_satisfied") is True,
            "Day97A report must have stop_point_satisfied=true.",
        ),
        (
            report.get("preview_only") is True,
            "Day97A source must remain preview_only=true.",
        ),
        (
            report.get("production_approved") is False,
            "Day97A source must not be production approved.",
        ),
        (
            report.get("writes_database") is False,
            "Day97A source must have writes_database=false.",
        ),
        (
            report.get("writes_predictions_table") is False,
            "Day97A source must have writes_predictions_table=false.",
        ),
        (
            report.get("writes_recommendations") is False,
            "Day97A source must have writes_recommendations=false.",
        ),
        (
            report.get("writes_squad_state") is False,
            "Day97A source must have writes_squad_state=false.",
        ),
        (
            report.get("recommendation_status") == RECOMMENDATION_STATUS,
            "Day97A source must have recommendation_status=preview_only.",
        ),
        (
            report.get("objective_mode") in EXPECTED_OBJECTIVE_MODES,
            "Day97A objective_mode is unsupported: %r." % report.get("objective_mode"),
        ),
    ]
    for valid, message in checks:
        if not valid:
            blockers.append(message)

    effective_horizon = nullable_int(report.get("effective_horizon"))
    requested_horizon = nullable_int(report.get("requested_horizon"))
    start_gw = nullable_int(report.get("start_gw"))
    if effective_horizon is None or effective_horizon < 1:
        blockers.append("Day97A effective_horizon must be a positive integer.")
    if requested_horizon is None or requested_horizon < 1:
        blockers.append("Day97A requested_horizon must be a positive integer.")
    if start_gw != 1:
        blockers.append("Day101A Fast Lane requires start_gw=1.")
    if report.get("objective_mode") == "gw1_only_fallback" and effective_horizon != 1:
        blockers.append("gw1_only_fallback requires effective_horizon=1.")
    if report.get("objective_mode") == "multi_gw" and effective_horizon != requested_horizon:
        blockers.append("multi_gw mode requires the full requested horizon to be effective.")
    if blockers:
        raise OpeningSquadOptimizerError("Unsafe Day97A report: %s" % " | ".join(blockers))


def resolve_day97_artifact(report: Mapping[str, Any], artifact_name: str) -> Path:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise OpeningSquadOptimizerError("Day97A report is missing artifacts metadata.")
    root = artifacts.get("root")
    keys = artifacts.get("keys")
    if not root or not isinstance(keys, Mapping):
        raise OpeningSquadOptimizerError("Day97A artifacts metadata is incomplete.")
    key = keys.get(artifact_name)
    if not key:
        raise OpeningSquadOptimizerError(
            "Day97A report is missing artifact key %s." % artifact_name
        )
    path = Path(str(root)).expanduser().resolve() / Path(str(key))
    if not path.is_file():
        raise OpeningSquadOptimizerError(
            "Day97A artifact %s does not exist: %s" % (artifact_name, path)
        )
    return path


def normalize_projection_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
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
        "manual_review_required",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise OpeningSquadOptimizerError(
            "Day97A optimizer projection rows are missing columns: %s" % missing
        )
    normalized = frame.copy()
    for column in ("player_id", "target_gw", "now_cost", "fallback_level"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in (
        "predicted_points",
        "expected_minutes",
        "start_probability",
        "appearance_probability",
        "uncertainty_lower",
        "uncertainty_upper",
    ):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in (
        "has_fixture",
        "fallback_used",
        "production_ready",
        "selection_eligible",
        "manual_review_required",
    ):
        normalized[column] = normalized[column].apply(bool_value)
    normalized["risk_flags"] = normalized["risk_flags"].apply(parse_list_cell)
    normalized["position"] = normalized["position"].astype(str).str.upper().str.strip()

    if normalized["player_id"].isna().any():
        raise OpeningSquadOptimizerError("Optimizer source contains missing player_id values.")
    if normalized["target_gw"].isna().any():
        raise OpeningSquadOptimizerError("Optimizer source contains missing target_gw values.")
    if normalized["predicted_points"].isna().any():
        raise OpeningSquadOptimizerError("Optimizer source contains missing predicted_points.")
    if normalized["expected_minutes"].isna().any():
        raise OpeningSquadOptimizerError("Optimizer source contains missing expected_minutes.")
    if normalized["start_probability"].isna().any():
        raise OpeningSquadOptimizerError("Optimizer source contains missing start_probability.")
    if normalized["appearance_probability"].isna().any():
        raise OpeningSquadOptimizerError("Optimizer source contains missing appearance_probability.")
    if normalized["now_cost"].isna().any():
        raise OpeningSquadOptimizerError("Optimizer source contains missing now_cost values.")
    if not normalized["selection_eligible"].all():
        raise OpeningSquadOptimizerError("Optimizer source contains selection-ineligible rows.")
    if normalized["production_ready"].any():
        raise OpeningSquadOptimizerError("Optimizer source contains production_ready=true rows.")
    invalid_positions = sorted(set(normalized["position"]) - set(VALID_POSITIONS))
    if invalid_positions:
        raise OpeningSquadOptimizerError(
            "Optimizer source contains invalid positions: %s" % invalid_positions
        )
    duplicates = normalized.duplicated(["player_id", "target_gw"], keep=False)
    if duplicates.any():
        raise OpeningSquadOptimizerError("Optimizer source contains duplicate player-GW rows.")

    normalized["player_id"] = normalized["player_id"].astype(int)
    normalized["target_gw"] = normalized["target_gw"].astype(int)
    normalized["now_cost"] = normalized["now_cost"].astype(int)
    normalized["fallback_level"] = normalized["fallback_level"].fillna(0).astype(int)
    return normalized.sort_values(["player_id", "target_gw"]).reset_index(drop=True)


def normalize_long_horizon(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "player_id",
        "target_gw",
        "player_name",
        "web_name",
        "team_id",
        "team_name",
        "team_short_name",
        "position",
        "now_cost",
        "selection_eligible",
        "prediction_available",
        "row_status",
        "fixture_eligibility_reason",
        "manual_review_required",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise OpeningSquadOptimizerError(
            "Day97A long horizon is missing metadata columns: %s" % missing
        )
    normalized = frame.copy()
    normalized["player_id"] = pd.to_numeric(normalized["player_id"], errors="coerce")
    normalized["target_gw"] = pd.to_numeric(normalized["target_gw"], errors="coerce")
    normalized["team_id"] = pd.to_numeric(normalized["team_id"], errors="coerce")
    normalized["now_cost"] = pd.to_numeric(normalized["now_cost"], errors="coerce")
    if normalized[["player_id", "target_gw", "team_id", "now_cost"]].isna().any().any():
        raise OpeningSquadOptimizerError("Day97A long horizon has invalid identity/price metadata.")
    for column in ("selection_eligible", "prediction_available", "manual_review_required"):
        normalized[column] = normalized[column].apply(bool_value)
    normalized["player_id"] = normalized["player_id"].astype(int)
    normalized["target_gw"] = normalized["target_gw"].astype(int)
    normalized["team_id"] = normalized["team_id"].astype(int)
    normalized["now_cost"] = normalized["now_cost"].astype(int)
    normalized["position"] = normalized["position"].astype(str).str.upper().str.strip()
    return normalized


def projection_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            {
                "player_id": int(row["player_id"]),
                "target_gw": int(row["target_gw"]),
                "predicted_points": float(row["predicted_points"]),
                "expected_minutes": float(row["expected_minutes"]),
                "start_probability": float(row["start_probability"]),
                "appearance_probability": float(row["appearance_probability"]),
                "has_fixture": bool(row["has_fixture"]),
                "fallback_used": bool(row["fallback_used"]),
                "fallback_level": int(row["fallback_level"]),
                "uncertainty_lower": nullable_float(row.get("uncertainty_lower")),
                "uncertainty_upper": nullable_float(row.get("uncertainty_upper")),
                "now_cost": int(row["now_cost"]),
                "position": str(row["position"]),
                "risk_flags": list(row.get("risk_flags") or []),
                "readiness_status": row.get("readiness_status"),
                "production_ready": bool(row.get("production_ready")),
                "selection_eligible": bool(row.get("selection_eligible")),
            }
        )
    return records


def build_player_metadata(
    long_frame: pd.DataFrame,
    projection_frame: pd.DataFrame,
    start_gw: int,
) -> pd.DataFrame:
    current = long_frame[long_frame["target_gw"] == start_gw].copy()
    if current["player_id"].duplicated().any():
        raise OpeningSquadOptimizerError("Day97A long horizon has duplicate GW1 player rows.")
    current = current.set_index("player_id", drop=False)
    projection_players = sorted(projection_frame["player_id"].unique().tolist())
    missing = [player_id for player_id in projection_players if player_id not in current.index]
    if missing:
        raise OpeningSquadOptimizerError(
            "Optimizer players are missing from Day97A long-horizon metadata: %s" % missing[:20]
        )
    rows: List[Dict[str, Any]] = []
    for player_id in projection_players:
        row = current.loc[player_id]
        projection = projection_frame[projection_frame["player_id"] == player_id].iloc[0]
        if str(row["position"]) != str(projection["position"]):
            raise OpeningSquadOptimizerError("Position mismatch for player_id=%s." % player_id)
        if int(row["now_cost"]) != int(projection["now_cost"]):
            raise OpeningSquadOptimizerError("Price mismatch for player_id=%s." % player_id)
        rows.append(
            {
                "player_id": int(player_id),
                "fpl_player_id": nullable_int(row.get("fpl_player_id")),
                "player_name": str(row.get("player_name") or ""),
                "web_name": str(row.get("web_name") or ""),
                "team_id": int(row["team_id"]),
                "team_name": str(row.get("team_name") or ""),
                "team_short_name": str(row.get("team_short_name") or ""),
                "position": str(row["position"]),
                "now_cost": int(row["now_cost"]),
                "selection_eligible": True,
                "eligibility_reason": str(row.get("fixture_eligibility_reason") or ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["position", "now_cost", "player_id"]).reset_index(drop=True)


def build_candidate_audit(
    long_frame: pd.DataFrame,
    projection_frame: pd.DataFrame,
    start_gw: int,
) -> pd.DataFrame:
    current = long_frame[long_frame["target_gw"] == start_gw].copy()
    candidate_ids = set(int(value) for value in projection_frame["player_id"].unique())
    rows: List[Dict[str, Any]] = []
    for row in current.sort_values("player_id").to_dict(orient="records"):
        player_id = int(row["player_id"])
        reasons: List[str] = []
        if not bool(row["selection_eligible"]):
            reasons.append("selection_ineligible")
        if not bool(row["prediction_available"]):
            reasons.append("gw1_prediction_unavailable")
        if player_id not in candidate_ids and not reasons:
            reasons.append("excluded_from_day97a_optimizer_source")
        fixture_reason = str(row.get("fixture_eligibility_reason") or "").strip()
        if not bool(row["selection_eligible"]) and fixture_reason:
            reasons.append("eligibility:%s" % fixture_reason)
        rows.append(
            {
                "player_id": player_id,
                "fpl_player_id": nullable_int(row.get("fpl_player_id")),
                "player_name": str(row.get("player_name") or ""),
                "web_name": str(row.get("web_name") or ""),
                "team_id": int(row["team_id"]),
                "team_name": str(row.get("team_name") or ""),
                "team_short_name": str(row.get("team_short_name") or ""),
                "position": str(row["position"]),
                "now_cost": int(row["now_cost"]),
                "selection_eligible": bool(row["selection_eligible"]),
                "prediction_available": bool(row["prediction_available"]),
                "row_status": str(row.get("row_status") or ""),
                "manual_review_required": bool(row.get("manual_review_required")),
                "candidate_eligible": player_id in candidate_ids,
                "exclusion_reasons": json.dumps(reasons, separators=(",", ":")),
            }
        )
    audit = pd.DataFrame(rows)
    if audit["player_id"].duplicated().any():
        raise OpeningSquadOptimizerError("Candidate audit has duplicate player IDs.")
    return audit


def base_player_utility(
    policy: OpeningSquadObjectivePolicy,
    row: Mapping[str, Any],
) -> float:
    points = float(row["predicted_points"])
    expected_minutes = float(row["expected_minutes"])
    start_probability = float(row["start_probability"])
    minutes_shortfall = max(0.0, policy.risk_penalties.expected_minutes_target - expected_minutes)
    minutes_penalty = (
        minutes_shortfall * policy.risk_penalties.minutes_shortfall_penalty_per_minute
    )
    start_shortfall = max(0.0, policy.risk_penalties.start_probability_target - start_probability)
    start_penalty = start_shortfall * policy.risk_penalties.start_probability_shortfall_penalty
    fallback_penalty = 0.0
    if bool(row["fallback_used"]):
        fallback_penalty = (
            policy.risk_penalties.fallback_used_penalty
            + int(row["fallback_level"]) * policy.risk_penalties.fallback_level_penalty
        )
    lower = nullable_float(row.get("uncertainty_lower"))
    upper = nullable_float(row.get("uncertainty_upper"))
    if lower is None or upper is None:
        uncertainty_penalty = policy.risk_penalties.missing_uncertainty_penalty
    else:
        uncertainty_penalty = (
            (upper - lower) * policy.risk_penalties.uncertainty_width_penalty
        )
    value_bonus = (
        points / float(max(int(row["now_cost"]), 1)) * policy.value_bank.value_bonus_weight
    )
    return (
        points
        - minutes_penalty
        - start_penalty
        - fallback_penalty
        - uncertainty_penalty
        + value_bonus
    )


def player_availability_risk(
    policy: OpeningSquadObjectivePolicy,
    row: Mapping[str, Any],
) -> float:
    expected_minutes = float(row["expected_minutes"])
    start_probability = float(row["start_probability"])
    minutes_shortfall = max(
        0.0, policy.risk_penalties.expected_minutes_target - expected_minutes
    )
    minutes_penalty = (
        minutes_shortfall
        * policy.risk_penalties.minutes_shortfall_penalty_per_minute
    )
    start_shortfall = max(
        0.0, policy.risk_penalties.start_probability_target - start_probability
    )
    start_penalty = (
        start_shortfall
        * policy.risk_penalties.start_probability_shortfall_penalty
    )
    return minutes_penalty + start_penalty


def aggregate_availability_risk(
    policy: OpeningSquadObjectivePolicy,
    projection_frame: pd.DataFrame,
    effective_gameweeks: Sequence[int],
) -> Dict[int, float]:
    risks: Dict[int, float] = {}
    effective_set = set(int(gw) for gw in effective_gameweeks)
    relevant = projection_frame[projection_frame["target_gw"].isin(effective_set)]
    for player_id, group in relevant.groupby("player_id", sort=True):
        total = 0.0
        seen: Set[int] = set()
        for row in group.to_dict(orient="records"):
            gw = int(row["target_gw"])
            seen.add(gw)
            total += (
                player_availability_risk(policy, row)
                * float(policy.gameweek_discounts[gw])
            )
        missing = effective_set - seen
        if missing:
            raise OpeningSquadOptimizerError(
                "Player %s is missing availability-risk GWs %s."
                % (player_id, sorted(missing))
            )
        risks[int(player_id)] = total
    return risks


def aggregate_player_utilities(
    policy: OpeningSquadObjectivePolicy,
    projection_frame: pd.DataFrame,
    effective_gameweeks: Sequence[int],
) -> Dict[int, float]:
    utilities: Dict[int, float] = {}
    effective_set = set(int(gw) for gw in effective_gameweeks)
    relevant = projection_frame[projection_frame["target_gw"].isin(effective_set)]
    for player_id, group in relevant.groupby("player_id", sort=True):
        total = 0.0
        seen: Set[int] = set()
        for row in group.to_dict(orient="records"):
            gw = int(row["target_gw"])
            seen.add(gw)
            total += base_player_utility(policy, row) * float(policy.gameweek_discounts[gw])
        missing = effective_set - seen
        if missing:
            raise OpeningSquadOptimizerError(
                "Player %s is missing effective projection GWs %s." % (player_id, sorted(missing))
            )
        utilities[int(player_id)] = total
    return utilities


def build_variant_policies(target_season: str) -> Dict[str, OpeningSquadObjectivePolicy]:
    primary = build_default_opening_squad_objective_policy(
        target_season=target_season,
        horizon_mode="gw1_gw5",
    )
    alternative_a = replace(
        primary,
        policy_version="fast_lane_2026_27_day101a_alt_a_v1",
        role_weights=RoleWeightPolicy(
            starter_weight=1.0,
            captain_bonus_weight=1.0,
            vice_captain_bonus_weight=0.0,
            vice_captain_treatment="contingency_only_no_deterministic_bonus",
            bench_goalkeeper_weight=0.02,
            bench_outfield_weights=(0.06, 0.03, 0.01),
        ),
    )
    primary_risk = primary.risk_penalties
    alternative_b = replace(
        primary,
        policy_version="fast_lane_2026_27_day101a_alt_b_v1",
        risk_penalties=RiskPenaltyPolicy(
            expected_minutes_target=max(75.0, primary_risk.expected_minutes_target),
            minutes_shortfall_penalty_per_minute=max(
                0.015, primary_risk.minutes_shortfall_penalty_per_minute * 1.5
            ),
            start_probability_target=max(0.80, primary_risk.start_probability_target),
            start_probability_shortfall_penalty=max(
                1.25, primary_risk.start_probability_shortfall_penalty * 1.5
            ),
            fallback_used_penalty=max(0.50, primary_risk.fallback_used_penalty * 1.4),
            fallback_level_penalty=max(0.20, primary_risk.fallback_level_penalty * 1.3),
            uncertainty_width_penalty=max(
                0.10, primary_risk.uncertainty_width_penalty * 1.25
            ),
            missing_uncertainty_penalty=max(
                0.65, primary_risk.missing_uncertainty_penalty * 1.3
            ),
        ),
    )
    return {
        "primary": primary,
        "alternative_a": alternative_a,
        "alternative_b": alternative_b,
    }


def minimum_raw_bench_cost(metadata: pd.DataFrame) -> int:
    goalkeepers = metadata[metadata["position"] == "GKP"].sort_values(
        ["now_cost", "player_id"]
    )
    outfield = metadata[metadata["position"] != "GKP"].sort_values(
        ["now_cost", "player_id"]
    )
    if goalkeepers.empty or len(outfield) < 3:
        raise OpeningSquadOptimizerError("Candidate pool cannot form a four-player bench.")
    return int(goalkeepers.iloc[0]["now_cost"] + outfield.head(3)["now_cost"].sum())


def selected_expression_indices(indexer: Mapping[Tuple[str, int], int], player_index: int) -> List[int]:
    return [indexer[(role, player_index)] for role in ROLE_NAMES]


def solve_variant(
    *,
    variant: str,
    policy: OpeningSquadObjectivePolicy,
    projection_frame: pd.DataFrame,
    metadata: pd.DataFrame,
    rules: Any,
    effective_gameweeks: Sequence[int],
    bench_cost_cap: Optional[int] = None,
    availability_risk_cap: Optional[float] = None,
    overlap_caps: Optional[Sequence[Tuple[Set[int], int, str]]] = None,
) -> Dict[str, Any]:
    if policy.role_weights.vice_captain_bonus_weight != 0.0:
        raise OpeningSquadOptimizerError(
            "Day101A provisional optimizer requires zero deterministic vice-captain bonus."
        )
    candidates = metadata.sort_values("player_id").reset_index(drop=True)
    player_ids = [int(value) for value in candidates["player_id"].tolist()]
    n_players = len(player_ids)
    if n_players < int(rules.squad["size"]):
        raise OpeningSquadOptimizerError("Candidate pool is smaller than the required squad size.")

    utilities = aggregate_player_utilities(policy, projection_frame, effective_gameweeks)
    availability_risks = aggregate_availability_risk(
        policy, projection_frame, effective_gameweeks
    )
    missing_utility = [player_id for player_id in player_ids if player_id not in utilities]
    if missing_utility:
        raise OpeningSquadOptimizerError(
            "Missing objective utility for players: %s" % missing_utility[:20]
        )

    variable_names: List[Tuple[str, int]] = []
    indexer: Dict[Tuple[str, int], int] = {}
    for role in ROLE_NAMES + ("captain",):
        for i in range(n_players):
            indexer[(role, i)] = len(variable_names)
            variable_names.append((role, i))
    n_variables = len(variable_names)

    c = np.zeros(n_variables, dtype=float)
    lower_bounds = np.zeros(n_variables, dtype=float)
    upper_bounds = np.ones(n_variables, dtype=float)
    sorted_ids = sorted(player_ids)
    rank_by_id = {player_id: rank for rank, player_id in enumerate(sorted_ids)}

    role_weights = {
        "starter": policy.role_weights.starter_weight,
        "bench_gk": policy.role_weights.bench_goalkeeper_weight,
        "bench_1": policy.role_weights.bench_outfield_weights[0],
        "bench_2": policy.role_weights.bench_outfield_weights[1],
        "bench_3": policy.role_weights.bench_outfield_weights[2],
    }
    for i, row in candidates.iterrows():
        player_id = int(row["player_id"])
        utility = float(utilities[player_id])
        price_units = int(row["now_cost"])
        deterministic_rank = float(rank_by_id[player_id] + 1) / float(max(n_players, 1))
        for role in ROLE_NAMES:
            idx = indexer[(role, i)]
            role_utility = utility * float(role_weights[role])
            # Bank bonus is linear because bank = budget - selected cost.
            role_utility -= price_units * float(policy.value_bank.bank_bonus_per_unit)
            c[idx] = -role_utility + TIE_BREAK_EPSILON * deterministic_rank
        cap_idx = indexer[("captain", i)]
        c[cap_idx] = (
            -utility * float(policy.role_weights.captain_bonus_weight)
            + TIE_BREAK_EPSILON * deterministic_rank
        )
        if str(row["position"]) != "GKP":
            upper_bounds[indexer[("bench_gk", i)]] = 0.0
        else:
            for role in ("bench_1", "bench_2", "bench_3"):
                upper_bounds[indexer[(role, i)]] = 0.0

    matrix_rows: List[int] = []
    matrix_cols: List[int] = []
    matrix_values: List[float] = []
    constraint_lbs: List[float] = []
    constraint_ubs: List[float] = []
    constraint_names: List[str] = []

    def add_constraint(
        coefficients: Mapping[int, float],
        lb: float,
        ub: float,
        name: str,
    ) -> None:
        row_index = len(constraint_lbs)
        for col, value in coefficients.items():
            if abs(float(value)) <= 0.0:
                continue
            matrix_rows.append(row_index)
            matrix_cols.append(int(col))
            matrix_values.append(float(value))
        constraint_lbs.append(float(lb))
        constraint_ubs.append(float(ub))
        constraint_names.append(name)

    for i in range(n_players):
        assignment = {indexer[(role, i)]: 1.0 for role in ROLE_NAMES}
        add_constraint(assignment, 0.0, 1.0, "one_squad_role_per_player[%s]" % player_ids[i])
        add_constraint(
            {
                indexer[("captain", i)]: 1.0,
                indexer[("starter", i)]: -1.0,
            },
            -np.inf,
            0.0,
            "captain_requires_starter[%s]" % player_ids[i],
        )

    add_constraint(
        {indexer[("starter", i)]: 1.0 for i in range(n_players)},
        float(rules.lineup["starting_size"]),
        float(rules.lineup["starting_size"]),
        "starting_size",
    )
    for role in ("bench_gk", "bench_1", "bench_2", "bench_3"):
        add_constraint(
            {indexer[(role, i)]: 1.0 for i in range(n_players)},
            1.0,
            1.0,
            "%s_count" % role,
        )
    add_constraint(
        {indexer[("captain", i)]: 1.0 for i in range(n_players)},
        1.0,
        1.0,
        "captain_count",
    )

    for position in VALID_POSITIONS:
        position_indices = [
            i for i, row in candidates.iterrows() if str(row["position"]) == position
        ]
        coefficients: Dict[int, float] = {}
        for i in position_indices:
            for role in ROLE_NAMES:
                coefficients[indexer[(role, i)]] = 1.0
        quota = int(rules.position_quotas[position])
        add_constraint(coefficients, float(quota), float(quota), "squad_position_%s" % position)

        starter_coefficients = {
            indexer[("starter", i)]: 1.0 for i in position_indices
        }
        bounds = rules.lineup["position_bounds"][position]
        add_constraint(
            starter_coefficients,
            float(bounds["min"]),
            float(bounds["max"]),
            "starter_position_%s" % position,
        )

    budget_coefficients: Dict[int, float] = {}
    for i, row in candidates.iterrows():
        for role in ROLE_NAMES:
            budget_coefficients[indexer[(role, i)]] = float(int(row["now_cost"]))
    add_constraint(
        budget_coefficients,
        -np.inf,
        float(rules.initial_budget_units),
        "budget_limit",
    )

    max_per_club = int(rules.squad["max_players_per_club"])
    for club_id, group in candidates.groupby("team_id", sort=True):
        coefficients = {}
        for i in group.index.tolist():
            for role in ROLE_NAMES:
                coefficients[indexer[(role, i)]] = 1.0
        add_constraint(
            coefficients,
            -np.inf,
            float(max_per_club),
            "club_limit[%s]" % club_id,
        )

    if bench_cost_cap is not None:
        coefficients = {}
        for i, row in candidates.iterrows():
            for role in ("bench_gk", "bench_1", "bench_2", "bench_3"):
                coefficients[indexer[(role, i)]] = float(int(row["now_cost"]))
        add_constraint(
            coefficients,
            -np.inf,
            float(bench_cost_cap),
            "bench_cost_cap",
        )

    if availability_risk_cap is not None:
        coefficients = {}
        for i, player_id in enumerate(player_ids):
            risk = float(availability_risks[player_id])
            for role in ROLE_NAMES:
                coefficients[indexer[(role, i)]] = risk
        add_constraint(
            coefficients,
            -np.inf,
            float(availability_risk_cap),
            "availability_risk_cap",
        )

    for selected_ids, max_overlap, label in overlap_caps or []:
        coefficients = {}
        for i, player_id in enumerate(player_ids):
            if player_id in selected_ids:
                for role in ROLE_NAMES:
                    coefficients[indexer[(role, i)]] = 1.0
        add_constraint(
            coefficients,
            -np.inf,
            float(max_overlap),
            "overlap_cap[%s]" % label,
        )

    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_cols)),
        shape=(len(constraint_lbs), n_variables),
        dtype=float,
    ).tocsr()
    constraints = LinearConstraint(
        matrix,
        np.asarray(constraint_lbs, dtype=float),
        np.asarray(constraint_ubs, dtype=float),
    )
    result = milp(
        c=c,
        integrality=np.ones(n_variables, dtype=int),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if result.status != 0 or result.x is None:
        raise OpeningSquadOptimizerError(
            "MILP failed for %s: status=%s message=%s"
            % (variant, result.status, result.message)
        )

    solution = np.asarray(result.x)
    selected_by_role: Dict[str, List[int]] = {role: [] for role in ROLE_NAMES}
    for role in ROLE_NAMES:
        for i, player_id in enumerate(player_ids):
            if solution[indexer[(role, i)]] >= 0.5:
                selected_by_role[role].append(player_id)
    captain_ids = [
        player_ids[i]
        for i in range(n_players)
        if solution[indexer[("captain", i)]] >= 0.5
    ]
    if len(captain_ids) != 1:
        raise OpeningSquadOptimizerError("MILP did not select exactly one provisional captain.")
    captain_id = captain_ids[0]
    starters = list(selected_by_role["starter"])
    bench_order = (
        list(selected_by_role["bench_gk"])
        + list(selected_by_role["bench_1"])
        + list(selected_by_role["bench_2"])
        + list(selected_by_role["bench_3"])
    )
    if len(starters) != int(rules.lineup["starting_size"]) or len(bench_order) != int(
        rules.lineup["bench_size"]
    ):
        raise OpeningSquadOptimizerError("MILP returned an invalid provisional role partition.")

    utility_order = sorted(
        [player_id for player_id in starters if player_id != captain_id],
        key=lambda player_id: (-utilities[player_id], player_id),
    )
    if not utility_order:
        raise OpeningSquadOptimizerError("No provisional vice-captain candidate exists.")
    vice_captain_id = utility_order[0]

    selected_ids = starters + bench_order
    selected_set = set(selected_ids)
    if len(selected_set) != int(rules.squad["size"]):
        raise OpeningSquadOptimizerError("MILP did not select exactly 15 unique players.")
    selected_metadata = candidates[candidates["player_id"].isin(selected_set)].copy()
    total_cost = int(selected_metadata["now_cost"].sum())
    bank_units = int(rules.initial_budget_units - total_cost)
    bench_cost = int(
        candidates[candidates["player_id"].isin(set(bench_order))]["now_cost"].sum()
    )
    availability_risk_score = float(
        sum(availability_risks[player_id] for player_id in selected_set)
    )
    if (
        availability_risk_cap is not None
        and availability_risk_score > float(availability_risk_cap) + 1e-7
    ):
        raise OpeningSquadOptimizerError(
            "Solved squad violates availability-risk cap: score=%s cap=%s."
            % (availability_risk_score, availability_risk_cap)
        )

    return {
        "variant": variant,
        "selected_player_ids": sorted(selected_set),
        "objective_evaluation_plan": {
            "starting_player_ids": starters,
            "bench_order": bench_order,
            "captain_player_id": captain_id,
            "vice_captain_player_id": vice_captain_id,
            "is_final_day101b_decision": False,
            "purpose": "Day100B objective evaluation only; Day101B must re-optimize lineup and captaincy.",
        },
        "total_cost_units": total_cost,
        "bank_units": bank_units,
        "bench_cost_units": bench_cost,
        "availability_risk_score": availability_risk_score,
        "solver": {
            "name": "scipy.optimize.milp_highs",
            "status": int(result.status),
            "message": str(result.message),
            "fun_with_deterministic_epsilon": float(result.fun),
            "mip_gap": nullable_float(getattr(result, "mip_gap", None)),
            "mip_node_count": nullable_int(getattr(result, "mip_node_count", None)),
            "constraint_count": len(constraint_names),
            "variable_count": n_variables,
            "constraint_names": constraint_names,
            "tie_break_epsilon": TIE_BREAK_EPSILON,
        },
        "controlled_constraints": {
            "bench_cost_cap_units": bench_cost_cap,
            "availability_risk_cap": availability_risk_cap,
            "overlap_caps": [
                {
                    "label": label,
                    "max_overlap": max_overlap,
                    "reference_player_count": len(selected_ids_ref),
                }
                for selected_ids_ref, max_overlap, label in (overlap_caps or [])
            ],
        },
    }


def legality_player_rows(metadata: pd.DataFrame, selected_ids: Set[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in metadata[metadata["player_id"].isin(selected_ids)].to_dict(orient="records"):
        rows.append(
            {
                "player_id": int(row["player_id"]),
                "player_name": str(row.get("player_name") or row.get("web_name") or ""),
                "position": str(row["position"]),
                "club_id": int(row["team_id"]),
                "price_units": int(row["now_cost"]),
                "selection_eligible": True,
                "eligibility_reason": str(row.get("eligibility_reason") or "eligible_day97a_optimizer_source"),
            }
        )
    return rows


def validate_and_evaluate_variant(
    *,
    variant_result: Dict[str, Any],
    policy: OpeningSquadObjectivePolicy,
    projection_records_all: Sequence[Mapping[str, Any]],
    metadata: pd.DataFrame,
    engine: SquadLegalityEngine,
) -> Dict[str, Any]:
    selected_set = set(int(value) for value in variant_result["selected_player_ids"])
    players = legality_player_rows(metadata, selected_set)
    squad_validation = engine.validate_squad(
        players,
        declared_bank_units=int(variant_result["bank_units"]),
        require_declared_bank=True,
    )
    if not squad_validation["valid"]:
        raise OpeningSquadOptimizerError(
            "Day100A rejected %s squad: %s"
            % (variant_result["variant"], squad_validation["issues"])
        )

    plan = variant_result["objective_evaluation_plan"]
    plan_validation = engine.validate_plan(
        players,
        starting_player_ids=plan["starting_player_ids"],
        bench_order=plan["bench_order"],
        captain_player_id=plan["captain_player_id"],
        vice_captain_player_id=plan["vice_captain_player_id"],
        declared_bank_units=int(variant_result["bank_units"]),
    )
    if not plan_validation["valid"]:
        raise OpeningSquadOptimizerError(
            "Day100A rejected %s provisional objective plan: %s"
            % (variant_result["variant"], plan_validation["issues"])
        )

    evaluation = evaluate_opening_squad_objective(
        policy,
        projection_records_all,
        starting_player_ids=plan["starting_player_ids"],
        bench_order=plan["bench_order"],
        captain_player_id=plan["captain_player_id"],
        vice_captain_player_id=plan["vice_captain_player_id"],
        bank_units=int(variant_result["bank_units"]),
    )
    reconciliation = reconcile_objective_evaluation(evaluation)
    if not reconciliation["passed"]:
        raise OpeningSquadOptimizerError(
            "Day100B objective reconciliation failed for %s." % variant_result["variant"]
        )
    variant_result["squad_legality"] = squad_validation
    variant_result["provisional_plan_legality"] = plan_validation
    variant_result["objective_evaluation"] = evaluation
    variant_result["objective_reconciliation"] = reconciliation
    return variant_result


def solve_with_relaxations(
    *,
    variant: str,
    policy: OpeningSquadObjectivePolicy,
    projection_frame: pd.DataFrame,
    metadata: pd.DataFrame,
    rules: Any,
    effective_gameweeks: Sequence[int],
    primary_ids: Optional[Set[int]] = None,
    alternative_a_ids: Optional[Set[int]] = None,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    if variant == "primary":
        result = solve_variant(
            variant=variant,
            policy=policy,
            projection_frame=projection_frame,
            metadata=metadata,
            rules=rules,
            effective_gameweeks=effective_gameweeks,
        )
        result["relaxation_attempts"] = attempts
        return result

    if primary_ids is None:
        raise OpeningSquadOptimizerError("Alternative variants require the primary squad.")

    if variant == "alternative_a":
        minimum_bench = minimum_raw_bench_cost(metadata)
        bench_caps: List[Optional[int]] = [
            minimum_bench + 10,
            minimum_bench + 20,
            minimum_bench + 30,
            None,
        ]
        overlap_caps = [13, 14]
        for overlap in overlap_caps:
            for bench_cap in bench_caps:
                try:
                    result = solve_variant(
                        variant=variant,
                        policy=policy,
                        projection_frame=projection_frame,
                        metadata=metadata,
                        rules=rules,
                        effective_gameweeks=effective_gameweeks,
                        bench_cost_cap=bench_cap,
                        overlap_caps=[(primary_ids, overlap, "primary")],
                    )
                    attempts.append(
                        {
                            "bench_cost_cap_units": bench_cap,
                            "primary_overlap_cap": overlap,
                            "status": "solved",
                        }
                    )
                    result["relaxation_attempts"] = attempts
                    result["alternative_design"] = {
                        "name": "stronger_starting_xi_cheaper_bench",
                        "minimum_raw_bench_cost_units": minimum_bench,
                        "description": (
                            "Lower bench weights plus a controlled bench-cost cap shift utility "
                            "toward the provisional starting XI."
                        ),
                    }
                    return result
                except OpeningSquadOptimizerError as exc:
                    attempts.append(
                        {
                            "bench_cost_cap_units": bench_cap,
                            "primary_overlap_cap": overlap,
                            "status": "infeasible",
                            "reason": str(exc),
                        }
                    )
        raise OpeningSquadOptimizerError("Alternative A remained infeasible after controlled relaxations.")

    if variant == "alternative_b":
        availability_risks = aggregate_availability_risk(
            policy, projection_frame, effective_gameweeks
        )
        primary_reference_risk = float(
            sum(availability_risks[player_id] for player_id in primary_ids)
        )
        if primary_reference_risk <= 0.0:
            raise OpeningSquadOptimizerError(
                "Alternative B requires a positive Primary availability-risk reference."
            )
        improvement_fractions = [0.15, 0.10, 0.05, 0.01]
        overlap_pairs = [(13, 13), (13, 14), (14, 14)]
        for improvement_fraction in improvement_fractions:
            risk_cap = primary_reference_risk * (1.0 - improvement_fraction)
            for primary_overlap, alt_a_overlap in overlap_pairs:
                overlap_constraints: List[Tuple[Set[int], int, str]] = [
                    (primary_ids, primary_overlap, "primary")
                ]
                if alternative_a_ids is not None:
                    overlap_constraints.append(
                        (alternative_a_ids, alt_a_overlap, "alternative_a")
                    )
                try:
                    result = solve_variant(
                        variant=variant,
                        policy=policy,
                        projection_frame=projection_frame,
                        metadata=metadata,
                        rules=rules,
                        effective_gameweeks=effective_gameweeks,
                        availability_risk_cap=risk_cap,
                        overlap_caps=overlap_constraints,
                    )
                    selected_risk = float(result["availability_risk_score"])
                    actual_improvement = (
                        primary_reference_risk - selected_risk
                    ) / primary_reference_risk
                    attempts.append(
                        {
                            "required_availability_risk_improvement_fraction": improvement_fraction,
                            "availability_risk_cap": risk_cap,
                            "primary_overlap_cap": primary_overlap,
                            "alternative_a_overlap_cap": alt_a_overlap,
                            "status": "solved",
                        }
                    )
                    result["relaxation_attempts"] = attempts
                    result["alternative_design"] = {
                        "name": "lower_risk_minutes_and_availability",
                        "primary_availability_risk_score": primary_reference_risk,
                        "selected_availability_risk_score": selected_risk,
                        "required_availability_risk_improvement_fraction": improvement_fraction,
                        "actual_availability_risk_improvement_fraction": actual_improvement,
                        "availability_risk_definition": (
                            "Unweighted Day100B expected-minutes shortfall penalty plus "
                            "start-probability shortfall penalty across the effective horizon."
                        ),
                        "description": (
                            "Uses stronger Day100B risk penalties and additionally requires "
                            "a measurable reduction in the Primary squad's minutes/start "
                            "availability-risk score while preserving hard Day97A eligibility."
                        ),
                    }
                    return result
                except OpeningSquadOptimizerError as exc:
                    attempts.append(
                        {
                            "required_availability_risk_improvement_fraction": improvement_fraction,
                            "availability_risk_cap": risk_cap,
                            "primary_overlap_cap": primary_overlap,
                            "alternative_a_overlap_cap": alt_a_overlap,
                            "status": "infeasible",
                            "reason": str(exc),
                        }
                    )
        raise OpeningSquadOptimizerError(
            "Alternative B remained infeasible after controlled availability-risk relaxations."
        )

    raise OpeningSquadOptimizerError("Unknown variant: %s" % variant)


def selected_squad_frame(
    variants: Mapping[str, Mapping[str, Any]],
    metadata: pd.DataFrame,
    projection_frame: pd.DataFrame,
) -> pd.DataFrame:
    projection_gw1 = projection_frame[projection_frame["target_gw"] == 1].set_index("player_id")
    metadata_by_id = metadata.set_index("player_id")
    rows: List[Dict[str, Any]] = []
    labels = {
        "primary": "balanced_risk_adjusted",
        "alternative_a": "stronger_starting_xi_cheaper_bench",
        "alternative_b": "lower_risk_minutes_and_availability",
    }
    for variant in ("primary", "alternative_a", "alternative_b"):
        result = variants[variant]
        plan = result["objective_evaluation_plan"]
        role_by_id: Dict[int, str] = {}
        for player_id in plan["starting_player_ids"]:
            role_by_id[int(player_id)] = "starter"
        for index, player_id in enumerate(plan["bench_order"]):
            role_by_id[int(player_id)] = "bench_gk" if index == 0 else "bench_%s" % index
        for player_id in result["selected_player_ids"]:
            player_id = int(player_id)
            meta = metadata_by_id.loc[player_id]
            pred = projection_gw1.loc[player_id]
            rows.append(
                {
                    "variant": variant,
                    "variant_label": labels[variant],
                    "player_id": player_id,
                    "fpl_player_id": meta.get("fpl_player_id"),
                    "player_name": meta.get("player_name"),
                    "web_name": meta.get("web_name"),
                    "team_id": int(meta["team_id"]),
                    "team_name": meta.get("team_name"),
                    "team_short_name": meta.get("team_short_name"),
                    "position": meta["position"],
                    "now_cost": int(meta["now_cost"]),
                    "gw1_predicted_points": float(pred["predicted_points"]),
                    "expected_minutes": float(pred["expected_minutes"]),
                    "start_probability": float(pred["start_probability"]),
                    "appearance_probability": float(pred["appearance_probability"]),
                    "fallback_used": bool(pred["fallback_used"]),
                    "fallback_level": int(pred["fallback_level"]),
                    "risk_flags": json.dumps(pred["risk_flags"], separators=(",", ":")),
                    "manual_review_required": bool(pred["manual_review_required"]),
                    "objective_evaluation_role": role_by_id[player_id],
                    "objective_evaluation_captain": player_id == int(plan["captain_player_id"]),
                    "objective_evaluation_vice_captain": player_id == int(plan["vice_captain_player_id"]),
                    "is_final_day101b_lineup_decision": False,
                    "variant_total_cost_units": int(result["total_cost_units"]),
                    "variant_bank_units": int(result["bank_units"]),
                    "variant_objective_value": float(
                        result["objective_evaluation"]["totals"]["objective_value"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["variant", "position", "now_cost", "player_id"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def objective_breakdown_frame(
    variants: Mapping[str, Mapping[str, Any]],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    metadata_by_id = metadata.set_index("player_id")
    rows: List[Dict[str, Any]] = []
    for variant in ("primary", "alternative_a", "alternative_b"):
        evaluation = variants[variant]["objective_evaluation"]
        for payload in evaluation["by_player"].values():
            player_id = int(payload["player_id"])
            meta = metadata_by_id.loc[player_id]
            totals = payload["totals"]
            rows.append(
                {
                    "variant": variant,
                    "player_id": player_id,
                    "player_name": meta.get("player_name"),
                    "web_name": meta.get("web_name"),
                    "team_id": int(meta["team_id"]),
                    "team_short_name": meta.get("team_short_name"),
                    "position": meta["position"],
                    "role": payload["role"],
                    "is_captain": bool(payload["is_captain"]),
                    "is_vice_captain": bool(payload["is_vice_captain"]),
                    "gross_expected_points": totals["gross_expected_points"],
                    "minutes_risk_penalty": totals["minutes_risk_penalty"],
                    "start_risk_penalty": totals["start_risk_penalty"],
                    "fallback_penalty": totals["fallback_penalty"],
                    "uncertainty_penalty": totals["uncertainty_penalty"],
                    "value_bonus": totals["value_bonus"],
                    "objective_value": totals["objective_value"],
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["variant", "objective_value", "player_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def update_candidate_selections(
    candidate_audit: pd.DataFrame,
    variants: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    updated = candidate_audit.copy()
    for variant in ("primary", "alternative_a", "alternative_b"):
        selected = set(int(value) for value in variants[variant]["selected_player_ids"])
        updated["selected_%s" % variant] = updated["player_id"].isin(selected)
    return updated


def variant_summary(result: Mapping[str, Any], policy: OpeningSquadObjectivePolicy) -> Dict[str, Any]:
    evaluation = result["objective_evaluation"]
    selected_ids = set(int(value) for value in result["selected_player_ids"])
    return {
        "variant": result["variant"],
        "selected_player_ids": sorted(selected_ids),
        "total_cost_units": int(result["total_cost_units"]),
        "bank_units": int(result["bank_units"]),
        "bench_cost_units": int(result["bench_cost_units"]),
        "availability_risk_score": float(result["availability_risk_score"]),
        "objective_value": evaluation["totals"]["objective_value"],
        "gross_expected_points": evaluation["totals"]["gross_expected_points"],
        "total_penalty": evaluation["totals"]["total_penalty"],
        "horizon_fallback_used": bool(evaluation["horizon_fallback_used"]),
        "effective_gameweeks": list(evaluation["effective_gameweeks"]),
        "manual_review_required": bool(evaluation["manual_review_required"]),
        "manual_review_reason_count": len(evaluation["manual_review_reasons"]),
        "objective_policy": policy.to_dict(),
        "objective_reconciliation": result["objective_reconciliation"],
        "squad_legality": {
            "valid": result["squad_legality"]["valid"],
            "issue_codes": result["squad_legality"]["issue_codes"],
        },
        "provisional_plan_legality": {
            "valid": result["provisional_plan_legality"]["valid"],
            "issue_codes": result["provisional_plan_legality"]["issue_codes"],
            "formation": result["provisional_plan_legality"]["lineup"]["formation"],
        },
        "objective_evaluation_plan": result["objective_evaluation_plan"],
        "solver": result["solver"],
        "controlled_constraints": result["controlled_constraints"],
        "relaxation_attempts": result.get("relaxation_attempts", []),
        "alternative_design": result.get("alternative_design"),
    }


def build_run_metadata_payload(
    *,
    report: Mapping[str, Any],
    target_season: str,
    created_at: str,
    source_metadata: Mapping[str, Mapping[str, Any]],
    rules: Any,
    primary_policy: OpeningSquadObjectivePolicy,
) -> Dict[str, Any]:
    source_run_metadata = report.get("run_metadata") or {}
    source_seasons = source_run_metadata.get("source_seasons") or ["2025_26"]
    parent_run_id = str(source_run_metadata.get("run_id") or "")
    rules_versions = dict((source_run_metadata.get("versions") or {}).get("rules_versions") or {})
    rules_versions["squad_transfer"] = rules.rules_version
    return build_run_metadata(
        run_id=None,
        run_type="optimization",
        artifact_type=ARTIFACT_TYPE,
        source_seasons=list(source_seasons),
        target_season=target_season,
        target_gw=1,
        horizon=int(report["effective_horizon"]),
        as_of_time=str(report["as_of_time_utc"]),
        prediction_mode="pre_gw1_prior",
        created_at=created_at,
        model_version=str(report.get("prediction_source") or "pre_gw1_heuristic_preview"),
        rules_versions=rules_versions,
        manifest_version=OPTIMIZER_VERSION,
        artifact_version=ARTIFACT_VERSION,
        additional_versions={
            "optimizer_version": OPTIMIZER_VERSION,
            "objective_contract_version": primary_policy.contract_version,
            "objective_policy_version": primary_policy.policy_version,
            "source_prediction_horizon_version": str(report.get("horizon_version") or "day97a_v1"),
            "objective_mode": str(report["objective_mode"]),
        },
        provenance={
            "producer": "ml.decision.optimize_opening_squad",
            "inputs": provenance_inputs_from_file_metadata(source_metadata),
            "parent_run_ids": [parent_run_id] if parent_run_id else [],
            "notes": [
                "Day101A Fast Lane opening-squad optimizer.",
                "Consumes only standardized Day97A prediction-horizon artifacts.",
                "Day100A validates all selected squads and provisional objective-evaluation plans.",
                "Day100B supplies the objective contract; provisional role assignments are not final Day101B decisions.",
                "All outputs remain preview_only with database and squad-state writes disabled.",
            ],
        },
    ).to_dict()


def artifact_definitions() -> Dict[str, Tuple[str, str]]:
    return {
        "opening_squad_csv": ("opening_squad", "csv"),
        "opening_squad_objective_breakdown_csv": (
            "opening_squad_objective_breakdown",
            "csv",
        ),
        "candidate_pool_csv": ("candidate_pool", "csv"),
        "run_metadata_json": ("run_metadata", "json"),
        "opening_squad_report_json": ("opening_squad_report", "json"),
        "opening_squad_report_md": ("opening_squad_report", "md"),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Day101A — Opening Squad Optimizer",
        "",
        "- Run ID: `%s`" % report["run_metadata"]["run_id"],
        "- Target season: `%s`" % report["target_season"],
        "- Objective mode: `%s`" % report["objective_mode"],
        "- Effective horizon: `%s`" % report["effective_horizon"],
        "- Preview only: `%s`" % report["preview_only"],
        "- Production approved: `%s`" % report["production_approved"],
        "- Writes database: `%s`" % report["writes_database"],
        "- Writes squad state: `%s`" % report["writes_squad_state"],
        "",
        "## Source audit",
        "",
        "```text",
        "source Day97A run: %s" % report["source_day97a_run_id"],
        "current players in horizon: %s" % report["candidate_audit"]["current_player_count"],
        "optimizer-safe candidates: %s" % report["candidate_audit"]["candidate_count"],
        "excluded players: %s" % report["candidate_audit"]["excluded_count"],
        "```",
        "",
        "## Variants",
        "",
    ]
    for variant in ("primary", "alternative_a", "alternative_b"):
        item = report["variants"][variant]
        lines.extend(
            [
                "### %s" % variant,
                "",
                "- Total cost: `%s`" % item["total_cost_units"],
                "- Bank: `%s`" % item["bank_units"],
                "- Objective value: `%s`" % item["objective_value"],
                "- Effective Gameweeks: `%s`" % item["effective_gameweeks"],
                "- Horizon fallback used: `%s`" % item["horizon_fallback_used"],
                "- Day100A squad legal: `%s`" % item["squad_legality"]["valid"],
                "- Objective reconciliation passed: `%s`"
                % item["objective_reconciliation"]["passed"],
                "",
            ]
        )
    lines.extend(
        [
            "## Important scope boundary",
            "",
            "> The XI, bench order, captain, and vice-captain stored in Day101A are provisional objective-evaluation assignments only. Day101B must re-optimize and decide the final GW1 lineup and captaincy plan.",
            "",
        ]
    )
    if report["objective_mode"] == "gw1_only_fallback":
        lines.extend(
            [
                "## GW1-only disclosure",
                "",
                "> Reliable GW2-GW5 player predictions are not available. All three Day101A squads are optimized on the explicit GW1-only fallback while retaining future fixture context for manual review.",
                "",
            ]
        )
    lines.extend(
        [
            "## Blockers",
            "",
        ]
    )
    if report["blockers"]:
        lines.extend("- %s" % item for item in report["blockers"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Stop point",
            "",
            "> A legal 15-player opening squad preview is reproducible from explicit versioned inputs.",
            "",
            "Stop point satisfied: `%s`" % report["stop_point_satisfied"],
            "Ready for Day101B: `%s`" % report["ready_for_day101b"],
            "",
        ]
    )
    return "\n".join(lines)


def write_immutable_outputs(
    *,
    artifact_root: Path,
    target_season: str,
    as_of_time: str,
    run_id: str,
    squad_frame: pd.DataFrame,
    objective_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    run_metadata: Mapping[str, Any],
    report: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    storage = LocalArtifactStorage(artifact_root)
    keys = {
        name: build_immutable_artifact_key(
            artifact_type=ARTIFACT_TYPE,
            season=target_season,
            target_gw=1,
            as_of_time=as_of_time,
            run_id=run_id,
            version=OPTIMIZER_VERSION,
            filename=filename,
            extension=extension,
        )
        for name, (filename, extension) in artifact_definitions().items()
    }
    payloads = {
        "opening_squad_csv": squad_frame.to_csv(index=False),
        "opening_squad_objective_breakdown_csv": objective_frame.to_csv(index=False),
        "candidate_pool_csv": candidate_frame.to_csv(index=False),
        "run_metadata_json": json.dumps(
            run_metadata, indent=2, sort_keys=True, ensure_ascii=False, default=str
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
        "opening_squad_report_json": json.dumps(
            report, indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        + "\n",
        "opening_squad_report_md": build_markdown_report(report),
    }
    for name, content in report_payloads.items():
        stored[name] = storage.write_immutable_text(keys[name], content).to_dict()
    return stored


def build_report(
    *,
    source_report: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    candidate_frame: pd.DataFrame,
    variants: Mapping[str, Mapping[str, Any]],
    policies: Mapping[str, OpeningSquadObjectivePolicy],
    source_metadata: Mapping[str, Mapping[str, Any]],
    rules_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    variant_summaries = {
        variant: variant_summary(variants[variant], policies[variant])
        for variant in ("primary", "alternative_a", "alternative_b")
    }
    selected_sets = [
        set(variant_summaries[variant]["selected_player_ids"])
        for variant in ("primary", "alternative_a", "alternative_b")
    ]
    variants_distinct = len({tuple(sorted(values)) for values in selected_sets}) == 3
    all_legal = all(
        summary["squad_legality"]["valid"]
        and summary["provisional_plan_legality"]["valid"]
        for summary in variant_summaries.values()
    )
    all_reconciled = all(
        summary["objective_reconciliation"]["passed"]
        for summary in variant_summaries.values()
    )
    blockers: List[str] = []
    if not variants_distinct:
        blockers.append("Primary and alternative squads are not all distinct.")
    if not all_legal:
        blockers.append("At least one Day101A squad failed Day100A legality.")
    if not all_reconciled:
        blockers.append("At least one Day100B objective evaluation failed reconciliation.")
    passed = not blockers
    warnings = [
        "All Day101A outputs are preview_only and not approved recommendations.",
        "Provisional XI/bench/captain assignments exist only to evaluate the Day100B objective; Day101B must make the final weekly decisions.",
    ]
    if source_report["objective_mode"] == "gw1_only_fallback":
        warnings.append(
            "Reliable GW2-GW5 player predictions are unavailable; all variants use the explicit GW1-only fallback."
        )
    return {
        "created_at_utc": utc_now(),
        "artifact_type": ARTIFACT_TYPE,
        "optimizer_version": OPTIMIZER_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "target_season": source_report["target_season"],
        "target_gw": 1,
        "requested_horizon": int(source_report["requested_horizon"]),
        "effective_horizon": int(source_report["effective_horizon"]),
        "objective_mode": source_report["objective_mode"],
        "future_fixture_context": source_report["future_fixture_context"],
        "as_of_time_utc": source_report["as_of_time_utc"],
        "recommendation_status": RECOMMENDATION_STATUS,
        "preview_only": True,
        "audit_only": True,
        "production_approved": False,
        "writes_database": False,
        "writes_predictions_table": False,
        "writes_recommendations": False,
        "writes_squad_state": False,
        "source_day97a_run_id": str((source_report.get("run_metadata") or {}).get("run_id") or ""),
        "run_metadata": dict(run_metadata),
        "source_artifacts": dict(source_metadata),
        "rules": dict(rules_metadata),
        "candidate_audit": {
            "current_player_count": int(len(candidate_frame)),
            "candidate_count": int(candidate_frame["candidate_eligible"].sum()),
            "excluded_count": int((~candidate_frame["candidate_eligible"]).sum()),
            "exclusion_reason_counts": {
                reason: int(count)
                for reason, count in sorted(
                    (
                        reason,
                        sum(
                            1
                            for raw in candidate_frame.loc[
                                ~candidate_frame["candidate_eligible"],
                                "exclusion_reasons",
                            ].tolist()
                            if reason in parse_list_cell(raw)
                        ),
                    )
                    for reason in sorted(
                        {
                            reason
                            for raw in candidate_frame.loc[
                                ~candidate_frame["candidate_eligible"],
                                "exclusion_reasons",
                            ].tolist()
                            for reason in parse_list_cell(raw)
                        }
                    )
                )
            },
        },
        "rejected_constraints_and_exclusions": {
            "selected_squad_legality_issues": {
                variant: list(summary["squad_legality"]["issue_codes"])
                for variant, summary in variant_summaries.items()
            },
            "provisional_plan_legality_issues": {
                variant: list(summary["provisional_plan_legality"]["issue_codes"])
                for variant, summary in variant_summaries.items()
            },
            "controlled_variant_relaxation_attempts": {
                variant: list(summary.get("relaxation_attempts") or [])
                for variant, summary in variant_summaries.items()
            },
            "excluded_player_count": int((~candidate_frame["candidate_eligible"]).sum()),
            "candidate_pool_csv_contains_player_level_exclusion_reasons": True,
        },
        "variants": variant_summaries,
        "variant_distinctness": {
            "all_three_distinct": variants_distinct,
            "primary_vs_alternative_a_overlap": len(selected_sets[0] & selected_sets[1]),
            "primary_vs_alternative_b_overlap": len(selected_sets[0] & selected_sets[2]),
            "alternative_a_vs_alternative_b_overlap": len(selected_sets[1] & selected_sets[2]),
        },
        "scope_boundary": {
            "final_lineup_selected": False,
            "final_captain_selected": False,
            "final_vice_captain_selected": False,
            "final_bench_order_selected": False,
            "provisional_objective_assignment_only": True,
            "day101b_required": True,
        },
        "passed": passed,
        "ready_for_day101b": passed,
        "stop_point_satisfied": passed,
        "blockers": blockers,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Day101A preview opening-squad optimizer from a validated "
            "Day97A prediction horizon. Read-only with immutable local artifacts."
        )
    )
    parser.add_argument(
        "--prediction-horizon-report-json",
        required=True,
        help="Day97A player_prediction_horizon_report.json path.",
    )
    parser.add_argument("--target-season", default="2026_27")
    parser.add_argument("--artifact-root", default="/private/tmp/fpl-artifacts")
    parser.add_argument("--squad-rules-config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.prediction_horizon_report_json).expanduser().resolve()
    source_report = load_json(report_path, "Day97A prediction horizon report")
    validate_source_report(source_report)
    if str(source_report.get("target_season")) != str(args.target_season):
        raise OpeningSquadOptimizerError(
            "Day97A target_season=%s does not match CLI target_season=%s."
            % (source_report.get("target_season"), args.target_season)
        )

    optimizer_path = resolve_day97_artifact(source_report, "optimizer_projection_rows_csv")
    long_horizon_path = resolve_day97_artifact(source_report, "player_prediction_horizon_csv")
    projection_frame = normalize_projection_rows(pd.read_csv(optimizer_path))
    long_frame = normalize_long_horizon(pd.read_csv(long_horizon_path))

    effective_gameweeks = sorted(projection_frame["target_gw"].unique().astype(int).tolist())
    expected_gameweeks = list(
        range(1, 1 + int(source_report["effective_horizon"]))
    )
    if effective_gameweeks != expected_gameweeks:
        raise OpeningSquadOptimizerError(
            "Optimizer source Gameweeks %s do not match effective horizon %s."
            % (effective_gameweeks, expected_gameweeks)
        )

    metadata = build_player_metadata(long_frame, projection_frame, start_gw=1)
    candidate_audit = build_candidate_audit(long_frame, projection_frame, start_gw=1)
    rules = load_squad_transfer_rules(
        args.target_season,
        config_path=(
            None
            if args.squad_rules_config is None
            else Path(args.squad_rules_config).expanduser().resolve()
        ),
    )
    engine = SquadLegalityEngine(rules)
    policies = build_variant_policies(args.target_season)
    projection_records_all = projection_records(projection_frame)

    primary = solve_with_relaxations(
        variant="primary",
        policy=policies["primary"],
        projection_frame=projection_frame,
        metadata=metadata,
        rules=rules,
        effective_gameweeks=effective_gameweeks,
    )
    primary = validate_and_evaluate_variant(
        variant_result=primary,
        policy=policies["primary"],
        projection_records_all=projection_records_all,
        metadata=metadata,
        engine=engine,
    )
    primary_ids = set(int(value) for value in primary["selected_player_ids"])

    alternative_a = solve_with_relaxations(
        variant="alternative_a",
        policy=policies["alternative_a"],
        projection_frame=projection_frame,
        metadata=metadata,
        rules=rules,
        effective_gameweeks=effective_gameweeks,
        primary_ids=primary_ids,
    )
    alternative_a = validate_and_evaluate_variant(
        variant_result=alternative_a,
        policy=policies["alternative_a"],
        projection_records_all=projection_records_all,
        metadata=metadata,
        engine=engine,
    )
    alternative_a_ids = set(
        int(value) for value in alternative_a["selected_player_ids"]
    )

    alternative_b = solve_with_relaxations(
        variant="alternative_b",
        policy=policies["alternative_b"],
        projection_frame=projection_frame,
        metadata=metadata,
        rules=rules,
        effective_gameweeks=effective_gameweeks,
        primary_ids=primary_ids,
        alternative_a_ids=alternative_a_ids,
    )
    alternative_b = validate_and_evaluate_variant(
        variant_result=alternative_b,
        policy=policies["alternative_b"],
        projection_records_all=projection_records_all,
        metadata=metadata,
        engine=engine,
    )

    variants = {
        "primary": primary,
        "alternative_a": alternative_a,
        "alternative_b": alternative_b,
    }
    squad_frame = selected_squad_frame(variants, metadata, projection_frame)
    objective_frame = objective_breakdown_frame(variants, metadata)
    candidate_audit = update_candidate_selections(candidate_audit, variants)

    source_run_id = str((source_report.get("run_metadata") or {}).get("run_id") or "")
    source_version = str(source_report.get("horizon_version") or "day97a_v1")
    source_metadata = {
        "day97a_report": file_metadata(
            report_path,
            artifact_type="player_prediction_horizon",
            run_id=source_run_id,
            version=source_version,
        ),
        "optimizer_projection_rows": file_metadata(
            optimizer_path,
            artifact_type="player_prediction_horizon",
            run_id=source_run_id,
            version=source_version,
        ),
        "player_prediction_horizon": file_metadata(
            long_horizon_path,
            artifact_type="player_prediction_horizon",
            run_id=source_run_id,
            version=source_version,
        ),
    }
    created_at = utc_now()
    run_metadata = build_run_metadata_payload(
        report=source_report,
        target_season=args.target_season,
        created_at=created_at,
        source_metadata=source_metadata,
        rules=rules,
        primary_policy=policies["primary"],
    )
    report = build_report(
        source_report=source_report,
        run_metadata=run_metadata,
        candidate_frame=candidate_audit,
        variants=variants,
        policies=policies,
        source_metadata=source_metadata,
        rules_metadata=engine.rules_metadata(),
    )
    if not report["passed"]:
        raise OpeningSquadOptimizerError(
            "Day101A validation failed: %s" % report["blockers"]
        )

    stored = write_immutable_outputs(
        artifact_root=Path(args.artifact_root),
        target_season=args.target_season,
        as_of_time=str(source_report["as_of_time_utc"]),
        run_id=str(run_metadata["run_id"]),
        squad_frame=squad_frame,
        objective_frame=objective_frame,
        candidate_frame=candidate_audit,
        run_metadata=run_metadata,
        report=report,
    )

    print("Day101A opening squad optimizer complete.")
    print("run_id:", run_metadata["run_id"])
    print("target_season:", report["target_season"])
    print("objective_mode:", report["objective_mode"])
    print("effective_horizon:", report["effective_horizon"])
    print("candidate_players:", report["candidate_audit"]["candidate_count"])
    print("excluded_players:", report["candidate_audit"]["excluded_count"])
    for variant in ("primary", "alternative_a", "alternative_b"):
        summary = report["variants"][variant]
        print(
            "%s: cost=%s bank=%s objective=%s legal=%s"
            % (
                variant,
                summary["total_cost_units"],
                summary["bank_units"],
                summary["objective_value"],
                summary["squad_legality"]["valid"],
            )
        )
    print("immutable_artifacts:", len(stored))
    print("preview_only:", str(report["preview_only"]).lower())
    print("writes_database:", str(report["writes_database"]).lower())
    print("writes_squad_state:", str(report["writes_squad_state"]).lower())
    print("production_approved:", str(report["production_approved"]).lower())
    print("ready_for_day101b:", str(report["ready_for_day101b"]).lower())
    print("stop_point_satisfied:", str(report["stop_point_satisfied"]).lower())


if __name__ == "__main__":
    main()
