from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "fpl_scoring_rules_v1"
VALID_POSITIONS = ("GKP", "DEF", "MID", "FWD")
DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "config" / "fpl"
DEFAULT_DAY72A_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "ml"
    / "predict"
    / "build_pre_gw1_player_prediction_preview.py"
)


class ScoringRulesError(ValueError):
    """Raised when a scoring-rules document is present but invalid."""


class ScoringRulesNotFoundError(FileNotFoundError):
    """Raised when no scoring-rules document exists for a requested season."""


@dataclass(frozen=True)
class ScoringRules:
    effective_season: str
    rules_version: str
    schema_version: str
    path: Path
    sha256: str
    data: Dict[str, Any]

    @property
    def scoring(self) -> Dict[str, Any]:
        return self.data["scoring"]

    def goal_points(self, position: str) -> int:
        validate_position(position)
        return int(self.scoring["goals"]["points_by_position"][position])

    def clean_sheet_points(self, position: str) -> int:
        validate_position(position)
        return int(
            self.scoring["clean_sheets"]["points_by_position"][position]
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def validate_position(position: str) -> None:
    if position not in VALID_POSITIONS:
        raise ScoringRulesError(
            "Invalid position %r. Expected one of %s."
            % (position, ", ".join(VALID_POSITIONS))
        )


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScoringRulesError("%s must be an object." % label)
    return value


def require_list(value: Any, label: str) -> List[Any]:
    if not isinstance(value, list):
        raise ScoringRulesError("%s must be a list." % label)
    return value


def require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScoringRulesError("%s must be a non-empty string." % label)
    return value.strip()


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScoringRulesError("%s must be an integer." % label)
    return value


def require_nonnegative_int(value: Any, label: str) -> int:
    parsed = require_int(value, label)
    if parsed < 0:
        raise ScoringRulesError("%s must be non-negative." % label)
    return parsed


def require_exact_keys(
    mapping: Mapping[str, Any],
    expected_keys: Sequence[str],
    label: str,
) -> None:
    expected = set(expected_keys)
    actual = set(mapping.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        messages: List[str] = []
        if missing:
            messages.append("missing=%s" % missing)
        if extra:
            messages.append("extra=%s" % extra)
        raise ScoringRulesError("%s keys are invalid (%s)." % (label, "; ".join(messages)))


def validate_position_points(value: Any, label: str) -> Dict[str, int]:
    mapping = require_mapping(value, label)
    require_exact_keys(mapping, VALID_POSITIONS, label)
    return {
        position: require_int(mapping[position], "%s.%s" % (label, position))
        for position in VALID_POSITIONS
    }


def validate_season_format(season: str) -> None:
    if re.fullmatch(r"\d{4}_\d{2}", season) is None:
        raise ScoringRulesError(
            "effective_season must use YYYY_YY format, for example 2025_26."
        )
    start_year = int(season[:4])
    expected_suffix = str((start_year + 1) % 100).zfill(2)
    if season[-2:] != expected_suffix:
        raise ScoringRulesError(
            "effective_season %s is not a consecutive season label." % season
        )


def validate_rules_document(
    document: Mapping[str, Any],
    expected_season: Optional[str] = None,
) -> None:
    required_top_level = {
        "schema_version",
        "competition",
        "effective_season",
        "rules_version",
        "status",
        "source_notes",
        "positions",
        "scoring",
        "deterministic_examples",
    }
    missing = sorted(required_top_level - set(document.keys()))
    if missing:
        raise ScoringRulesError(
            "Scoring-rules document is missing top-level keys: %s." % missing
        )

    schema_version = require_nonempty_string(
        document["schema_version"], "schema_version"
    )
    if schema_version != SCHEMA_VERSION:
        raise ScoringRulesError(
            "Unsupported schema_version=%s; expected %s."
            % (schema_version, SCHEMA_VERSION)
        )

    competition = require_nonempty_string(document["competition"], "competition")
    if competition != "FPL":
        raise ScoringRulesError("competition must be FPL.")

    season = require_nonempty_string(
        document["effective_season"], "effective_season"
    )
    validate_season_format(season)
    if expected_season is not None and season != expected_season:
        raise ScoringRulesError(
            "Rules effective_season=%s does not match requested season=%s."
            % (season, expected_season)
        )

    require_nonempty_string(document["rules_version"], "rules_version")
    require_nonempty_string(document["status"], "status")
    require_list(document["source_notes"], "source_notes")

    positions = require_list(document["positions"], "positions")
    if positions != list(VALID_POSITIONS):
        raise ScoringRulesError(
            "positions must be exactly %s in canonical order." % (list(VALID_POSITIONS),)
        )

    scoring = require_mapping(document["scoring"], "scoring")
    required_rule_names = {
        "appearance",
        "goals",
        "assists",
        "clean_sheets",
        "saves",
        "penalty_saves",
        "cards",
        "own_goals",
        "penalty_misses",
        "goals_conceded",
        "bonus",
        "defensive_contributions",
    }
    missing_rules = sorted(required_rule_names - set(scoring.keys()))
    if missing_rules:
        raise ScoringRulesError("scoring is missing rules: %s." % missing_rules)

    appearance = require_mapping(scoring["appearance"], "scoring.appearance")
    bands = require_list(appearance.get("bands"), "scoring.appearance.bands")
    if len(bands) != 2:
        raise ScoringRulesError("scoring.appearance.bands must contain two bands.")
    first_band = require_mapping(bands[0], "scoring.appearance.bands[0]")
    second_band = require_mapping(bands[1], "scoring.appearance.bands[1]")
    expected_first = {"min_minutes": 1, "max_minutes": 59, "points": 1}
    expected_second = {"min_minutes": 60, "max_minutes": None, "points": 2}
    if dict(first_band) != expected_first or dict(second_band) != expected_second:
        raise ScoringRulesError(
            "appearance bands must encode 1 point for 1-59 minutes and 2 points for 60+."
        )

    goals = require_mapping(scoring["goals"], "scoring.goals")
    validate_position_points(
        goals.get("points_by_position"), "scoring.goals.points_by_position"
    )

    assists = require_mapping(scoring["assists"], "scoring.assists")
    if require_int(assists.get("points"), "scoring.assists.points") != 3:
        raise ScoringRulesError("scoring.assists.points must be 3.")
    require_nonempty_string(
        assists.get("definition_version"), "scoring.assists.definition_version"
    )

    clean_sheets = require_mapping(
        scoring["clean_sheets"], "scoring.clean_sheets"
    )
    validate_position_points(
        clean_sheets.get("points_by_position"),
        "scoring.clean_sheets.points_by_position",
    )
    if require_int(
        clean_sheets.get("minimum_minutes"),
        "scoring.clean_sheets.minimum_minutes",
    ) != 60:
        raise ScoringRulesError("clean-sheet minimum_minutes must be 60.")

    saves = require_mapping(scoring["saves"], "scoring.saves")
    if require_int(saves.get("per_saves"), "scoring.saves.per_saves") != 3:
        raise ScoringRulesError("save points must be awarded per 3 saves.")
    if require_int(saves.get("points"), "scoring.saves.points") != 1:
        raise ScoringRulesError("save points must award 1 point per band.")

    penalty_saves = require_mapping(
        scoring["penalty_saves"], "scoring.penalty_saves"
    )
    if require_int(
        penalty_saves.get("points"), "scoring.penalty_saves.points"
    ) != 5:
        raise ScoringRulesError("penalty saves must award 5 points.")

    cards = require_mapping(scoring["cards"], "scoring.cards")
    if require_int(cards.get("yellow_points"), "scoring.cards.yellow_points") != -1:
        raise ScoringRulesError("yellow cards must score -1.")
    if require_int(cards.get("red_points"), "scoring.cards.red_points") != -3:
        raise ScoringRulesError("red cards must score -3.")
    if cards.get("red_points_include_yellow_deductions") is not True:
        raise ScoringRulesError(
            "cards.red_points_include_yellow_deductions must be true."
        )

    own_goals = require_mapping(scoring["own_goals"], "scoring.own_goals")
    if require_int(own_goals.get("points"), "scoring.own_goals.points") != -2:
        raise ScoringRulesError("own goals must score -2.")

    penalty_misses = require_mapping(
        scoring["penalty_misses"], "scoring.penalty_misses"
    )
    if require_int(
        penalty_misses.get("points"), "scoring.penalty_misses.points"
    ) != -2:
        raise ScoringRulesError("penalty misses must score -2.")

    goals_conceded = require_mapping(
        scoring["goals_conceded"], "scoring.goals_conceded"
    )
    eligible_positions = require_list(
        goals_conceded.get("eligible_positions"),
        "scoring.goals_conceded.eligible_positions",
    )
    if eligible_positions != ["GKP", "DEF"]:
        raise ScoringRulesError(
            "goals-conceded eligible_positions must be ['GKP', 'DEF']."
        )
    if require_int(
        goals_conceded.get("per_goals"), "scoring.goals_conceded.per_goals"
    ) != 2:
        raise ScoringRulesError("goals-conceded deductions must apply per 2 goals.")
    if require_int(
        goals_conceded.get("points"), "scoring.goals_conceded.points"
    ) != -1:
        raise ScoringRulesError("goals-conceded rule must deduct 1 point.")

    bonus = require_mapping(scoring["bonus"], "scoring.bonus")
    allowed_bonus = require_list(
        bonus.get("allowed_player_points"), "scoring.bonus.allowed_player_points"
    )
    if allowed_bonus != [0, 1, 2, 3]:
        raise ScoringRulesError("bonus.allowed_player_points must be [0, 1, 2, 3].")
    rank_points = require_mapping(
        bonus.get("rank_points"), "scoring.bonus.rank_points"
    )
    if dict(rank_points) != {"1": 3, "2": 2, "3": 1}:
        raise ScoringRulesError("bonus.rank_points must be {'1': 3, '2': 2, '3': 1}.")

    defensive = require_mapping(
        scoring["defensive_contributions"],
        "scoring.defensive_contributions",
    )
    by_position = require_mapping(
        defensive.get("by_position"),
        "scoring.defensive_contributions.by_position",
    )
    require_exact_keys(by_position, VALID_POSITIONS, "defensive contributions")
    expected_thresholds = {"GKP": None, "DEF": 10, "MID": 12, "FWD": 12}
    for position in VALID_POSITIONS:
        policy = require_mapping(
            by_position[position],
            "scoring.defensive_contributions.by_position.%s" % position,
        )
        threshold = policy.get("threshold")
        if threshold != expected_thresholds[position]:
            raise ScoringRulesError(
                "Defensive-contribution threshold for %s must be %s."
                % (position, expected_thresholds[position])
            )
        if position == "GKP":
            if policy.get("points") != 0:
                raise ScoringRulesError("GKP defensive-contribution points must be 0.")
        else:
            if policy.get("points") != 2:
                raise ScoringRulesError(
                    "%s defensive-contribution points must be 2." % position
                )
            if policy.get("max_points_per_match") != 2:
                raise ScoringRulesError(
                    "%s defensive contributions must be capped at 2 points."
                    % position
                )

    examples = require_list(
        document["deterministic_examples"], "deterministic_examples"
    )
    if len(examples) < len(VALID_POSITIONS):
        raise ScoringRulesError(
            "deterministic_examples must contain at least one example per position."
        )
    example_positions = [
        require_mapping(example, "deterministic example").get("position")
        for example in examples
    ]
    missing_example_positions = sorted(set(VALID_POSITIONS) - set(example_positions))
    if missing_example_positions:
        raise ScoringRulesError(
            "Missing deterministic examples for positions: %s."
            % missing_example_positions
        )


def config_path_for_season(
    season: str,
    config_root: Optional[Path] = None,
) -> Path:
    validate_season_format(season)
    root = config_root or DEFAULT_CONFIG_ROOT
    return root / ("scoring_rules_%s.json" % season)


def load_scoring_rules(
    season: str,
    config_path: Optional[Path] = None,
    config_root: Optional[Path] = None,
) -> ScoringRules:
    path = config_path or config_path_for_season(season, config_root=config_root)
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ScoringRulesNotFoundError(
            "No FPL scoring-rules registry exists for season=%s at %s."
            % (season, path)
        )
    if not path.is_file():
        raise ScoringRulesNotFoundError(
            "Scoring-rules path is not a file: %s." % path
        )

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScoringRulesError(
            "Invalid JSON in scoring-rules file %s: %s." % (path, exc)
        )

    document_mapping = require_mapping(document, "scoring-rules document")
    validate_rules_document(document_mapping, expected_season=season)
    return ScoringRules(
        effective_season=str(document_mapping["effective_season"]),
        rules_version=str(document_mapping["rules_version"]),
        schema_version=str(document_mapping["schema_version"]),
        path=path,
        sha256=sha256_file(path),
        data=dict(document_mapping),
    )


def event_int(events: Mapping[str, Any], key: str) -> int:
    value = events.get(key, 0)
    return require_nonnegative_int(value, "events.%s" % key)


def score_player_match(
    rules: ScoringRules,
    position: str,
    events: Mapping[str, Any],
) -> Dict[str, int]:
    """Score one player-match event row using only the loaded registry."""

    validate_position(position)
    scoring = rules.scoring

    minutes = event_int(events, "minutes")
    goals = event_int(events, "goals")
    assists = event_int(events, "assists")
    saves = event_int(events, "saves")
    penalty_saves = event_int(events, "penalty_saves")
    yellow_cards = event_int(events, "yellow_cards")
    red_cards = event_int(events, "red_cards")
    own_goals = event_int(events, "own_goals")
    penalty_misses = event_int(events, "penalty_misses")
    goals_conceded = event_int(events, "goals_conceded_for_scoring")
    bonus_points = event_int(events, "bonus_points")
    defensive_contributions = event_int(events, "defensive_contributions")
    clean_sheet = bool_value(events.get("clean_sheet"))
    acted_as_goalkeeper = bool_value(events.get("acted_as_goalkeeper")) or position == "GKP"

    appearance_points = 0
    if minutes >= 60:
        appearance_points = 2
    elif minutes >= 1:
        appearance_points = 1

    goal_points = goals * rules.goal_points(position)
    assist_points = assists * int(scoring["assists"]["points"])

    clean_sheet_points = 0
    if clean_sheet and minutes >= int(scoring["clean_sheets"]["minimum_minutes"]):
        clean_sheet_points = rules.clean_sheet_points(position)

    if (saves > 0 or penalty_saves > 0) and not acted_as_goalkeeper:
        raise ScoringRulesError(
            "Save events require acted_as_goalkeeper=true for non-GKP positions."
        )
    save_points = 0
    penalty_save_points = 0
    if acted_as_goalkeeper:
        save_points = (
            saves // int(scoring["saves"]["per_saves"])
        ) * int(scoring["saves"]["points"])
        penalty_save_points = penalty_saves * int(
            scoring["penalty_saves"]["points"]
        )

    if red_cards > 0:
        card_points = red_cards * int(scoring["cards"]["red_points"])
    else:
        card_points = yellow_cards * int(scoring["cards"]["yellow_points"])

    own_goal_points = own_goals * int(scoring["own_goals"]["points"])
    penalty_miss_points = penalty_misses * int(
        scoring["penalty_misses"]["points"]
    )

    goals_conceded_points = 0
    if position in scoring["goals_conceded"]["eligible_positions"]:
        goals_conceded_points = (
            goals_conceded // int(scoring["goals_conceded"]["per_goals"])
        ) * int(scoring["goals_conceded"]["points"])

    allowed_bonus = scoring["bonus"]["allowed_player_points"]
    if bonus_points not in allowed_bonus:
        raise ScoringRulesError(
            "events.bonus_points=%s is invalid; expected one of %s."
            % (bonus_points, allowed_bonus)
        )

    defensive_policy = scoring["defensive_contributions"]["by_position"][position]
    defensive_points = 0
    threshold = defensive_policy.get("threshold")
    if threshold is not None and defensive_contributions >= int(threshold):
        defensive_points = min(
            int(defensive_policy["points"]),
            int(defensive_policy["max_points_per_match"]),
        )

    components = {
        "appearance_points": appearance_points,
        "goal_points": goal_points,
        "assist_points": assist_points,
        "clean_sheet_points": clean_sheet_points,
        "save_points": save_points,
        "penalty_save_points": penalty_save_points,
        "card_points": card_points,
        "own_goal_points": own_goal_points,
        "penalty_miss_points": penalty_miss_points,
        "goals_conceded_points": goals_conceded_points,
        "bonus_points": bonus_points,
        "defensive_contribution_points": defensive_points,
    }
    components["total_points"] = sum(components.values())
    return components


def validate_deterministic_examples(
    rules: ScoringRules,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    results: List[Dict[str, Any]] = []
    blockers: List[str] = []
    for index, example_value in enumerate(rules.data["deterministic_examples"]):
        example = require_mapping(
            example_value, "deterministic_examples[%s]" % index
        )
        example_id = require_nonempty_string(
            example.get("id"), "deterministic_examples[%s].id" % index
        )
        position = require_nonempty_string(
            example.get("position"),
            "deterministic_examples[%s].position" % index,
        )
        validate_position(position)
        events = require_mapping(
            example.get("events"),
            "deterministic_examples[%s].events" % index,
        )
        expected_total = require_int(
            example.get("expected_total_points"),
            "deterministic_examples[%s].expected_total_points" % index,
        )
        actual_components = score_player_match(rules, position, events)
        actual_total = actual_components["total_points"]
        passed = actual_total == expected_total
        if not passed:
            blockers.append(
                "Deterministic example %s expected %s points but produced %s."
                % (example_id, expected_total, actual_total)
            )
        results.append(
            {
                "id": example_id,
                "position": position,
                "expected_total_points": expected_total,
                "actual_total_points": actual_total,
                "passed": passed,
                "components": actual_components,
            }
        )
    return results, blockers


def literal_assignment(tree: ast.AST, variable_name: str) -> Optional[Any]:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target: Optional[ast.expr]
        value: Optional[ast.expr]
        if isinstance(node, ast.Assign):
            target = node.targets[0] if len(node.targets) == 1 else None
            value = node.value
        else:
            target = node.target
            value = node.value
        if isinstance(target, ast.Name) and target.id == variable_name and value is not None:
            try:
                return ast.literal_eval(value)
            except (ValueError, TypeError):
                return None
    return None


def extract_day72a_assumptions(source_path: Path) -> Dict[str, Any]:
    path = source_path.expanduser().resolve()
    if not path.exists():
        raise ScoringRulesNotFoundError("Day72A source file not found: %s." % path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    goal_points = literal_assignment(tree, "GOAL_POINTS_BY_POSITION")
    clean_sheet_points = literal_assignment(
        tree, "CLEAN_SHEET_POINTS_BY_POSITION"
    )

    assist_match = re.search(
        r"expected_assist_points\s*=\s*expected_assists\s*\*\s*([0-9.]+)",
        source,
    )
    assist_points = float(assist_match.group(1)) if assist_match else None

    appearance_proxy_present = re.search(
        r"expected_appearance_points\s*=\s*appearance_probability\s*\+\s*start_probability",
        source,
    ) is not None
    other_points_reconciliation_present = re.search(
        r"expected_other_points\s*=\s*guarded_points\s*-\s*component_known_sum",
        source,
    ) is not None
    unresolved_version_present = "target_season_rules_unresolved" in source

    return {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "goal_points_by_position": goal_points,
        "clean_sheet_points_by_position": clean_sheet_points,
        "assist_points": assist_points,
        "appearance_probability_plus_start_probability_proxy": (
            appearance_proxy_present
        ),
        "other_points_reconciles_unknown_components": (
            other_points_reconciliation_present
        ),
        "target_season_rules_unresolved_default_present": (
            unresolved_version_present
        ),
    }


def compare_day72a_assumptions(
    rules: ScoringRules,
    day72a_source: Path,
) -> Dict[str, Any]:
    assumptions = extract_day72a_assumptions(day72a_source)
    scoring = rules.scoring
    registry_goals = scoring["goals"]["points_by_position"]
    registry_clean_sheets = scoring["clean_sheets"]["points_by_position"]

    comparisons: List[Dict[str, Any]] = []

    day72_goals = assumptions.get("goal_points_by_position")
    goal_differences: Dict[str, Dict[str, Any]] = {}
    if isinstance(day72_goals, Mapping):
        for position in VALID_POSITIONS:
            day72_value = day72_goals.get(position)
            registry_value = registry_goals.get(position)
            if day72_value != registry_value:
                goal_differences[position] = {
                    "day72a": day72_value,
                    "registry": registry_value,
                }
    comparisons.append(
        {
            "rule": "goal_points_by_position",
            "status": "exact_match" if not goal_differences else "mismatch",
            "differences": goal_differences,
            "day72a": day72_goals,
            "registry": registry_goals,
        }
    )

    day72_clean_sheets = assumptions.get("clean_sheet_points_by_position")
    clean_sheet_match = day72_clean_sheets == registry_clean_sheets
    comparisons.append(
        {
            "rule": "clean_sheet_points_by_position",
            "status": "exact_match" if clean_sheet_match else "mismatch",
            "day72a": day72_clean_sheets,
            "registry": registry_clean_sheets,
        }
    )

    day72_assist_points = assumptions.get("assist_points")
    registry_assist_points = scoring["assists"]["points"]
    assist_match = day72_assist_points == float(registry_assist_points)
    comparisons.append(
        {
            "rule": "assist_points",
            "status": "exact_match" if assist_match else "mismatch",
            "day72a": day72_assist_points,
            "registry": registry_assist_points,
        }
    )

    comparisons.append(
        {
            "rule": "appearance_points",
            "status": "proxy_not_exact_rule",
            "day72a": (
                "appearance_probability + start_probability"
                if assumptions[
                    "appearance_probability_plus_start_probability_proxy"
                ]
                else None
            ),
            "registry": "1 point for 1-59 completed minutes; 2 points for 60+",
        }
    )

    comparisons.append(
        {
            "rule": "bonus_points",
            "status": "rate_proxy_not_exact_rule",
            "day72a": "historical bonus_per90 proxy",
            "registry": "individual awarded bonus must be 0, 1, 2, or 3",
        }
    )

    explicit_day72_components = {
        "saves": False,
        "penalty_saves": False,
        "cards": False,
        "own_goals": False,
        "penalty_misses": False,
        "goals_conceded": False,
        "defensive_contributions": False,
    }
    for rule_name, explicit in explicit_day72_components.items():
        comparisons.append(
            {
                "rule": rule_name,
                "status": "not_explicitly_modeled_in_day72a",
                "day72a": (
                    "absorbed into expected_other_points reconciliation"
                    if assumptions["other_points_reconciles_unknown_components"]
                    else "not detected"
                ),
                "registry": "explicit_rule_present",
                "day72a_explicit_component": explicit,
            }
        )

    status_counts: Dict[str, int] = {}
    for comparison in comparisons:
        status = str(comparison["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "day72a_assumptions": assumptions,
        "comparisons": comparisons,
        "status_counts": status_counts,
        "explicit_mismatch_count": status_counts.get("mismatch", 0),
        "registry_matches_all_day72a_explicit_constants": (
            status_counts.get("mismatch", 0) == 0
        ),
        "day72a_uses_registry": False,
        "expected_known_mismatch": {
            "rule": "goal_points_by_position.GKP",
            "day72a": (
                day72_goals.get("GKP")
                if isinstance(day72_goals, Mapping)
                else None
            ),
            "registry": registry_goals.get("GKP"),
        },
    }


def build_validation_report(
    rules: ScoringRules,
    day72a_source: Path,
) -> Dict[str, Any]:
    blockers: List[str] = []
    warnings: List[str] = []

    example_results, example_blockers = validate_deterministic_examples(rules)
    blockers.extend(example_blockers)

    try:
        comparison = compare_day72a_assumptions(rules, day72a_source)
    except (OSError, SyntaxError, ScoringRulesError) as exc:
        comparison = {
            "day72a_assumptions": {},
            "comparisons": [],
            "status_counts": {},
            "explicit_mismatch_count": None,
            "registry_matches_all_day72a_explicit_constants": None,
            "day72a_uses_registry": False,
        }
        blockers.append("Unable to compare Day72A assumptions: %s" % exc)

    if comparison.get("explicit_mismatch_count", 0):
        warnings.append(
            "Day72A explicit scoring constants do not fully match the target-season registry."
        )
    if comparison.get("day72a_uses_registry") is False:
        warnings.append(
            "Day72A does not load this registry yet; Day74A intentionally does not change model code."
        )

    passed = len(blockers) == 0
    return {
        "created_at": utc_now(),
        "artifact_type": "target_season_scoring_rules_validation",
        "validation_version": "day74a_v1",
        "schema_version": rules.schema_version,
        "effective_season": rules.effective_season,
        "rules_version": rules.rules_version,
        "rules_path": str(rules.path),
        "rules_sha256": rules.sha256,
        "passed": passed,
        "audit_only": True,
        "writes_database": False,
        "changes_model_code": False,
        "ready_for_target_season_rules_loading": passed,
        "ready_for_deterministic_rule_scoring": passed,
        "ready_for_future_model_integration": passed,
        "day72a_uses_registry": False,
        "ready_for_production_prediction": False,
        "deterministic_examples": {
            "example_count": len(example_results),
            "passed_count": sum(1 for item in example_results if item["passed"]),
            "results": example_results,
        },
        "day72a_comparison": comparison,
        "blockers": blockers,
        "warnings": warnings,
        "production_boundary": (
            "The registry can be loaded and tested independently, but Day72A still uses "
            "embedded heuristic assumptions. Model integration, historical validation, "
            "and calibration remain separate future work."
        ),
    }


def write_json(report: Mapping[str, Any], path_value: str) -> None:
    if not path_value:
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_markdown(report: Mapping[str, Any], path_value: str) -> None:
    if not path_value:
        return
    path = Path(path_value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    examples = report["deterministic_examples"]
    comparison = report["day72a_comparison"]
    lines = [
        "# Day74A Target-Season Scoring Rules Validation",
        "",
        "- Effective season: `%s`" % report["effective_season"],
        "- Rules version: `%s`" % report["rules_version"],
        "- Schema version: `%s`" % report["schema_version"],
        "- Rules SHA256: `%s`" % report["rules_sha256"],
        "- Passed: `%s`" % report["passed"],
        "- Audit only: `%s`" % report["audit_only"],
        "- Writes database: `%s`" % report["writes_database"],
        "- Day72A uses registry: `%s`" % report["day72a_uses_registry"],
        "",
        "## Deterministic Examples",
        "",
        "- Passed: `%s/%s`"
        % (examples["passed_count"], examples["example_count"]),
        "",
    ]
    for result in examples["results"]:
        lines.append(
            "- `%s` (%s): expected `%s`, actual `%s`, passed `%s`"
            % (
                result["id"],
                result["position"],
                result["expected_total_points"],
                result["actual_total_points"],
                result["passed"],
            )
        )

    lines.extend(["", "## Day72A Comparison", ""])
    for item in comparison.get("comparisons", []):
        lines.append("- `%s`: `%s`" % (item["rule"], item["status"]))

    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend("- %s" % item for item in report["blockers"])
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend("- %s" % item for item in report["warnings"])
    else:
        lines.append("- None")

    lines.extend(["", "## Production Boundary", "", report["production_boundary"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(report: Mapping[str, Any], out_json: str, out_md: str) -> None:
    examples = report["deterministic_examples"]
    comparison = report["day72a_comparison"]
    print("=== Day74A Target-Season Scoring Rules Registry ===")
    print("validation_version:", report["validation_version"])
    print("effective_season:", report["effective_season"])
    print("rules_version:", report["rules_version"])
    print("schema_version:", report["schema_version"])
    print("rules_sha256:", report["rules_sha256"])
    print("passed:", report["passed"])
    print("audit_only:", report["audit_only"])
    print("writes_database:", report["writes_database"])
    print(
        "ready_for_target_season_rules_loading:",
        report["ready_for_target_season_rules_loading"],
    )
    print(
        "ready_for_deterministic_rule_scoring:",
        report["ready_for_deterministic_rule_scoring"],
    )
    print("day72a_uses_registry:", report["day72a_uses_registry"])
    print("ready_for_production_prediction:", report["ready_for_production_prediction"])
    print()
    print("Deterministic examples:")
    print("- example_count:", examples["example_count"])
    print("- passed_count:", examples["passed_count"])
    print()
    print("Day72A comparison:")
    print(
        "- explicit_mismatch_count:",
        comparison.get("explicit_mismatch_count"),
    )
    print(
        "- registry_matches_all_day72a_explicit_constants:",
        comparison.get("registry_matches_all_day72a_explicit_constants"),
    )
    for item in comparison.get("comparisons", []):
        print("- %s: %s" % (item["rule"], item["status"]))
    print()
    print("Blockers:", report["blockers"] or "none")
    print("Warnings:", report["warnings"] or "none")
    if out_json:
        print("saved_json:", out_json)
    if out_md:
        print("saved_md:", out_md)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load and validate a versioned target-season FPL scoring-rules registry, "
            "run deterministic examples, and compare it with Day72A assumptions."
        )
    )
    parser.add_argument("--season", required=True, help="Season label, e.g. 2025_26.")
    parser.add_argument(
        "--config-path",
        default="",
        help="Optional explicit scoring-rules JSON path.",
    )
    parser.add_argument(
        "--day72a-source",
        default=str(DEFAULT_DAY72A_SOURCE),
        help="Day72A source file used only for assumption comparison.",
    )
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_path) if args.config_path else None
    rules = load_scoring_rules(args.season, config_path=config_path)
    report = build_validation_report(rules, Path(args.day72a_source))
    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    print_report(report, args.out_json, args.out_md)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
