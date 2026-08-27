#!/usr/bin/env python3
"""Leakage-safe Gameweek POST evaluation from frozen PRE evidence + official actuals.

Python 3.9 compatible. No database access. No prediction regeneration. No mutation of
frozen PRE evidence. Designed first for the 2026/27 GW1 frozen package, while keeping
input discovery generic enough for later compatible Gameweeks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

EPS = 1e-15
POSITIONS = ("GKP", "DEF", "MID", "FWD")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a frozen FPL Gameweek against separately captured official actuals.")
    p.add_argument("--repo-root", default=None)
    p.add_argument("--planning-root", default=None)
    p.add_argument("--season", default="2026_27")
    p.add_argument("--gw", type=int, default=1)
    p.add_argument("--actual-manifest", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--require-final", action="store_true")
    return p.parse_args()


def detect_repo_root(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    script = Path(__file__).resolve()
    for parent in (script.parent,) + tuple(script.parents):
        if (parent / ".git").exists():
            return parent
    return script.parents[3].resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def boolish(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def f(v: Any) -> float:
    if v is None or v == "":
        return float("nan")
    return float(v)


def i(v: Any) -> int:
    return int(float(v))


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def mae(pred: Sequence[float], actual: Sequence[float]) -> float:
    return mean([abs(p - a) for p, a in zip(pred, actual)])


def rmse(pred: Sequence[float], actual: Sequence[float]) -> float:
    return math.sqrt(mean([(p - a) ** 2 for p, a in zip(pred, actual)]))


def bias(pred: Sequence[float], actual: Sequence[float]) -> float:
    return mean([p - a for p, a in zip(pred, actual)])


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg = (pos + 1 + end) / 2.0
        for k in range(pos, end):
            ranks[order[k]] = avg
        pos = end
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(average_ranks(x), average_ranks(y))


def dcg(relevances: Sequence[float]) -> float:
    return sum(max(0.0, rel) / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def ndcg_at_k(rows: Sequence[Dict[str, Any]], k: int) -> float:
    predicted_order = sorted(rows, key=lambda r: (-r["predicted_points"], r["fpl_player_id"]))[:k]
    ideal_order = sorted(rows, key=lambda r: (-r["actual_points"], r["fpl_player_id"]))[:k]
    score = dcg([r["actual_points"] for r in predicted_order])
    ideal = dcg([r["actual_points"] for r in ideal_order])
    return score / ideal if ideal > 0 else float("nan")


def point_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pred = [r["predicted_points"] for r in rows]
    actual = [r["actual_points"] for r in rows]
    return {
        "n": len(rows),
        "mae": mae(pred, actual),
        "rmse": rmse(pred, actual),
        "mean_error_bias_pred_minus_actual": bias(pred, actual),
    }


def top_k_metrics(rows: Sequence[Dict[str, Any]], k: int) -> Dict[str, Any]:
    kk = min(k, len(rows))
    pred_top = sorted(rows, key=lambda r: (-r["predicted_points"], r["fpl_player_id"]))[:kk]
    actual_top = sorted(rows, key=lambda r: (-r["actual_points"], r["fpl_player_id"]))[:kk]
    pred_ids = {r["fpl_player_id"] for r in pred_top}
    actual_ids = {r["fpl_player_id"] for r in actual_top}
    hits = len(pred_ids & actual_ids)
    captured = sum(r["actual_points"] for r in pred_top)
    ideal = sum(r["actual_points"] for r in actual_top)
    hit_rows = [r for r in pred_top if r["fpl_player_id"] in actual_ids]
    return {
        "k": kk,
        "hits": hits,
        "precision_at_k": hits / kk if kk else float("nan"),
        "recall_at_k": hits / kk if kk else float("nan"),
        "actual_points_of_predicted_top_k": captured,
        "actual_points_of_actual_top_k": ideal,
        "points_capture_ratio": captured / ideal if ideal != 0 else float("nan"),
        "predicted_top_ids": [r["fpl_player_id"] for r in pred_top],
        "actual_top_ids": [r["fpl_player_id"] for r in actual_top],
        "hit_ids": [r["fpl_player_id"] for r in hit_rows],
        "hit_names": [r["web_name"] for r in hit_rows],
    }


def classification_accuracy(rows: Sequence[Dict[str, Any]], probability_key: str, actual_key: str) -> float:
    if not rows:
        return float("nan")
    correct = 0
    for r in rows:
        pred = r[probability_key] >= 0.5
        actual = bool(r[actual_key])
        correct += int(pred == actual)
    return correct / len(rows)


def normalize_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_value.lower())


def discover_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one %s under %s; found %d: %s" % (pattern, root, len(matches), matches))
    return matches[0]


def latest_actual_manifest(planning_root: Path, season: str, gw: int) -> Path:
    root = planning_root / "gw-post" / season / ("gw%02d" % gw) / "actuals"
    matches = sorted(root.glob("gw%d_actuals_manifest_*.json" % gw))
    if not matches:
        raise RuntimeError("No actual manifest found under %s" % root)
    return matches[-1]


def parse_sha256_sums(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            result[parts[1].strip()] = parts[0].strip()
    return result


def result_label(home: int, away: int) -> str:
    if home > away:
        return "home_win"
    if home < away:
        return "away_win"
    return "draw"


def validate_formation(players: Sequence[Dict[str, Any]]) -> bool:
    counts = {p: 0 for p in POSITIONS}
    for r in players:
        counts[r["position"]] += 1
    return (
        counts["GKP"] == 1
        and 3 <= counts["DEF"] <= 5
        and 2 <= counts["MID"] <= 5
        and 1 <= counts["FWD"] <= 3
        and len(players) == 11
    )


def apply_model_team_autosubs(starters: List[Dict[str, Any]], bench: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    current = list(starters)
    actions: List[Dict[str, Any]] = []
    # Goalkeeper first.
    gk_starters = [r for r in current if r["position"] == "GKP"]
    gk_bench = [r for r in bench if r["position"] == "GKP"]
    if gk_starters and gk_starters[0]["actual_minutes"] == 0 and gk_bench and gk_bench[0]["actual_minutes"] > 0:
        outgoing, incoming = gk_starters[0], gk_bench[0]
        current[current.index(outgoing)] = incoming
        actions.append({"out": outgoing["web_name"], "in": incoming["web_name"], "reason": "goalkeeper_autosub"})

    outfield_bench = [r for r in bench if r["position"] != "GKP"]
    dnp = [r for r in list(current) if r["position"] != "GKP" and r["actual_minutes"] == 0]
    used: set = set()
    for outgoing in dnp:
        for incoming in outfield_bench:
            if incoming["fpl_player_id"] in used or incoming["actual_minutes"] == 0:
                continue
            candidate = [r for r in current if r["fpl_player_id"] != outgoing["fpl_player_id"]] + [incoming]
            if validate_formation(candidate):
                current[current.index(outgoing)] = incoming
                used.add(incoming["fpl_player_id"])
                actions.append({"out": outgoing["web_name"], "in": incoming["web_name"], "reason": "outfield_autosub"})
                break
    return current, actions


def captain_bonus(players: Sequence[Dict[str, Any]], captain_id: int, vice_id: int) -> Dict[str, Any]:
    """Apply captain/vice fallback only within the XI whose points are being scored."""
    by_id = {r["fpl_player_id"]: r for r in players}
    c = by_id.get(captain_id)
    v = by_id.get(vice_id)
    if c is not None and c["actual_minutes"] > 0:
        return {"effective_captain_id": captain_id, "effective_captain": c["web_name"], "bonus_points": c["actual_points"], "vice_triggered": False}
    if v is not None and v["actual_minutes"] > 0:
        return {"effective_captain_id": vice_id, "effective_captain": v["web_name"], "bonus_points": v["actual_points"], "vice_triggered": True}
    return {"effective_captain_id": None, "effective_captain": None, "bonus_points": 0, "vice_triggered": False}


def evaluate_team(label: str, squad: List[Dict[str, Any]], starters: List[Dict[str, Any]], bench: List[Dict[str, Any]], captain_id: int, vice_id: int, autosub_mode: str) -> Dict[str, Any]:
    """Evaluate frozen-XI decision quality separately from official-FPL autosub realization.

    PRIMARY headline contract:
      - score exactly the frozen starting XI;
      - a frozen starter with 0 minutes contributes its actual 0 points;
      - bench players never replace the frozen XI in the primary score;
      - captain/vice fallback is evaluated only within that frozen XI.

    SECONDARY reference contract:
      - where bench order is authoritative, apply automatic substitutions and legal
        formation constraints to approximate the official realized FPL score;
      - if bench order is not authoritative and a starter did not play, leave the
        secondary official-style score unavailable rather than guessing.
    """
    frozen_xi_raw = sum(r["actual_points"] for r in starters)
    frozen_xi_cap = captain_bonus(starters, captain_id, vice_id)
    primary_actual_total = frozen_xi_raw + frozen_xi_cap["bonus_points"]

    predicted_raw = sum(r["predicted_points"] for r in starters)
    predicted_cap = next(r["predicted_points"] for r in starters if r["fpl_player_id"] == captain_id)
    predicted_total = predicted_raw + predicted_cap

    actions: List[Dict[str, Any]] = []
    effective: Optional[List[Dict[str, Any]]]
    if autosub_mode == "model_team":
        effective, actions = apply_model_team_autosubs(starters, bench)
        autosub_status = "evaluated_from_frozen_bench_order"
    elif all(r["actual_minutes"] > 0 for r in starters):
        effective = list(starters)
        autosub_status = "not_needed_all_frozen_starters_played"
    else:
        effective = None
        autosub_status = "not_evaluated_bench_order_not_authoritative"

    if effective is not None:
        secondary_raw: Optional[float] = sum(r["actual_points"] for r in effective)
        secondary_cap = captain_bonus(effective, captain_id, vice_id)
        secondary_total: Optional[float] = secondary_raw + secondary_cap["bonus_points"]
        secondary_error: Optional[float] = secondary_total - predicted_total
        effective_ids: Optional[List[int]] = [r["fpl_player_id"] for r in effective]
    else:
        secondary_raw = None
        secondary_cap = None
        secondary_total = None
        secondary_error = None
        effective_ids = None

    return {
        "label": label,
        "primary_scoring_contract": "frozen_starting_xi_no_autosub_plus_captain_vice_fallback",
        "secondary_scoring_contract": "official_fpl_autosub_reference_when_evaluable",
        "selected_15_actual_points_raw": sum(r["actual_points"] for r in squad),
        "frozen_starting_xi_actual_points_raw": frozen_xi_raw,
        "frozen_xi_captain_bonus_actual_points": frozen_xi_cap["bonus_points"],
        "primary_frozen_xi_actual_total": primary_actual_total,
        "submitted_predicted_xi_points_raw": predicted_raw,
        "submitted_predicted_total_with_captain": predicted_total,
        "primary_actual_minus_predicted_total": primary_actual_total - predicted_total,
        "bench_actual_points_raw": sum(r["actual_points"] for r in bench),
        "captain": next(r["web_name"] for r in squad if r["fpl_player_id"] == captain_id),
        "vice_captain": next(r["web_name"] for r in squad if r["fpl_player_id"] == vice_id),
        "primary_captain_result": frozen_xi_cap,
        "secondary_official_fpl_available": effective is not None,
        "secondary_post_autosub_xi_actual_points_raw": secondary_raw,
        "secondary_official_fpl_captain_result": secondary_cap,
        "secondary_official_fpl_realized_total": secondary_total,
        "secondary_actual_minus_predicted_total": secondary_error,
        "autosub_status": autosub_status,
        "autosub_actions": actions,
        "frozen_starter_ids": [r["fpl_player_id"] for r in starters],
        "secondary_effective_starter_ids": effective_ids,
        "bench_ids": [r["fpl_player_id"] for r in bench],
        "player_rows": squad,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def build_markdown(result: Dict[str, Any]) -> str:
    p = result["player_model"]
    m = result["match_model"]
    mt = result["model_team"]
    ta = result["team_alex"]
    lines = [
        "# FPL GW%d POST Evaluation — %s" % (result["gw"], result["evaluation_status"]),
        "",
        "**Season:** `%s`  " % result["season"],
        "**Actual capture:** `%s`  " % result["actuals"]["captured_at_utc"],
        "**Leakage-safe:** `True`  ",
        "**Database access:** `False`  ",
        "**Frozen PRE mutation:** `False`",
        "",
        "> %s" % ("PROVISIONAL — official fixtures are finished_provisional, not final." if result["evaluation_status"] == "PROVISIONAL" else "FINAL — official fixture finalization requirement satisfied."),
        "",
        "## Player Model",
        "",
        "- Frozen prediction source: `%s`; production-approved rows: **%d**." % (p.get("frozen_prediction_source"), p.get("production_approved_true_rows", 0)),
        "- Frozen rows: **%d**; matched actuals: **%d**; actual-only rows excluded from primary denominator: **%d**." % (p["coverage"]["frozen_rows"], p["coverage"]["matched_frozen_rows"], p["coverage"]["actual_only_rows"]),
        "- Primary eligible universe: **%d** players." % p["coverage"]["primary_eligible_rows"],
        "- Eligible MAE / RMSE / bias: **%.4f / %.4f / %.4f**." % (p["cohorts"]["all_eligible_players"]["mae"], p["cohorts"]["all_eligible_players"]["rmse"], p["cohorts"]["all_eligible_players"]["mean_error_bias_pred_minus_actual"]),
        "- Spearman: **%.4f**; NDCG@20: **%.4f**." % (p["ranking_quality"]["spearman_rank_correlation"], p["ranking_quality"]["ndcg_at_20"]),
        "- Appearance accuracy: **%.4f**; start accuracy: **%.4f**; minutes MAE: **%.4f**." % (p["availability_and_minutes"]["appearance_accuracy_threshold_0_5"], p["availability_and_minutes"]["start_accuracy_threshold_0_5"], p["availability_and_minutes"]["minutes_mae"]),
        "",
        "### Top-K",
        "",
        "| K | Hits | Precision | Recall | Points capture |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in (10, 20, 50):
        x = p["top_k"][str(k)]
        lines.append("| %d | %d | %.3f | %.3f | %.3f |" % (k, x["hits"], x["precision_at_k"], x["recall_at_k"], x["points_capture_ratio"]))
    lines += [
        "",
        "## Match Model",
        "",
        "- Frozen model: `%s`; probability type: `%s`; calibration: `%s`; production-ready rows: **%d**." % (m.get("frozen_model_name"), m.get("frozen_probability_type"), m.get("frozen_calibration_status"), m.get("production_ready_true_rows", 0)),
        "- Fixtures evaluated: **%d**." % m["n"],
        "- 1X2 accuracy: **%.3f**; log loss: **%.4f**; multiclass Brier: **%.4f**." % (m["one_x_two_accuracy"], m["log_loss"], m["multiclass_brier"]),
        "- Home / away / total goals MAE: **%.3f / %.3f / %.3f**." % (m["home_goals_mae"], m["away_goals_mae"], m["total_goals_mae"]),
        "- Exact score Top-1 / Top-3 / Top-5: **%.3f / %.3f / %.3f**." % (m["exact_score_top1_accuracy"], m["exact_score_top3_accuracy"], m["exact_score_top5_accuracy"]),
        "",
        "## Model Team",
        "",
        "- **PRIMARY frozen-XI decision score:** predicted XI+C **%.3f**; actual frozen XI+C **%.1f**; error actual-predicted **%.3f**." % (mt["submitted_predicted_total_with_captain"], mt["primary_frozen_xi_actual_total"], mt["primary_actual_minus_predicted_total"]),
        "- Frozen XI raw points: **%.1f**; captain/vice bonus within frozen XI: **%.1f**." % (mt["frozen_starting_xi_actual_points_raw"], mt["frozen_xi_captain_bonus_actual_points"]),
        "- Secondary official-FPL autosub reference: **%s**; autosub status: `%s`; actions: `%s`." % (("%.1f" % mt["secondary_official_fpl_realized_total"]) if mt["secondary_official_fpl_realized_total"] is not None else "unavailable", mt["autosub_status"], json.dumps(mt["autosub_actions"], ensure_ascii=False)),
        "- Captain: **%s**; vice: **%s**." % (mt["captain"], mt["vice_captain"]),
        "",
        "## Team Alex / Gliding Tiger",
        "",
        "- **PRIMARY frozen-XI decision score:** predicted XI+C **%.3f**; actual frozen XI+C **%.1f**; error actual-predicted **%.3f**." % (ta["submitted_predicted_total_with_captain"], ta["primary_frozen_xi_actual_total"], ta["primary_actual_minus_predicted_total"]),
        "- Frozen XI raw points: **%.1f**; captain/vice bonus within frozen XI: **%.1f**." % (ta["frozen_starting_xi_actual_points_raw"], ta["frozen_xi_captain_bonus_actual_points"]),
        "- Secondary official-FPL autosub reference: **%s**; autosub status: `%s`." % (("%.1f" % ta["secondary_official_fpl_realized_total"]) if ta["secondary_official_fpl_realized_total"] is not None else "unavailable", ta["autosub_status"]),
        "- Captain: **%s**; vice: **%s**." % (ta["captain"], ta["vice_captain"]),
        "",
        "## Evaluation definitions",
        "",
        "- Primary player denominator = frozen rows with `selection_eligible=True` and `prediction_available=True`.",
        "- Top-K compares predicted Top-K vs actual Top-K within that frozen eligible universe; ties use deterministic FPL player-id ordering.",
        "- NDCG@20 uses `max(actual_points, 0)` as linear relevance and logarithmic rank discount.",
        "- Appearance/start accuracy thresholds frozen probabilities at 0.5; actual labels are `minutes > 0` and `starts > 0`.",
        "- Match Brier is multiclass sum-of-squared probability error averaged across fixtures.",
        "- Team PRIMARY headline score uses exactly the frozen starting XI; a 0-minute frozen starter contributes 0 and bench points never replace it in the primary decision score.",
        "- Captain/vice fallback is applied only within the XI being scored.",
        "- Automatic substitutions are reported only as a SECONDARY official-FPL realized-score reference when bench order is authoritative; they never change the primary frozen-XI evaluation.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    repo_root = detect_repo_root(args.repo_root)
    planning_root = Path(args.planning_root).expanduser().resolve() if args.planning_root else (repo_root.parent / "private-planning").resolve()
    season, gw = args.season, args.gw
    frozen_root = planning_root / "frozen-snapshots" / season / ("gw%02d" % gw)
    actual_manifest_path = Path(args.actual_manifest).expanduser().resolve() if args.actual_manifest else latest_actual_manifest(planning_root, season, gw)
    actual_manifest = read_json(actual_manifest_path)
    actual_dir = actual_manifest_path.parent
    event_path = actual_dir / actual_manifest["event_live_file"]
    fixtures_path = actual_dir / actual_manifest["fixtures_file"]

    # Actual evidence integrity.
    actual_hash_checks = {
        "event_live": {"expected": actual_manifest["event_live_sha256"], "actual": sha256_file(event_path)},
        "fixtures": {"expected": actual_manifest["fixtures_sha256"], "actual": sha256_file(fixtures_path)},
    }
    for key, row in actual_hash_checks.items():
        row["passed"] = row["expected"] == row["actual"]
        if not row["passed"]:
            raise RuntimeError("Actual evidence hash mismatch: %s" % key)

    final = bool(actual_manifest.get("all_fixtures_finished"))
    if args.require_final and not final:
        raise RuntimeError("--require-final requested but actual manifest is provisional")
    evaluation_status = "FINAL" if final else "PROVISIONAL"

    player_snapshot_path = discover_one(frozen_root, "player-model/**/global_player_prediction_snapshot.csv")
    model_team_path = discover_one(frozen_root, "model-team/**/model_team_snapshot.csv")
    match_preview_path = discover_one(frozen_root, "match-model/**/match_preview_csv.csv")
    scoreline_path = discover_one(frozen_root, "match-model/**/scoreline_csv.csv")
    freeze_manifest_candidates = sorted(frozen_root.glob("**/*FINAL_FREEZE_MANIFEST.json"))
    freeze_manifest_path = None
    freeze_manifest: Dict[str, Any] = {}
    for candidate in freeze_manifest_candidates:
        obj = read_json(candidate)
        if set(obj.get("tracks", {}).keys()) >= {"A_player_model", "B_match_model", "C_model_team", "D_team_alex"}:
            freeze_manifest_path = candidate
            freeze_manifest = obj
            break
    if freeze_manifest_path is None:
        raise RuntimeError("Could not locate complete final freeze manifest")

    # Frozen evidence integrity against durable hash list and freeze manifest.
    all_hashes = parse_sha256_sums(frozen_root / "SHA256SUMS_ALL.txt")
    frozen_hash_checks: Dict[str, Dict[str, Any]] = {}
    for label, path in {
        "player_snapshot": player_snapshot_path,
        "model_team_snapshot": model_team_path,
        "match_preview": match_preview_path,
        "scoreline": scoreline_path,
        "freeze_manifest": freeze_manifest_path,
    }.items():
        rel = path.relative_to(frozen_root).as_posix()
        actual_sha = sha256_file(path)
        expected_sha = all_hashes.get(rel)
        frozen_hash_checks[label] = {"path": rel, "expected": expected_sha, "actual": actual_sha, "passed": (expected_sha == actual_sha) if expected_sha else None}
        if expected_sha and expected_sha != actual_sha:
            raise RuntimeError("Frozen evidence hash mismatch: %s" % label)
    match_track = freeze_manifest["tracks"]["B_match_model"]
    if sha256_file(match_preview_path) != match_track["match_prediction_sha256"]:
        raise RuntimeError("Match preview hash does not match final freeze manifest")
    if sha256_file(scoreline_path) != match_track["scoreline_sha256"]:
        raise RuntimeError("Scoreline hash does not match final freeze manifest")

    frozen_players = read_csv(player_snapshot_path)
    actual_elements = read_json(event_path).get("elements", [])
    actual_by_id = {int(row["id"]): row.get("stats", {}) for row in actual_elements}
    frozen_ids = {i(r["fpl_player_id"]) for r in frozen_players}

    player_rows: List[Dict[str, Any]] = []
    missing_actual_ids: List[int] = []
    for r in frozen_players:
        pid = i(r["fpl_player_id"])
        stats = actual_by_id.get(pid)
        if stats is None:
            missing_actual_ids.append(pid)
            continue
        player_rows.append({
            "fpl_player_id": pid,
            "player_name": r["player_name"],
            "web_name": r["web_name"],
            "position": r["position"],
            "team_short_name": r["team_short_name"],
            "selection_eligible": boolish(r["selection_eligible"]),
            "prediction_available": boolish(r["prediction_available"]),
            "predicted_points": f(r["predicted_points"]),
            "expected_minutes": f(r["expected_minutes"]),
            "appearance_probability": f(r["appearance_probability"]),
            "start_probability": f(r["start_probability"]),
            "actual_points": i(stats.get("total_points", 0)),
            "actual_minutes": i(stats.get("minutes", 0)),
            "actual_starts": i(stats.get("starts", 0)),
            "actual_appearance": i(stats.get("minutes", 0)) > 0,
            "actual_start": i(stats.get("starts", 0)) > 0,
        })
    actual_only_ids = sorted(set(actual_by_id) - frozen_ids)
    if missing_actual_ids:
        raise RuntimeError("Frozen prediction rows missing actual records: %s" % missing_actual_ids)

    eligible = [r for r in player_rows if r["selection_eligible"] and r["prediction_available"]]
    played = [r for r in eligible if r["actual_minutes"] > 0]
    sixty = [r for r in eligible if r["actual_minutes"] >= 60]
    position_metrics: Dict[str, Any] = {}
    for pos in POSITIONS:
        subset = [r for r in eligible if r["position"] == pos]
        top5 = top_k_metrics(subset, 5)
        position_metrics[pos] = {"n": len(subset), "mae": point_metrics(subset)["mae"], "top_5_hits": top5["hits"], "top_5_precision": top5["precision_at_k"]}

    player_result = {
        "frozen_prediction_source": frozen_players[0].get("prediction_source") if frozen_players else None,
        "frozen_recommendation_status": frozen_players[0].get("recommendation_status") if frozen_players else None,
        "production_approved_true_rows": sum(1 for r in frozen_players if boolish(r.get("production_approved"))),
        "coverage": {
            "frozen_rows": len(frozen_players),
            "matched_frozen_rows": len(player_rows),
            "missing_actual_for_frozen_rows": len(missing_actual_ids),
            "official_actual_rows": len(actual_elements),
            "actual_only_rows": len(actual_only_ids),
            "actual_only_ids": actual_only_ids,
            "primary_eligible_rows": len(eligible),
        },
        "cohorts": {
            "all_eligible_players": point_metrics(eligible),
            "actually_played_players": point_metrics(played),
            "60_plus_minutes_players": point_metrics(sixty),
        },
        "top_k": {str(k): top_k_metrics(eligible, k) for k in (10, 20, 50)},
        "ranking_quality": {
            "spearman_rank_correlation": spearman([r["predicted_points"] for r in eligible], [r["actual_points"] for r in eligible]),
            "ndcg_at_20": ndcg_at_k(eligible, 20),
        },
        "position_level": position_metrics,
        "availability_and_minutes": {
            "appearance_accuracy_threshold_0_5": classification_accuracy(eligible, "appearance_probability", "actual_appearance"),
            "start_accuracy_threshold_0_5": classification_accuracy(eligible, "start_probability", "actual_start"),
            "minutes_mae": mae([r["expected_minutes"] for r in eligible], [r["actual_minutes"] for r in eligible]),
        },
    }

    # Match evaluation.
    match_rows = read_csv(match_preview_path)
    score_rows = read_csv(scoreline_path)
    score_by_fixture = {i(r["fpl_fixture_id"]): r for r in score_rows}
    actual_fixtures = {i(r["id"]): r for r in read_json(fixtures_path)}
    per_match: List[Dict[str, Any]] = []
    log_losses: List[float] = []
    briers: List[float] = []
    home_errors: List[float] = []
    away_errors: List[float] = []
    total_errors: List[float] = []
    correct = top1 = top3 = top5 = 0
    for r in match_rows:
        fid = i(r["fpl_fixture_id"])
        actual = actual_fixtures[fid]
        h, a = i(actual["team_h_score"]), i(actual["team_a_score"])
        label = result_label(h, a)
        probs = {"home_win": f(r["home_win_probability"]), "draw": f(r["draw_probability"]), "away_win": f(r["away_win_probability"])}
        p_actual = max(EPS, min(1.0, probs[label]))
        log_losses.append(-math.log(p_actual))
        y = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}; y[label] = 1.0
        briers.append(sum((probs[k] - y[k]) ** 2 for k in y))
        correct += int(r["predicted_result_label"] == label)
        sr = score_by_fixture[fid]
        eh, ea = f(sr["expected_home_goals"]), f(sr["expected_away_goals"])
        home_errors.append(abs(eh - h)); away_errors.append(abs(ea - a)); total_errors.append(abs((eh + ea) - (h + a)))
        actual_score = "%d-%d" % (h, a)
        tops = [sr["top_%d_scoreline" % k] for k in range(1, 6)]
        top1 += int(actual_score in tops[:1]); top3 += int(actual_score in tops[:3]); top5 += int(actual_score in tops[:5])
        per_match.append({
            "fixture_id": fid, "home_team": r["home_team_short_name"], "away_team": r["away_team_short_name"],
            "actual_home_goals": h, "actual_away_goals": a, "actual_result": label,
            "predicted_result": r["predicted_result_label"], "home_win_probability": probs["home_win"], "draw_probability": probs["draw"], "away_win_probability": probs["away_win"],
            "expected_home_goals": eh, "expected_away_goals": ea, "actual_scoreline": actual_score,
            "top_1_scoreline": tops[0], "top_2_scoreline": tops[1], "top_3_scoreline": tops[2], "top_4_scoreline": tops[3], "top_5_scoreline": tops[4],
            "top1_hit": actual_score in tops[:1], "top3_hit": actual_score in tops[:3], "top5_hit": actual_score in tops[:5],
        })
    nmatch = len(per_match)
    match_result = {
        "frozen_model_name": match_rows[0].get("model_name") if match_rows else None,
        "frozen_probability_type": match_rows[0].get("probability_type") if match_rows else None,
        "frozen_calibration_status": match_rows[0].get("calibration_status") if match_rows else None,
        "production_ready_true_rows": sum(1 for r in match_rows if boolish(r.get("production_ready"))),
        "n": nmatch,
        "one_x_two_accuracy": correct / nmatch,
        "log_loss": mean(log_losses),
        "multiclass_brier": mean(briers),
        "home_goals_mae": mean(home_errors),
        "away_goals_mae": mean(away_errors),
        "total_goals_mae": mean(total_errors),
        "exact_score_top1_accuracy": top1 / nmatch,
        "exact_score_top3_accuracy": top3 / nmatch,
        "exact_score_top5_accuracy": top5 / nmatch,
    }

    # Enrich frozen player lookup for team evaluation.
    eval_by_id = {r["fpl_player_id"]: r for r in player_rows}
    frozen_by_norm: Dict[str, List[Dict[str, Any]]] = {}
    for r in player_rows:
        for name in {r["web_name"], r["player_name"]}:
            frozen_by_norm.setdefault(normalize_name(name), []).append(r)

    model_team_csv = read_csv(model_team_path)
    model_squad: List[Dict[str, Any]] = []
    model_starters: List[Dict[str, Any]] = []
    model_bench: List[Dict[str, Any]] = []
    captain_id = vice_id = None
    bench_rank = {"bench_gk": 0, "bench_1": 1, "bench_2": 2, "bench_3": 3}
    for r in model_team_csv:
        pid = i(r["fpl_player_id"])
        base = dict(eval_by_id[pid])
        base["role"] = r["role"]
        base["predicted_points"] = f(r["gw1_predicted_points"])
        model_squad.append(base)
        if boolish(r["is_starter"]): model_starters.append(base)
        else: model_bench.append(base)
        if boolish(r["is_captain"]): captain_id = pid
        if boolish(r["is_vice_captain"]): vice_id = pid
    model_bench.sort(key=lambda r: bench_rank.get(r["role"], 99))
    if captain_id is None or vice_id is None:
        raise RuntimeError("Model Team captain/vice not found")
    model_team_result = evaluate_team("Model Team", model_squad, model_starters, model_bench, captain_id, vice_id, "model_team")

    # Team Alex frozen manifest names -> frozen ids.
    alex = freeze_manifest["tracks"]["D_team_alex"]
    def resolve_alias(name: str) -> Dict[str, Any]:
        matches = frozen_by_norm.get(normalize_name(name), [])
        unique = {r["fpl_player_id"]: r for r in matches}
        if len(unique) != 1:
            raise RuntimeError("Could not uniquely resolve Team Alex alias %r; candidates=%s" % (name, list(unique)))
        return dict(next(iter(unique.values())))
    alex_starters: List[Dict[str, Any]] = []
    for pos in POSITIONS:
        for name in alex["starting_xi"].get(pos, []):
            alex_starters.append(resolve_alias(name))
    alex_bench = [resolve_alias(alex["bench"]["GKP"])] + [resolve_alias(name) for name in alex["bench"]["outfield"]]
    alex_squad = alex_starters + alex_bench
    alex_c = resolve_alias(alex["captain"])["fpl_player_id"]
    alex_v = resolve_alias(alex["vice_captain"])["fpl_player_id"]
    team_alex_result = evaluate_team("Team Alex / Gliding Tiger", alex_squad, alex_starters, alex_bench, alex_c, alex_v, "team_alex")

    capture = str(actual_manifest["captured_at_utc"])
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = planning_root / "gw-post" / season / ("gw%02d" % gw) / "evaluation" / ("%s_%s" % (evaluation_status.lower(), capture))
    out_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "contract_version": "fpl_gameweek_post_evaluation_v1_1",
        "season": season,
        "gw": gw,
        "evaluation_status": evaluation_status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "leakage_safe": True,
        "database_access": False,
        "prediction_regeneration": False,
        "writes_frozen_pre_artifacts": False,
        "actuals": {**actual_manifest, "manifest_path": str(actual_manifest_path), "hash_checks": actual_hash_checks},
        "frozen_inputs": {"root": str(frozen_root), "hash_checks": frozen_hash_checks, "freeze_manifest": str(freeze_manifest_path)},
        "player_model": player_result,
        "match_model": match_result,
        "model_team": model_team_result,
        "team_alex": team_alex_result,
    }
    safe_result = json_safe(result)
    (out_dir / "evaluation_summary.json").write_text(json.dumps(safe_result, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    (out_dir / "evaluation_summary.md").write_text(build_markdown(result), encoding="utf-8")
    write_csv(out_dir / "player_evaluation_rows.csv", player_rows, [
        "fpl_player_id", "player_name", "web_name", "position", "team_short_name", "selection_eligible", "prediction_available", "predicted_points", "actual_points", "expected_minutes", "actual_minutes", "appearance_probability", "actual_appearance", "start_probability", "actual_start", "actual_starts"
    ])
    write_csv(out_dir / "match_evaluation_rows.csv", per_match, [
        "fixture_id", "home_team", "away_team", "actual_home_goals", "actual_away_goals", "actual_result", "predicted_result", "home_win_probability", "draw_probability", "away_win_probability", "expected_home_goals", "expected_away_goals", "actual_scoreline", "top_1_scoreline", "top_2_scoreline", "top_3_scoreline", "top_4_scoreline", "top_5_scoreline", "top1_hit", "top3_hit", "top5_hit"
    ])
    (out_dir / "model_team_evaluation.json").write_text(json.dumps(json_safe(model_team_result), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    (out_dir / "team_alex_evaluation.json").write_text(json.dumps(json_safe(team_alex_result), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    artifact_rows = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            artifact_rows.append({"name": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    eval_manifest = {
        "contract_version": "fpl_gameweek_post_evaluation_manifest_v1_1",
        "season": season, "gw": gw, "evaluation_status": evaluation_status, "actual_capture": capture,
        "output_dir": str(out_dir), "artifacts": artifact_rows,
        "final_freeze_allowed": final,
    }
    (out_dir / "evaluation_manifest.json").write_text(json.dumps(eval_manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("=== GW%d POST Evaluation ===" % gw)
    print("status:", evaluation_status)
    print("output_dir:", out_dir)
    print("frozen_player_rows:", len(frozen_players))
    print("matched_frozen_actual_rows:", len(player_rows))
    print("primary_eligible_rows:", len(eligible))
    print("actual_only_rows:", len(actual_only_ids))
    print("player_mae:", round(player_result["cohorts"]["all_eligible_players"]["mae"], 6))
    print("player_rmse:", round(player_result["cohorts"]["all_eligible_players"]["rmse"], 6))
    print("player_bias:", round(player_result["cohorts"]["all_eligible_players"]["mean_error_bias_pred_minus_actual"], 6))
    print("spearman:", round(player_result["ranking_quality"]["spearman_rank_correlation"], 6))
    print("ndcg_at_20:", round(player_result["ranking_quality"]["ndcg_at_20"], 6))
    print("match_1x2_accuracy:", round(match_result["one_x_two_accuracy"], 6))
    print("match_log_loss:", round(match_result["log_loss"], 6))
    print("match_brier:", round(match_result["multiclass_brier"], 6))
    print("model_team_primary_frozen_xi_total:", model_team_result["primary_frozen_xi_actual_total"])
    print("model_team_secondary_official_fpl_total:", model_team_result["secondary_official_fpl_realized_total"])
    print("team_alex_primary_frozen_xi_total:", team_alex_result["primary_frozen_xi_actual_total"])
    print("team_alex_secondary_official_fpl_total:", team_alex_result["secondary_official_fpl_realized_total"])
    print("final_freeze_allowed:", final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
