from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from app.rules.squad import load_squad_transfer_rules, validate_lineup, validate_squad
from ml.contracts.gameweek_cycle import build_phase_id
from ml.contracts.squad_state import (
    FreeTransferState,
    SquadPlayerState,
    SquadSelectionState,
    SquadState,
    SquadStateError,
    predecessor_reference,
    load_squad_state_json,
)


SNAPSHOT_VERSION = "fpl_gameweek_pre_deadline_snapshot_v1"
SNAPSHOT_KIND_CANDIDATE = "pre_deadline_candidate"
SNAPSHOT_KIND_FINAL = "final_pre_deadline"
STATUS_CANDIDATE = "PASS_CANDIDATE_SNAPSHOT"
STATUS_FINAL = "PASS_FINAL_FREEZE"
TRACK_NAMES: Tuple[str, ...] = (
    "player_model",
    "match_model",
    "model_team",
    "team_alex",
)


class GameweekPreDeadlineSnapshotError(ValueError):
    """Raised when a GW2+ pre-deadline evidence package is unsafe or invalid."""


def _parse_utc(value: Any, label: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise GameweekPreDeadlineSnapshotError("%s is required." % label)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise GameweekPreDeadlineSnapshotError(
            "%s must be ISO-8601." % label
        ) from exc
    if parsed.tzinfo is None:
        raise GameweekPreDeadlineSnapshotError(
            "%s must include a timezone." % label
        )
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp(value: Optional[datetime] = None) -> str:
    current = value or _utc_now()
    return current.strftime("%Y%m%dT%H%M%SZ")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    path = path.resolve()
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise GameweekPreDeadlineSnapshotError(
            "Artifact path does not exist: %s" % path
        )

    rows: List[Dict[str, str]] = []
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rows.append(
            {
                "path": child.relative_to(path).as_posix(),
                "sha256": _sha256_file(child),
            }
        )
    return _sha256_bytes(_canonical_json_bytes(rows))


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise GameweekPreDeadlineSnapshotError(
            "Refusing to overwrite existing artifact: %s" % path
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise GameweekPreDeadlineSnapshotError(
            "Refusing to overwrite existing artifact: %s" % path
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mapping(value: Any, label: str) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise GameweekPreDeadlineSnapshotError(
        "%s must be a mapping, dataclass, or to_dict()-compatible value." % label
    )


def _normalize_previous_state(value: Any) -> SquadState:
    if isinstance(value, SquadState):
        state = value
    elif isinstance(value, Mapping):
        from ml.contracts.squad_state import squad_state_from_mapping
        try:
            state = squad_state_from_mapping(value)
        except Exception as exc:
            raise GameweekPreDeadlineSnapshotError(
                "previous_model_team_state is not a valid canonical SquadState: %s"
                % exc
            ) from exc
    else:
        raise GameweekPreDeadlineSnapshotError(
            "previous_model_team_state must be a canonical SquadState or mapping."
        )

    if state.state_kind != "model_team":
        raise GameweekPreDeadlineSnapshotError(
            "previous_model_team_state must have state_kind=model_team."
        )
    if state.state_status != "frozen":
        raise GameweekPreDeadlineSnapshotError(
            "previous_model_team_state must be frozen."
        )
    return state


def _normalize_plan(value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GameweekPreDeadlineSnapshotError("chosen_plan must be a mapping.")
    if isinstance(value.get("winner"), Mapping):
        return dict(value["winner"])
    return dict(value)


def _validate_artifact_spec(
    value: Mapping[str, Any],
    *,
    label: str,
    expected_season: str,
    expected_gw: int,
    snapshot_as_of: datetime,
) -> Dict[str, Any]:
    spec = _mapping(value, label)
    required = ("run_id", "artifact_kind", "path", "season", "target_gw", "as_of_utc")
    missing = [key for key in required if spec.get(key) in (None, "")]
    if missing:
        raise GameweekPreDeadlineSnapshotError(
            "%s is missing required fields: %s" % (label, missing)
        )

    if str(spec["season"]) != expected_season:
        raise GameweekPreDeadlineSnapshotError(
            "%s season does not match target season." % label
        )
    if int(spec["target_gw"]) != int(expected_gw):
        raise GameweekPreDeadlineSnapshotError(
            "%s target_gw does not match target Gameweek." % label
        )

    source_as_of = _parse_utc(spec["as_of_utc"], "%s.as_of_utc" % label)
    if source_as_of > snapshot_as_of:
        raise GameweekPreDeadlineSnapshotError(
            "%s was produced after snapshot as-of time." % label
        )

    source_path = Path(str(spec["path"])).expanduser().resolve()
    if not source_path.exists():
        raise GameweekPreDeadlineSnapshotError(
            "%s path does not exist: %s" % (label, source_path)
        )

    observed_hash = _sha256_path(source_path)
    expected_hash = spec.get("sha256")
    if expected_hash not in (None, "") and str(expected_hash) != observed_hash:
        raise GameweekPreDeadlineSnapshotError(
            "%s sha256 does not match current source artifact." % label
        )

    return {
        "run_id": str(spec["run_id"]),
        "artifact_kind": str(spec["artifact_kind"]),
        "path": str(source_path),
        "season": expected_season,
        "target_gw": int(expected_gw),
        "as_of_utc": _format_utc(source_as_of),
        "sha256": observed_hash,
        "source_is_directory": source_path.is_dir(),
    }


def _copy_artifact_immutable(source_path: Path, destination: Path) -> None:
    if destination.exists():
        raise GameweekPreDeadlineSnapshotError(
            "Refusing to overwrite copied artifact: %s" % destination
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.is_dir():
        shutil.copytree(str(source_path), str(destination))
    else:
        shutil.copy2(str(source_path), str(destination))


def _player_rule_row(player: SquadPlayerState) -> Dict[str, Any]:
    return {
        "player_id": int(player.fpl_player_id),
        "position": str(player.position),
        "club_id": int(player.club_id),
        "price_units": int(player.current_price_units),
    }


def _build_target_model_team_state(
    *,
    previous: SquadState,
    chosen_plan: Mapping[str, Any],
    season: str,
    target_gw: int,
    as_of_utc: str,
    run_id: str,
    final_freeze: bool,
) -> SquadState:
    plan = dict(chosen_plan)
    transfers = list(plan.get("transfers") or [])
    if int(plan.get("transfer_count", len(transfers))) != len(transfers):
        raise GameweekPreDeadlineSnapshotError(
            "chosen_plan transfer_count does not match transfer rows."
        )

    if int(plan.get("bank_before_units", previous.bank_units)) != int(previous.bank_units):
        raise GameweekPreDeadlineSnapshotError(
            "chosen_plan bank_before_units does not match previous frozen state."
        )

    players: Dict[int, SquadPlayerState] = {
        int(player.fpl_player_id): player for player in previous.players
    }

    for index, transfer in enumerate(transfers):
        if not isinstance(transfer, Mapping):
            raise GameweekPreDeadlineSnapshotError(
                "chosen_plan.transfers[%s] must be a mapping." % index
            )
        try:
            out_id = int(transfer["out_fpl_player_id"])
            in_id = int(transfer["in_fpl_player_id"])
            position = str(transfer["in_position"])
            club_id = int(transfer["in_club_id"])
            price_units = int(transfer["in_price_units"])
        except (KeyError, TypeError, ValueError) as exc:
            raise GameweekPreDeadlineSnapshotError(
                "chosen_plan transfer row is missing required normalized fields."
            ) from exc

        if out_id not in players:
            raise GameweekPreDeadlineSnapshotError(
                "Outgoing player_id=%s is not owned in previous frozen state." % out_id
            )
        if in_id in players:
            raise GameweekPreDeadlineSnapshotError(
                "Incoming player_id=%s is already owned." % in_id
            )
        outgoing = players[out_id]
        if str(outgoing.position) != position:
            raise GameweekPreDeadlineSnapshotError(
                "Transfer is not position-compatible for outgoing player_id=%s."
                % out_id
            )

        players.pop(out_id)
        players[in_id] = SquadPlayerState(
            fpl_player_id=in_id,
            player_name=transfer.get("in_name"),
            position=position,
            club_id=club_id,
            purchase_price_units=price_units,
            current_price_units=price_units,
            selling_price_units=price_units,
        )

    lineup = plan.get("lineup")
    if not isinstance(lineup, Mapping):
        raise GameweekPreDeadlineSnapshotError(
            "chosen_plan must contain the re-optimized lineup."
        )

    try:
        starting_ids = tuple(int(v) for v in lineup["starting_player_ids"])
        bench_ids = tuple(int(v) for v in lineup["bench_order"])
        captain_id = int(lineup["captain_player_id"])
        vice_id = int(lineup["vice_captain_player_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GameweekPreDeadlineSnapshotError(
            "chosen_plan lineup is missing XI/bench/captain/vice fields."
        ) from exc

    bank_after = int(plan.get("bank_after_units"))
    ft_next = int(plan.get("free_transfers_next_gameweek"))

    try:
        selection = SquadSelectionState(
            starting_xi_player_ids=starting_ids,
            bench_order_player_ids=bench_ids,
            captain_player_id=captain_id,
            vice_captain_player_id=vice_id,
        )
        target_state = SquadState(
            season=season,
            gameweek=target_gw,
            as_of_utc=as_of_utc,
            state_version="%s_%s" % (SNAPSHOT_VERSION, run_id),
            state_kind="model_team",
            state_status="frozen" if final_freeze else "planning",
            source_phase_id=build_phase_id(
                target_gw, "freeze" if final_freeze else "pre"
            ),
            source_run_id=run_id,
            players=tuple(
                sorted(
                    players.values(),
                    key=lambda player: int(player.fpl_player_id),
                )
            ),
            selection=selection,
            bank_units=bank_after,
            chip_inventory=previous.chip_inventory,
            free_transfers=FreeTransferState(
                available_for_gameweek=target_gw + 1,
                count=ft_next,
            ),
            predecessor=predecessor_reference(previous),
            shadow_optimal=None,
        )
    except SquadStateError as exc:
        raise GameweekPreDeadlineSnapshotError(
            "Chosen plan cannot form a valid target Model Team state: %s" % exc
        ) from exc

    rules = load_squad_transfer_rules(season)
    structural = validate_squad(
        rules,
        [_player_rule_row(player) for player in target_state.players],
        budget_limit_units=(
            sum(int(player.current_price_units) for player in target_state.players)
            + int(target_state.bank_units)
        ),
    )
    if not structural["valid"]:
        raise GameweekPreDeadlineSnapshotError(
            "Target Model Team is structurally illegal: %s" % structural["errors"]
        )

    lineup_report = validate_lineup(
        rules,
        [_player_rule_row(player) for player in target_state.players],
        starting_ids,
        bench_ids,
        captain_id,
        vice_id,
    )
    if not lineup_report["valid"]:
        raise GameweekPreDeadlineSnapshotError(
            "Target Model Team lineup is illegal: %s" % lineup_report["errors"]
        )

    return target_state


def _team_alex_payload(
    value: Optional[Mapping[str, Any]],
    *,
    season: str,
    target_gw: int,
    snapshot_as_of: datetime,
) -> Dict[str, Any]:
    if value is None:
        return {
            "provided": False,
            "state_kind": "team_alex",
            "model_team_logic_consumed_team_alex": False,
        }

    payload = _mapping(value, "team_alex_reference")
    if payload.get("season") not in (None, season):
        raise GameweekPreDeadlineSnapshotError(
            "Team Alex reference season does not match."
        )
    if payload.get("target_gw") not in (None, target_gw):
        raise GameweekPreDeadlineSnapshotError(
            "Team Alex reference target_gw does not match."
        )
    if payload.get("as_of_utc") not in (None, ""):
        alex_as_of = _parse_utc(
            payload["as_of_utc"], "team_alex_reference.as_of_utc"
        )
        if alex_as_of > snapshot_as_of:
            raise GameweekPreDeadlineSnapshotError(
                "Team Alex reference was produced after snapshot as-of time."
            )

    result = dict(payload)
    result["provided"] = True
    result["state_kind"] = "team_alex"
    result["model_team_logic_consumed_team_alex"] = False

    if result.get("path"):
        alex_path = Path(str(result["path"])).expanduser().resolve()
        if not alex_path.exists():
            raise GameweekPreDeadlineSnapshotError(
                "Team Alex reference path does not exist: %s" % alex_path
            )
        observed_hash = _sha256_path(alex_path)
        if result.get("sha256") not in (None, ""):
            if str(result["sha256"]) != observed_hash:
                raise GameweekPreDeadlineSnapshotError(
                    "Team Alex reference sha256 mismatch."
                )
        result["path"] = str(alex_path)
        result["sha256"] = observed_hash
        result["source_is_directory"] = alex_path.is_dir()

    return result


def _snapshot_run_id(
    *,
    season: str,
    target_gw: int,
    final_freeze: bool,
    as_of: datetime,
) -> str:
    mode = "final_freeze" if final_freeze else "candidate"
    return "gameweek_pre_deadline_snapshot_%s_gw%s_%s_%s_%s" % (
        season,
        target_gw,
        mode,
        _utc_stamp(as_of),
        uuid.uuid4().hex[:8],
    )


def export_gameweek_pre_deadline_snapshot(
    *,
    artifact_root: Path,
    season: str,
    target_gw: int,
    as_of_time: str,
    fpl_deadline_time: str,
    player_model_artifact: Mapping[str, Any],
    match_model_artifact: Mapping[str, Any],
    previous_model_team_state: Any,
    chosen_plan: Mapping[str, Any],
    transfer_ledger_state: Any,
    team_alex_reference: Optional[Mapping[str, Any]] = None,
    final_freeze: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Write immutable GW2+ four-track pre-deadline evidence.

    Candidate and FINAL outputs are both append-only.  FINAL requires the
    explicit final_freeze=True call; a candidate export never silently promotes.
    """

    season = str(season)
    target_gw = int(target_gw)
    if target_gw < 2:
        raise GameweekPreDeadlineSnapshotError(
            "Day127B is the reusable GW2+ exporter; target_gw must be >= 2."
        )

    as_of = _parse_utc(as_of_time, "as_of_time")
    deadline = _parse_utc(fpl_deadline_time, "fpl_deadline_time")
    if as_of >= deadline:
        raise GameweekPreDeadlineSnapshotError(
            "Snapshot as_of_time must be strictly before the FPL deadline."
        )

    previous = _normalize_previous_state(previous_model_team_state)
    if previous.season != season:
        raise GameweekPreDeadlineSnapshotError(
            "Previous Model Team season does not match target season."
        )
    if int(previous.gameweek) != target_gw - 1:
        raise GameweekPreDeadlineSnapshotError(
            "Previous frozen Model Team must be from target_gw - 1."
        )

    plan = _normalize_plan(chosen_plan)
    ledger = _mapping(transfer_ledger_state, "transfer_ledger_state")
    if str(ledger.get("season")) != season:
        raise GameweekPreDeadlineSnapshotError(
            "Transfer ledger season does not match target season."
        )
    if str(ledger.get("state_kind")) != "model_team":
        raise GameweekPreDeadlineSnapshotError(
            "Transfer ledger state_kind must be model_team."
        )
    if int(ledger.get("gameweek") or -1) != target_gw:
        raise GameweekPreDeadlineSnapshotError(
            "Transfer ledger gameweek must match target_gw."
        )

    ledger_ft = ledger.get("available_free_transfers")
    if ledger_ft is None:
        ledger_ft = ledger.get("free_transfers")
    if ledger_ft is None:
        raise GameweekPreDeadlineSnapshotError(
            "Transfer ledger must expose available_free_transfers."
        )
    if int(plan.get("free_transfers_before")) != int(ledger_ft):
        raise GameweekPreDeadlineSnapshotError(
            "Chosen plan FT-before count does not match transfer ledger."
        )

    player_spec = _validate_artifact_spec(
        player_model_artifact,
        label="player_model_artifact",
        expected_season=season,
        expected_gw=target_gw,
        snapshot_as_of=as_of,
    )
    match_spec = _validate_artifact_spec(
        match_model_artifact,
        label="match_model_artifact",
        expected_season=season,
        expected_gw=target_gw,
        snapshot_as_of=as_of,
    )
    alex_payload = _team_alex_payload(
        team_alex_reference,
        season=season,
        target_gw=target_gw,
        snapshot_as_of=as_of,
    )

    resolved_run_id = str(
        run_id
        or _snapshot_run_id(
            season=season,
            target_gw=target_gw,
            final_freeze=final_freeze,
            as_of=as_of,
        )
    )
    if not resolved_run_id.strip():
        raise GameweekPreDeadlineSnapshotError("run_id must not be empty.")

    snapshot_kind = (
        SNAPSHOT_KIND_FINAL if final_freeze else SNAPSHOT_KIND_CANDIDATE
    )
    mode_dir = "final" if final_freeze else "candidate"
    package_dir = (
        Path(artifact_root).expanduser().resolve()
        / season
        / ("gw%02d" % target_gw)
        / mode_dir
        / resolved_run_id
    )
    if package_dir.exists():
        raise GameweekPreDeadlineSnapshotError(
            "Snapshot already exists; refusing overwrite: %s" % package_dir
        )
    package_dir.mkdir(parents=True, exist_ok=False)

    try:
        target_state = _build_target_model_team_state(
            previous=previous,
            chosen_plan=plan,
            season=season,
            target_gw=target_gw,
            as_of_utc=_format_utc(as_of),
            run_id=resolved_run_id,
            final_freeze=final_freeze,
        )

        # Track A: Player Model
        player_track_dir = package_dir / "tracks" / "player_model"
        player_source = Path(player_spec["path"])
        player_copy = player_track_dir / "source"
        _copy_artifact_immutable(player_source, player_copy)
        _write_json_new(
            player_track_dir / "reference.json",
            {
                **player_spec,
                "snapshot_copy_path": str(player_copy.relative_to(package_dir)),
                "snapshot_copy_sha256": _sha256_path(player_copy),
                "immutable": True,
            },
        )

        # Track B: Match Model
        match_track_dir = package_dir / "tracks" / "match_model"
        match_source = Path(match_spec["path"])
        match_copy = match_track_dir / "source"
        _copy_artifact_immutable(match_source, match_copy)
        _write_json_new(
            match_track_dir / "reference.json",
            {
                **match_spec,
                "snapshot_copy_path": str(match_copy.relative_to(package_dir)),
                "snapshot_copy_sha256": _sha256_path(match_copy),
                "immutable": True,
            },
        )

        # Track C: Model Team
        model_team_dir = package_dir / "tracks" / "model_team"
        model_team_dir.mkdir(parents=True, exist_ok=False)
        target_state_path = model_team_dir / "model_team_state.json"
        _write_json_new(target_state_path, target_state.to_dict())
        _write_json_new(model_team_dir / "chosen_plan.json", plan)
        _write_json_new(model_team_dir / "transfer_ledger_state.json", ledger)

        # Track D: Team Alex, always structurally separate.
        alex_dir = package_dir / "tracks" / "team_alex"
        alex_dir.mkdir(parents=True, exist_ok=False)
        alex_reference_to_write = dict(alex_payload)
        if alex_payload.get("provided") and alex_payload.get("path"):
            source = Path(str(alex_payload["path"]))
            alex_copy = alex_dir / "source"
            _copy_artifact_immutable(source, alex_copy)
            alex_reference_to_write["snapshot_copy_path"] = str(
                alex_copy.relative_to(package_dir)
            )
            alex_reference_to_write["snapshot_copy_sha256"] = _sha256_path(
                alex_copy
            )
        _write_json_new(alex_dir / "reference.json", alex_reference_to_write)

        predecessor = target_state.predecessor
        previous_fingerprint = (
            predecessor.owned_state_fingerprint if predecessor is not None else None
        )

        track_summary = {
            "player_model": {
                "path": "tracks/player_model",
                "source_run_id": player_spec["run_id"],
                "sha256": _sha256_path(player_track_dir),
                "immutable": True,
            },
            "match_model": {
                "path": "tracks/match_model",
                "source_run_id": match_spec["run_id"],
                "sha256": _sha256_path(match_track_dir),
                "immutable": True,
            },
            "model_team": {
                "path": "tracks/model_team",
                "source_run_id": resolved_run_id,
                "sha256": _sha256_path(model_team_dir),
                "immutable": True,
            },
            "team_alex": {
                "path": "tracks/team_alex",
                "source_run_id": alex_payload.get("run_id"),
                "sha256": _sha256_path(alex_dir),
                "immutable": True,
                "provided": bool(alex_payload.get("provided")),
            },
        }

        manifest_base: Dict[str, Any] = {
            "snapshot_version": SNAPSHOT_VERSION,
            "status": STATUS_FINAL if final_freeze else STATUS_CANDIDATE,
            "snapshot_kind": snapshot_kind,
            "final_pre_deadline_snapshot_frozen": bool(final_freeze),
            "explicit_final_freeze_mode": bool(final_freeze),
            "season": season,
            "target_gw": target_gw,
            "run_id": resolved_run_id,
            "as_of_utc": _format_utc(as_of),
            "fpl_deadline_utc": _format_utc(deadline),
            "as_of_before_deadline": True,
            "tracks": track_summary,
            "model_team": {
                "previous_state_id": previous.state_id,
                "previous_owned_state_fingerprint": previous_fingerprint,
                "previous_gameweek": int(previous.gameweek),
                "target_state_id": target_state.state_id,
                "target_owned_state_fingerprint": target_state.owned_state_fingerprint,
                "state_status": target_state.state_status,
                "bank_before_units": int(previous.bank_units),
                "bank_after_units": int(target_state.bank_units),
                "free_transfers_before": int(plan["free_transfers_before"]),
                "free_transfers_used": int(plan["free_transfers_used"]),
                "charged_transfers": int(plan["charged_transfers"]),
                "transfer_hit_points": int(plan["transfer_hit_points"]),
                "free_transfers_next_gameweek": int(
                    plan["free_transfers_next_gameweek"]
                ),
                "action": str(plan["action"]),
                "transfer_count": int(plan["transfer_count"]),
                "lineup": dict(plan["lineup"]),
            },
            "transfer_ledger_state": ledger,
            "team_alex": {
                "provided": bool(alex_payload.get("provided")),
                "reference_run_id": alex_payload.get("run_id"),
                "model_team_logic_consumed_team_alex": False,
            },
            "safety": {
                "writes_database": False,
                "writes_manager_state": False,
                "writes_predictions": False,
                "opening_squad_optimizer_used": False,
                "team_alex_fed_into_model_team_logic": False,
                "target_gw_actuals_consumed": False,
                "candidate_auto_promoted_to_final": False,
                "overwrite_allowed": False,
                "post_deadline_reconstruction_allowed": False,
            },
        }
        manifest_base["package_fingerprint"] = _sha256_bytes(
            _canonical_json_bytes(manifest_base)
        )

        _write_json_new(package_dir / "snapshot_manifest.json", manifest_base)

        checksum_rows: List[str] = []
        for child in sorted(p for p in package_dir.rglob("*") if p.is_file()):
            if child.name == "SHA256SUMS.txt":
                continue
            checksum_rows.append(
                "%s  %s"
                % (
                    _sha256_file(child),
                    child.relative_to(package_dir).as_posix(),
                )
            )
        _write_text_new(
            package_dir / "SHA256SUMS.txt",
            "\n".join(checksum_rows) + "\n",
        )

        return {
            "status": manifest_base["status"],
            "snapshot_kind": snapshot_kind,
            "final_pre_deadline_snapshot_frozen": bool(final_freeze),
            "run_id": resolved_run_id,
            "snapshot_dir": str(package_dir),
            "manifest_path": str(package_dir / "snapshot_manifest.json"),
            "package_fingerprint": manifest_base["package_fingerprint"],
            "track_count": len(TRACK_NAMES),
            "tracks": track_summary,
            "model_team_state_path": str(target_state_path),
            "previous_owned_state_fingerprint": previous_fingerprint,
            "bank_after_units": int(target_state.bank_units),
            "free_transfers_next_gameweek": int(
                plan["free_transfers_next_gameweek"]
            ),
            "team_alex_provided": bool(alex_payload.get("provided")),
        }
    except Exception:
        # A failed export must not leave a misleading partial snapshot package.
        if package_dir.exists():
            shutil.rmtree(str(package_dir))
        raise


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise GameweekPreDeadlineSnapshotError(
            "%s file does not exist: %s" % (label, path)
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GameweekPreDeadlineSnapshotError(
            "Could not parse %s JSON: %s" % (label, path)
        ) from exc
    if not isinstance(value, Mapping):
        raise GameweekPreDeadlineSnapshotError(
            "%s JSON must contain an object." % label
        )
    return dict(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export immutable GW2+ pre-deadline evidence."
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--season", required=True)
    parser.add_argument("--target-gw", required=True, type=int)
    parser.add_argument("--as-of-time", required=True)
    parser.add_argument("--deadline-time", required=True)
    parser.add_argument("--player-model-spec-json", required=True)
    parser.add_argument("--match-model-spec-json", required=True)
    parser.add_argument("--previous-model-team-state-json", required=True)
    parser.add_argument("--chosen-plan-json", required=True)
    parser.add_argument("--transfer-ledger-json", required=True)
    parser.add_argument("--team-alex-reference-json")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--final-freeze",
        action="store_true",
        help="Explicitly create FINAL pre-deadline freeze rather than candidate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    team_alex = (
        _load_json(
            Path(args.team_alex_reference_json),
            "team_alex_reference",
        )
        if args.team_alex_reference_json
        else None
    )
    result = export_gameweek_pre_deadline_snapshot(
        artifact_root=Path(args.artifact_root),
        season=args.season,
        target_gw=args.target_gw,
        as_of_time=args.as_of_time,
        fpl_deadline_time=args.deadline_time,
        player_model_artifact=_load_json(
            Path(args.player_model_spec_json), "player_model_spec"
        ),
        match_model_artifact=_load_json(
            Path(args.match_model_spec_json), "match_model_spec"
        ),
        previous_model_team_state=load_squad_state_json(
            Path(args.previous_model_team_state_json)
        ),
        chosen_plan=_load_json(Path(args.chosen_plan_json), "chosen_plan"),
        transfer_ledger_state=_load_json(
            Path(args.transfer_ledger_json), "transfer_ledger"
        ),
        team_alex_reference=team_alex,
        final_freeze=bool(args.final_freeze),
        run_id=args.run_id,
    )
    print("status:", result["status"])
    print("snapshot_kind:", result["snapshot_kind"])
    print(
        "final_pre_deadline_snapshot_frozen:",
        result["final_pre_deadline_snapshot_frozen"],
    )
    print("run_id:", result["run_id"])
    print("snapshot_dir:", result["snapshot_dir"])
    print("manifest_path:", result["manifest_path"])


if __name__ == "__main__":
    main()
