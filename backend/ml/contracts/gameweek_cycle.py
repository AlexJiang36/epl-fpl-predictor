from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


CONTRACT_VERSION = "fpl_gameweek_cycle_v1"


class GameweekCycleError(ValueError):
    """Raised when lifecycle state, evidence, or a transition violates the contract."""


class GameweekPhase(str, Enum):
    PRE = "pre"
    FREEZE = "freeze"
    POST = "post"


class ActualsStatus(str, Enum):
    NONE = "none"
    PROVISIONAL = "provisional"
    FINAL = "final"


class ArtifactTrack(str, Enum):
    PLAYER_MODEL = "player_model"
    MATCH_MODEL = "match_model"
    MODEL_TEAM = "model_team"
    TEAM_ALEX = "team_alex"


INPUT_HISTORICAL_PRIORS = "historical_priors"
INPUT_PRIOR_GW_ACTUALS = "prior_gameweek_actuals"
INPUT_TARGET_FIXTURES = "target_gameweek_fixtures"
INPUT_LIVE_PLAYER_STATE = "live_player_state"
INPUT_CURRENT_PRICES = "current_prices"
INPUT_PREVIOUS_OWNED_SQUAD = "previous_owned_squad"
INPUT_PRE_ARTIFACTS = "pre_artifacts"
INPUT_FROZEN_ARTIFACTS = "frozen_artifacts"
INPUT_TARGET_GW_ACTUALS = "target_gameweek_actuals"

OUTPUT_PHASE_MANIFEST = "phase_manifest"
OUTPUT_PRE_PUBLISH_RECEIPT = "pre_publish_receipt"
OUTPUT_FREEZE_MANIFEST = "freeze_manifest"
OUTPUT_POST_EVALUATION_MANIFEST = "post_evaluation_manifest"


PRE_REQUIRED_ARTIFACTS: Mapping[str, str] = {
    ArtifactTrack.PLAYER_MODEL.value: "player_predictions_pre",
    ArtifactTrack.MATCH_MODEL.value: "match_predictions_pre",
    ArtifactTrack.MODEL_TEAM.value: "model_team_decision_pre",
    ArtifactTrack.TEAM_ALEX.value: "team_alex_decision_pre",
}

FREEZE_REQUIRED_ARTIFACTS: Mapping[str, str] = {
    ArtifactTrack.PLAYER_MODEL.value: "player_predictions_final",
    ArtifactTrack.MATCH_MODEL.value: "match_predictions_final",
    ArtifactTrack.MODEL_TEAM.value: "model_team_final",
    ArtifactTrack.TEAM_ALEX.value: "team_alex_final",
}

POST_REQUIRED_ARTIFACTS: Mapping[str, str] = {
    ArtifactTrack.PLAYER_MODEL.value: "player_model_evaluation",
    ArtifactTrack.MATCH_MODEL.value: "match_model_evaluation",
    ArtifactTrack.MODEL_TEAM.value: "model_team_evaluation",
    ArtifactTrack.TEAM_ALEX.value: "team_alex_evaluation",
}


@dataclass(frozen=True)
class PhaseContract:
    phase: str
    allowed_inputs: Tuple[str, ...]
    prohibited_inputs: Tuple[str, ...]
    allowed_outputs: Tuple[str, ...]
    required_artifacts: Mapping[str, str]
    target_gw_actuals_allowed: bool
    target_gw_actuals_required: bool
    safe_rerun_allowed: bool
    outputs_immutable: bool

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["required_artifacts"] = dict(self.required_artifacts)
        return payload


PHASE_CONTRACTS: Mapping[str, PhaseContract] = {
    GameweekPhase.PRE.value: PhaseContract(
        phase=GameweekPhase.PRE.value,
        allowed_inputs=(
            INPUT_HISTORICAL_PRIORS,
            INPUT_PRIOR_GW_ACTUALS,
            INPUT_TARGET_FIXTURES,
            INPUT_LIVE_PLAYER_STATE,
            INPUT_CURRENT_PRICES,
            INPUT_PREVIOUS_OWNED_SQUAD,
        ),
        prohibited_inputs=(INPUT_TARGET_GW_ACTUALS, INPUT_FROZEN_ARTIFACTS),
        allowed_outputs=(
            *tuple(PRE_REQUIRED_ARTIFACTS.values()),
            OUTPUT_PHASE_MANIFEST,
            OUTPUT_PRE_PUBLISH_RECEIPT,
        ),
        required_artifacts=PRE_REQUIRED_ARTIFACTS,
        target_gw_actuals_allowed=False,
        target_gw_actuals_required=False,
        safe_rerun_allowed=True,
        outputs_immutable=False,
    ),
    GameweekPhase.FREEZE.value: PhaseContract(
        phase=GameweekPhase.FREEZE.value,
        allowed_inputs=(INPUT_PRE_ARTIFACTS,),
        prohibited_inputs=(INPUT_TARGET_GW_ACTUALS,),
        allowed_outputs=(
            *tuple(FREEZE_REQUIRED_ARTIFACTS.values()),
            OUTPUT_PHASE_MANIFEST,
            OUTPUT_FREEZE_MANIFEST,
        ),
        required_artifacts=FREEZE_REQUIRED_ARTIFACTS,
        target_gw_actuals_allowed=False,
        target_gw_actuals_required=False,
        safe_rerun_allowed=False,
        outputs_immutable=True,
    ),
    GameweekPhase.POST.value: PhaseContract(
        phase=GameweekPhase.POST.value,
        allowed_inputs=(INPUT_FROZEN_ARTIFACTS, INPUT_TARGET_GW_ACTUALS),
        prohibited_inputs=(INPUT_PRE_ARTIFACTS,),
        allowed_outputs=(
            *tuple(POST_REQUIRED_ARTIFACTS.values()),
            OUTPUT_PHASE_MANIFEST,
            OUTPUT_POST_EVALUATION_MANIFEST,
        ),
        required_artifacts=POST_REQUIRED_ARTIFACTS,
        target_gw_actuals_allowed=True,
        target_gw_actuals_required=True,
        safe_rerun_allowed=True,
        outputs_immutable=False,
    ),
}


ACTUALS_STATUS_ORDER = {
    ActualsStatus.NONE.value: 0,
    ActualsStatus.PROVISIONAL.value: 1,
    ActualsStatus.FINAL.value: 2,
}


@dataclass(frozen=True)
class GameweekCycleState:
    season: str
    target_gw: int
    phase: str
    target_actuals_status: str = ActualsStatus.NONE.value
    freeze_run_id: Optional[str] = None
    freeze_fingerprint: Optional[str] = None
    post_eval_run_id: Optional[str] = None
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        season = _nonempty_text(self.season, "season")
        target_gw = _positive_int(self.target_gw, "target_gw")
        phase = normalize_phase(self.phase)
        actuals_status = normalize_actuals_status(self.target_actuals_status)

        object.__setattr__(self, "season", season)
        object.__setattr__(self, "target_gw", target_gw)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "target_actuals_status", actuals_status)

        if self.contract_version != CONTRACT_VERSION:
            raise GameweekCycleError(
                "contract_version=%s is unsupported; expected %s."
                % (self.contract_version, CONTRACT_VERSION)
            )

        if phase in (GameweekPhase.PRE.value, GameweekPhase.FREEZE.value):
            if actuals_status != ActualsStatus.NONE.value:
                raise GameweekCycleError(
                    "%s state cannot contain target-GW actuals." % phase
                )

        if phase == GameweekPhase.PRE.value:
            if self.freeze_run_id is not None or self.freeze_fingerprint is not None:
                raise GameweekCycleError("PRE state cannot claim a final freeze.")
            if self.post_eval_run_id is not None:
                raise GameweekCycleError("PRE state cannot claim a POST evaluation.")

        if phase in (GameweekPhase.FREEZE.value, GameweekPhase.POST.value):
            _nonempty_text(self.freeze_run_id, "freeze_run_id")
            _nonempty_text(self.freeze_fingerprint, "freeze_fingerprint")

        if phase == GameweekPhase.FREEZE.value and self.post_eval_run_id is not None:
            raise GameweekCycleError("FREEZE state cannot claim a POST evaluation.")

        if phase == GameweekPhase.POST.value:
            if actuals_status == ActualsStatus.NONE.value:
                raise GameweekCycleError(
                    "POST state requires provisional or final target-GW actuals."
                )
            _nonempty_text(self.post_eval_run_id, "post_eval_run_id")

    @property
    def phase_id(self) -> str:
        return build_phase_id(self.target_gw, self.phase)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["phase_id"] = self.phase_id
        return payload


@dataclass(frozen=True)
class LifecycleArtifactRef:
    track: str
    phase: str
    artifact_kind: str
    cycle_phase_id: str
    artifact_run_id: str
    artifact_path: str
    sha256: Optional[str] = None
    immutable: bool = False
    source_freeze_run_id: Optional[str] = None
    source_freeze_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        track = normalize_track(self.track)
        phase = normalize_phase(self.phase)
        object.__setattr__(self, "track", track)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "artifact_kind", _nonempty_text(self.artifact_kind, "artifact_kind"))
        object.__setattr__(self, "cycle_phase_id", _nonempty_text(self.cycle_phase_id, "cycle_phase_id"))
        object.__setattr__(self, "artifact_run_id", _nonempty_text(self.artifact_run_id, "artifact_run_id"))
        object.__setattr__(self, "artifact_path", _nonempty_text(self.artifact_path, "artifact_path"))
        if self.sha256 is not None:
            object.__setattr__(self, "sha256", _nonempty_text(self.sha256, "sha256"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GameweekCycleError("%s must be a non-empty string." % label)
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GameweekCycleError("%s must be a positive integer." % label)
    return int(value)


def normalize_phase(value: Any) -> str:
    if isinstance(value, GameweekPhase):
        return value.value
    raw = _nonempty_text(value, "phase").lower()
    if raw not in PHASE_CONTRACTS:
        raise GameweekCycleError(
            "phase=%s is invalid; expected one of %s."
            % (raw, tuple(PHASE_CONTRACTS.keys()))
        )
    return raw


def normalize_actuals_status(value: Any) -> str:
    if isinstance(value, ActualsStatus):
        return value.value
    raw = _nonempty_text(value, "target_actuals_status").lower()
    if raw not in ACTUALS_STATUS_ORDER:
        raise GameweekCycleError(
            "target_actuals_status=%s is invalid; expected one of %s."
            % (raw, tuple(ACTUALS_STATUS_ORDER.keys()))
        )
    return raw


def normalize_track(value: Any) -> str:
    if isinstance(value, ArtifactTrack):
        return value.value
    raw = _nonempty_text(value, "track").lower()
    valid = tuple(track.value for track in ArtifactTrack)
    if raw not in valid:
        raise GameweekCycleError(
            "track=%s is invalid; expected one of %s." % (raw, valid)
        )
    return raw


def get_phase_contract(phase: Any) -> PhaseContract:
    return PHASE_CONTRACTS[normalize_phase(phase)]


def build_phase_id(target_gw: int, phase: Any) -> str:
    gw = _positive_int(target_gw, "target_gw")
    normalized = normalize_phase(phase)
    if normalized == GameweekPhase.PRE.value:
        suffix = "PRE"
    elif normalized == GameweekPhase.FREEZE.value:
        suffix = "FREEZE"
    else:
        suffix = "POST-EVAL"
    return "GW%02d-%s" % (gw, suffix)


def build_execution_run_id(target_gw: int, phase: Any, unique_suffix: str) -> str:
    suffix = _nonempty_text(unique_suffix, "unique_suffix")
    return "%s-%s" % (build_phase_id(target_gw, phase), suffix)


def validate_actuals_window(
    phase: Any,
    target_gw: int,
    actual_gws: Sequence[int],
) -> Tuple[int, ...]:
    normalized = normalize_phase(phase)
    target = _positive_int(target_gw, "target_gw")

    normalized_gws = []
    for index, value in enumerate(actual_gws):
        gw = _positive_int(value, "actual_gws[%s]" % index)
        normalized_gws.append(gw)
    ordered = tuple(sorted(set(normalized_gws)))

    if any(gw > target for gw in ordered):
        raise GameweekCycleError(
            "Actuals from a future GW cannot enter target_gw=%s lifecycle: %s."
            % (target, ordered)
        )

    if normalized in (GameweekPhase.PRE.value, GameweekPhase.FREEZE.value):
        if target in ordered:
            raise GameweekCycleError(
                "Target-GW actuals may enter only in POST; target_gw=%s phase=%s."
                % (target, normalized)
            )
    else:
        if target not in ordered:
            raise GameweekCycleError(
                "POST requires target-GW actuals for target_gw=%s." % target
            )

    return ordered


def validate_pre_window(
    deadline_passed: bool,
    target_results_observed: bool,
) -> None:
    if not isinstance(deadline_passed, bool):
        raise GameweekCycleError("deadline_passed must be boolean.")
    if not isinstance(target_results_observed, bool):
        raise GameweekCycleError("target_results_observed must be boolean.")
    if deadline_passed:
        raise GameweekCycleError(
            "PRE refresh/regeneration is prohibited after the target-GW deadline."
        )
    if target_results_observed:
        raise GameweekCycleError(
            "PRE refresh/regeneration is prohibited after target-GW results are observed."
        )


def validate_post_entry_window(
    deadline_passed: bool,
    target_results_observed: bool,
) -> None:
    if not isinstance(deadline_passed, bool):
        raise GameweekCycleError("deadline_passed must be boolean.")
    if not isinstance(target_results_observed, bool):
        raise GameweekCycleError("target_results_observed must be boolean.")
    if not deadline_passed:
        raise GameweekCycleError(
            "POST cannot begin before the target-GW deadline has passed."
        )
    if not target_results_observed:
        raise GameweekCycleError(
            "POST cannot begin before target-GW actuals/results are observed."
        )


def validate_freeze_window(
    deadline_passed: bool,
    target_results_observed: bool,
) -> None:
    if not isinstance(deadline_passed, bool):
        raise GameweekCycleError("deadline_passed must be boolean.")
    if not isinstance(target_results_observed, bool):
        raise GameweekCycleError("target_results_observed must be boolean.")
    if deadline_passed:
        raise GameweekCycleError(
            "A final pre-deadline freeze cannot be created after the target-GW deadline."
        )
    if target_results_observed:
        raise GameweekCycleError(
            "A final freeze cannot be created after target-GW results are observed."
        )


def validate_transition(
    current: GameweekCycleState,
    proposed: GameweekCycleState,
    *,
    deadline_passed: bool,
    target_results_observed: bool,
) -> None:
    if current.contract_version != proposed.contract_version:
        raise GameweekCycleError("Lifecycle contract version cannot change mid-GW.")
    if current.season != proposed.season or current.target_gw != proposed.target_gw:
        raise GameweekCycleError(
            "Lifecycle transition cannot change season or target_gw."
        )

    edge = (current.phase, proposed.phase)
    allowed_edges = {
        (GameweekPhase.PRE.value, GameweekPhase.PRE.value),
        (GameweekPhase.PRE.value, GameweekPhase.FREEZE.value),
        (GameweekPhase.FREEZE.value, GameweekPhase.FREEZE.value),
        (GameweekPhase.FREEZE.value, GameweekPhase.POST.value),
        (GameweekPhase.POST.value, GameweekPhase.POST.value),
    }
    if edge not in allowed_edges:
        raise GameweekCycleError(
            "Illegal lifecycle transition: %s -> %s." % edge
        )

    if edge == (GameweekPhase.PRE.value, GameweekPhase.PRE.value):
        validate_pre_window(deadline_passed, target_results_observed)

    if edge == (GameweekPhase.PRE.value, GameweekPhase.FREEZE.value):
        validate_freeze_window(deadline_passed, target_results_observed)

    if edge == (GameweekPhase.FREEZE.value, GameweekPhase.POST.value):
        validate_post_entry_window(deadline_passed, target_results_observed)

    if edge == (GameweekPhase.FREEZE.value, GameweekPhase.FREEZE.value):
        if (
            current.freeze_run_id != proposed.freeze_run_id
            or current.freeze_fingerprint != proposed.freeze_fingerprint
        ):
            raise GameweekCycleError(
                "FREEZE rerun is allowed only as an idempotent verification of the exact same freeze."
            )

    if proposed.phase == GameweekPhase.POST.value:
        if current.freeze_run_id != proposed.freeze_run_id:
            raise GameweekCycleError("POST must preserve freeze_run_id lineage.")
        if current.freeze_fingerprint != proposed.freeze_fingerprint:
            raise GameweekCycleError("POST must preserve freeze_fingerprint lineage.")

    if edge == (GameweekPhase.POST.value, GameweekPhase.POST.value):
        current_rank = ACTUALS_STATUS_ORDER[current.target_actuals_status]
        proposed_rank = ACTUALS_STATUS_ORDER[proposed.target_actuals_status]
        if proposed_rank < current_rank:
            raise GameweekCycleError(
                "POST actuals status cannot move backward from %s to %s."
                % (current.target_actuals_status, proposed.target_actuals_status)
            )


def validate_phase_artifacts(
    state: GameweekCycleState,
    artifacts: Sequence[LifecycleArtifactRef],
) -> Dict[str, LifecycleArtifactRef]:
    contract = get_phase_contract(state.phase)
    required = dict(contract.required_artifacts)
    found: Dict[str, LifecycleArtifactRef] = {}

    for artifact in artifacts:
        if artifact.phase != state.phase:
            continue
        expected_phase_id = build_phase_id(state.target_gw, state.phase)
        if artifact.cycle_phase_id != expected_phase_id:
            raise GameweekCycleError(
                "Artifact %s has cycle_phase_id=%s; expected %s."
                % (artifact.artifact_run_id, artifact.cycle_phase_id, expected_phase_id)
            )
        expected_kind = required.get(artifact.track)
        if expected_kind is None or artifact.artifact_kind != expected_kind:
            continue
        if artifact.track in found:
            raise GameweekCycleError(
                "Duplicate required artifact for track=%s phase=%s."
                % (artifact.track, state.phase)
            )
        found[artifact.track] = artifact

    missing = [track for track in required if track not in found]
    if missing:
        raise GameweekCycleError(
            "Missing required %s artifact track(s): %s."
            % (state.phase, tuple(missing))
        )

    if state.phase == GameweekPhase.FREEZE.value:
        for track, artifact in found.items():
            if not artifact.immutable:
                raise GameweekCycleError(
                    "FREEZE artifact for track=%s must be immutable." % track
                )
            if artifact.sha256 is None:
                raise GameweekCycleError(
                    "FREEZE artifact for track=%s must include sha256." % track
                )

    if state.phase == GameweekPhase.POST.value:
        for track, artifact in found.items():
            if artifact.source_freeze_run_id != state.freeze_run_id:
                raise GameweekCycleError(
                    "POST artifact for track=%s must reference freeze_run_id=%s."
                    % (track, state.freeze_run_id)
                )
            if artifact.source_freeze_fingerprint != state.freeze_fingerprint:
                raise GameweekCycleError(
                    "POST artifact for track=%s must reference the frozen fingerprint."
                    % track
                )

    return found


def validate_input_kinds(phase: Any, input_kinds: Iterable[str]) -> Tuple[str, ...]:
    contract = get_phase_contract(phase)
    allowed = set(contract.allowed_inputs)
    prohibited = set(contract.prohibited_inputs)
    normalized = tuple(_nonempty_text(value, "input_kind") for value in input_kinds)

    invalid = [value for value in normalized if value not in allowed]
    explicitly_prohibited = [value for value in normalized if value in prohibited]
    if explicitly_prohibited:
        raise GameweekCycleError(
            "Input kind(s) explicitly prohibited in phase=%s: %s."
            % (contract.phase, tuple(explicitly_prohibited))
        )
    if invalid:
        raise GameweekCycleError(
            "Input kind(s) not allowed in phase=%s: %s."
            % (contract.phase, tuple(invalid))
        )
    return normalized


def validate_output_kinds(phase: Any, output_kinds: Iterable[str]) -> Tuple[str, ...]:
    contract = get_phase_contract(phase)
    allowed = set(contract.allowed_outputs)
    normalized = tuple(_nonempty_text(value, "output_kind") for value in output_kinds)

    invalid = [value for value in normalized if value not in allowed]
    if invalid:
        raise GameweekCycleError(
            "Output kind(s) not allowed in phase=%s: %s."
            % (contract.phase, tuple(invalid))
        )
    return normalized
