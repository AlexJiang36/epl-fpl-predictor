# FPL Gameweek Lifecycle Contract

**Contract:** `fpl_gameweek_cycle_v1`  
**Milestone:** Day125A  
**Scope:** Every live FPL Gameweek follows one explicit PRE → FREEZE → POST lifecycle.

## 1. Purpose

This contract separates prediction/decision generation from deadline-final evidence and from result evaluation.

The lifecycle is:

```text
PRE
  ↓
FREEZE
  ↓
POST
```

The central leakage rule is:

```text
Target-GW actuals/results MUST NOT enter PRE or FREEZE.
Target-GW actuals first become legal inputs in POST.
```

A PRE publish is not a final freeze. A successful model/transfer preview is not a final Gameweek decision.

---

## 2. Phase identifiers

Each target GW has stable lifecycle identifiers:

```text
GW02-PRE
GW02-FREEZE
GW02-POST-EVAL
```

Execution runs append a unique suffix:

```text
GW02-PRE-20260825T120000Z
GW02-FREEZE-20260828T173000Z
GW02-POST-EVAL-20260901T020000Z
```

The stable phase identifier describes lifecycle state. The execution run identifier describes one concrete attempt/artifact lineage.

---

## 3. PRE

### Goal

Generate and validate the latest target-GW predictions and decisions before the deadline.

### Allowed inputs

```text
historical priors
actuals from GWs strictly before target_gw
target-GW fixtures
live player availability/status
current prices
previous finalized/frozen owned squad
```

For GW2, GW1 actuals are legal PRE inputs. GW2 actuals are not.

### Required phase-complete evidence tracks

```text
Player Model → player_predictions_pre
Match Model  → match_predictions_pre
Model Team   → model_team_decision_pre
Team Alex    → team_alex_decision_pre
```

These are required for a phase-complete PRE package. Individual internal PRE jobs may run earlier, but the lifecycle cannot claim complete PRE readiness until all four tracks are represented.

### Allowed lifecycle outputs

```text
player_predictions_pre
match_predictions_pre
model_team_decision_pre
team_alex_decision_pre
phase_manifest
pre_publish_receipt
```

Supplementary diagnostics may exist outside the lifecycle contract, but a lifecycle-owned output kind must be explicitly allowed by `PhaseContract.allowed_outputs`.

### Reruns

PRE is intentionally rerunnable. New availability, prices, transfer news, official prior-GW actuals, or model refreshes may supersede earlier PRE attempts.

Rules:

```text
PRE → PRE is legal only before the target-GW deadline and before target-GW results are observed.
Every attempt gets a new execution run ID.
Old evidence is preserved; do not overwrite lineage.
A PRE publish may update serving tables through an explicit gate.
A PRE publish MUST NOT set final_deadline_freeze=true.
```

Once target-GW actuals/results are observed, PRE regeneration is prohibited.

---

## 4. FREEZE

### Goal

Create the one immutable, leakage-safe, final pre-deadline evidence bundle for the target GW.

### Entry rules

FREEZE is legal only from PRE and only when:

```text
target-GW deadline has not passed
target-GW actuals/results have not been observed
all required PRE decision/model evidence is ready
```

If the deadline is missed without a valid freeze, the system must not backfill a fake pre-deadline freeze later.

### Required frozen evidence tracks

```text
Player Model → player_predictions_final
Match Model  → match_predictions_final
Model Team   → model_team_final
Team Alex    → team_alex_final
```

Allowed lifecycle outputs for FREEZE are:

```text
player_predictions_final
match_predictions_final
model_team_final
team_alex_final
phase_manifest
freeze_manifest
```

Every required FREEZE artifact must be:

```text
immutable = true
content-addressed / SHA256 recorded
copied or referenced through durable freeze lineage
```

### Reruns

A final freeze cannot be replaced.

```text
FREEZE → FREEZE
```

is legal only as an idempotent verification of the exact same `freeze_run_id` and `freeze_fingerprint`.

Different bytes, different decisions, different predictions, or a different fingerprint require rejection. There is no `--replace-final-freeze` escape hatch.

---

## 5. POST

### Goal

Evaluate the frozen predictions and decisions against separately captured target-GW actuals.

### Allowed inputs

```text
exact frozen Player Model evidence
exact frozen Match Model evidence
exact frozen Model Team evidence
exact frozen Team Alex evidence
target-GW official actuals
```

### Required evaluation tracks

```text
Player Model → player_model_evaluation
Match Model  → match_model_evaluation
Model Team   → model_team_evaluation
Team Alex    → team_alex_evaluation
```

Allowed lifecycle outputs for POST are:

```text
player_model_evaluation
match_model_evaluation
model_team_evaluation
team_alex_evaluation
phase_manifest
post_evaluation_manifest
```

Every POST artifact must reference the exact final freeze lineage:

```text
source_freeze_run_id
source_freeze_fingerprint
```

### Provisional and final actuals

POST may be evaluated first against provisional official actuals and rerun later against finalized actuals.

Allowed progression:

```text
provisional → provisional
provisional → final
final → final
```

Forbidden:

```text
final → provisional
```

A later POST run creates new evaluation evidence. It never changes the frozen predictions or decisions.

---

## 6. State machine

Legal transitions:

```text
PRE    → PRE       safe refresh/rerun
PRE    → FREEZE    create final pre-deadline evidence
FREEZE → FREEZE    exact idempotent verification only
FREEZE → POST      target-GW actuals are now available
POST   → POST      evaluation rerun; actual status may only progress
```

Illegal transitions include:

```text
PRE    → POST      skipped final freeze
FREEZE → PRE       prediction/decision regeneration after freeze
POST   → PRE       post-result leakage
POST   → FREEZE    retroactive final freeze
```

The lifecycle is monotonic. Transition validation is also window-aware:

```text
PRE → PRE       requires deadline_passed=false and target_results_observed=false
PRE → FREEZE    requires deadline_passed=false and target_results_observed=false
FREEZE → POST   requires deadline_passed=true and target_results_observed=true
FREEZE → FREEZE exact idempotent verification may happen later
POST → POST     may happen later while actual status only progresses
```

This closes the post-result prediction-regeneration loophole: callers cannot claim a safe PRE rerun merely by omitting target-GW actuals from the input list.

---

## 7. Actuals policy

For target GW `N`:

### PRE / FREEZE

Allowed actuals:

```text
GW1 ... GW(N-1)
```

Forbidden:

```text
GWN
GW(N+1) or later
```

### POST

Required:

```text
GWN actuals
```

Future actuals beyond the evaluated target GW are forbidden because they introduce evaluation leakage.

---

## 8. Player Model / Match Model / Model Team / Team Alex separation

The four evidence tracks remain independent so failures can be attributed correctly.

```text
Player Model
Match Model
Model Team
Team Alex
```

Model Team decisions must not be overwritten by Team Alex choices. Team Alex evidence is an independent user-decision track.

POST evaluations must preserve the same separation.

---

## 9. PRE publish vs FINAL freeze

Serving predictions during PRE is allowed when an explicit publish gate validates them.

Example:

```text
GW2 early-season Player + Match predictions
→ PRE preview
→ PRE publish gate
→ canonical DB publish
→ post-publish verification
```

This still remains:

```text
phase = PRE
final_deadline_freeze = false
```

Near the deadline, the system may refresh PRE again. Only the later FREEZE package is the immutable baseline used for POST evaluation.

---

## 10. Safe rerun rules

| Operation | Safe? | Rule |
|---|---:|---|
| PRE refresh before deadline | Yes | New execution run ID; preserve old artifacts |
| PRE serving-table republish | Yes, gated | Must remain non-final and preserve lineage |
| PRE refresh after deadline | No | Deadline-final evidence must already be frozen |
| PRE refresh after target results | No | Leakage risk |
| Replace final FREEZE | No | Immutable |
| Verify exact same FREEZE | Yes | Same run ID + fingerprint only |
| POST provisional rerun | Yes | New evaluation artifact |
| POST provisional → final | Yes | Same freeze lineage |
| POST final → provisional | No | Actual-status regression |
| Regenerate predictions from POST actuals | No | Leakage |

---

## 11. Minimum lifecycle metadata

Every phase-level manifest should carry at least:

```text
contract_version = fpl_gameweek_cycle_v1
season
target_gw
phase
phase_id
execution_run_id
created_at
source artifact/run IDs
source hashes where applicable
writes_database
final_deadline_freeze
```

FREEZE additionally requires:

```text
freeze_run_id
freeze_fingerprint
immutable = true
```

POST additionally requires:

```text
target_actuals_status = provisional | final
source_freeze_run_id
source_freeze_fingerprint
```

---

## 12. Current 2026/27 mapping

### GW1

```text
GW01-FREEZE
= existing immutable final pre-deadline Player / Match / Model Team / Team Alex evidence

GW01-POST-EVAL
= provisional evaluation exists
= may rerun when official FPL actuals become final
= must reference the same GW01 freeze
```

### GW2

Current early-season prediction publish is still:

```text
GW02-PRE
```

The current transfer-optimizer result is also PRE/provisional. It is not a freeze.

Near the deadline:

```text
refresh GW02-PRE
→ finalize Model Team + Team Alex evidence
→ GW02-FREEZE
```

After GW2 official actuals are captured:

```text
GW02-FREEZE + GW2 actuals
→ GW02-POST-EVAL
```

---

## 13. Contract stop condition

Day125A is complete when any Gameweek can be represented and validated through this state machine, including explicit input/output contracts and rejection of illegal transitions, late PRE regeneration, actuals leakage, mutable final-freeze artifacts, and broken POST freeze lineage.
