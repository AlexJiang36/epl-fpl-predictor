# Constraints and Assumptions

Project: EPL / FPL ML Prediction Platform  
Area: Transfer Recommendation Constraint Engine

## Purpose
This document records the assumptions currently used by the transfer recommendation MVP so future chats and future refactors have a stable source of truth.

## Scope
These assumptions apply to the current backend endpoint:
- `POST /recommendations/transfers`

They describe the **current MVP transfer engine**, not the final long-term optimizer.

---

## 1. Transfer model scope

### Current scope
The current transfer engine models a **single 1-for-1 transfer**:
- exactly one outgoing player
- one incoming player at a time
- recommendations are returned as candidate replacements for the selected outgoing player

### Not yet included
The current MVP does **not** yet model:
- 2-transfer or multi-transfer search
- full-sequence optimization
- wildcard / free hit / other chip behavior
- long-horizon planning across multiple future gameweeks

---

## 2. Input assumptions

The endpoint currently assumes the caller provides a valid squad snapshot via:
- `target_gw`
- `model_name`
- `squad_player_ids`
- `bank`
- `free_transfers`
- `limit`

### Assumption: caller-provided squad snapshot
The backend does **not** yet own a user account system or stored user squad state.
Instead, the caller provides the current squad snapshot directly.

This means the recommendation engine is currently a **snapshot-based decision engine**, not a persistent user-state system.

---

## 3. Outgoing-player assumption

### Current outgoing selection rule
The current MVP selects **one outgoing player** from the provided squad snapshot.

The selected outgoing player is chosen by ranking current squad players by:
1. lowest predicted points first
2. if tied, higher cost first
3. then stable tie-break by player id

### Important note
This is a pragmatic MVP rule.
It is **not** intended to be the final long-term outgoing policy.

The current engine does **not** assume that any non-available player must always be sold immediately.
A doubtful or injured player may still be worth keeping depending on horizon and context, so that logic is intentionally not hard-coded as a mandatory sell rule.

---

## 4. Transfer constraint assumptions

The current engine evaluates candidate incoming players using reusable helper checks.

### 4.1 Budget cap
Assumption:
- a valid incoming player must satisfy the 1-for-1 budget cap

Rule:
- `incoming.now_cost <= outgoing.now_cost + bank`

Interpretation:
- current bank is assumed to be caller-provided remaining budget
- current MVP treats prices using the same integer `now_cost` units already stored in the database

### 4.2 Max 3 players per club
Assumption:
- a valid transfer must not create a resulting squad with more than 3 players from the same club

Rule:
- same-team swap does not increase club count
- different-team incoming transfer increases incoming club count by 1
- resulting count must be `<= 3`

### 4.3 Squad size / squad transition rules
Assumption:
- because the current MVP only models **1-for-1 transfers**, total squad size must remain unchanged
- a valid transfer must preserve valid position counts after the swap

Rule:
- outgoing and incoming are evaluated as a squad transition
- resulting position counts must remain valid

For the current MVP, this is enforced through same-position 1-for-1 transfer logic.

### 4.4 Position compatibility
Assumption:
- the current MVP only recommends **same-position replacements**

Rule:
- outgoing position must equal incoming position

Examples:
- FWD -> FWD is allowed
- MID -> FWD is not allowed

### 4.5 Availability filtering
Assumption:
- current incoming transfer recommendations should only include available players

Rule:
- incoming player must satisfy `status == "a"`

Important note:
- this filter currently applies to **incoming** recommendations
- it is not currently used as a forced sell rule for outgoing selection

---

## 5. Transfer-cost assumptions

### Free transfer rule
Current assumption:
- if `free_transfers >= 1`, transfer cost is 0
- otherwise the current MVP charges `4` points for the suggested 1-for-1 move

Rule:
- `transfer_cost_points = 0` if free transfers remain
- `transfer_cost_points = 4` otherwise

### Net gain definition
Current output field:
- `net_gain_after_cost = projected_gain - transfer_cost_points`

This means the API currently ranks not just by raw projected gain, but by gain after the immediate transfer cost.

---

## 6. Recommendation list assumptions

### Current recommendation shape
The endpoint currently returns:
- one `selected_outgoing`
- a list of candidate incoming replacements

### Current list size
The endpoint returns at most `limit` rows.
Current preferred default is small (for example, 3) to avoid overloading the response with too many options.

---

## 7. Unit-style checks

The constraint engine is expected to include unit-style checks for both valid and invalid cases.

### Covered check categories
- budget cap pass / fail
- club cap pass / fail
- position compatibility pass / fail
- availability pass / fail
- squad transition pass / fail

### Self-test entrypoint
The backend should expose:
- `GET /recommendations/transfers/self-test`

Expected response:
```json
{"ok": true}
```

This endpoint is intended as a lightweight regression check for the constraint layer.

---

## 8. What this document does not assume

This document does **not** assume:
- persistent user accounts
- stored user squads in the database
- official FPL API integration for a specific logged-in manager
- multi-step search over all possible transfer combinations
- full chip logic
- perfect real-world FPL strategy

It only documents the current MVP assumptions used by the backend transfer recommendation engine.

---

## 9. Future extension points

These assumptions are intentionally narrow so the engine can grow later.
Possible future upgrades:
- saved squad snapshots
- multi-transfer optimization
- hit-aware ranking beyond a single move
- injury horizon / expected return logic
- club and formation-aware multi-step transition validation
- richer outgoing ranking rules
- frontend transfer recommendation page backed by the same snapshot API
