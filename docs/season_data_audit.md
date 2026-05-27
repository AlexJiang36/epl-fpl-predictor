# Season Data Audit

## Purpose
This document audits season-specific assumptions in the current platform and defines how the project should evolve to support:

- previous-season data
- multi-season historical analysis
- season rollover
- season-aware feature engineering
- season-aware model evaluation

The goal is to prevent the project from feeling tied to only one short live season.

---

## Why this audit is needed
The project is starting to move beyond:
- simple current-season rolling features
- one-season-only refresh workflows
- one-season-only evaluation assumptions

Future work will require:
- previous-season player priors
- historical team strength
- multi-season feature snapshots
- multi-season backtesting
- cleaner season rollover support

That means the data layer needs clearer season semantics.

---

## Current risk areas

### 1. Tables that may behave like single-season tables
Some current workflows likely assume only one active season is present.

Risk areas include:
- `fixtures`
- `player_gw_stats`
- `predictions`
- `match_predictions`

If these tables do not encode `season`, multi-season data can become ambiguous.

### 2. Feature pipelines using implicit season assumptions
Current feature exports and rolling calculations may assume:
- only one season is present
- GW numbering is globally unique enough
- previous GW means previous GW in the same season

That becomes risky once multiple seasons coexist.

### 3. Evaluation ambiguity
Historical evaluation can become confusing if:
- prediction rows from different seasons mix together
- actuals are joined only on GW without season
- feature snapshots do not record season scope

---

## Tables / layers that should become season-aware

## Highest-priority season-aware layers
These are the most important places to support explicit `season`:

### `fixtures`
Reason:
- GW numbers repeat every season
- fixture context should be season-specific
- future feature joins will need season-safe filtering

### `player_gw_stats`
Reason:
- GW numbers repeat
- rolling player features should stay within the correct season unless intentionally crossing seasons
- previous-season prior creation needs clean season grouping

### `predictions`
Reason:
- `target_gw` alone is not enough once multiple seasons exist
- model outputs should be attributable to a season

### `match_predictions`
Reason:
- match forecasts should not rely on GW alone
- future historical comparisons need season-safe joins

## Metadata / artifact layers that should also encode season
These do not necessarily need DB changes first, but should become season-aware in metadata:

- feature snapshots
- run snapshots
- model metadata
- evaluation artifacts
- docs / runbooks

---

## Previous-season data strategy

## Recommended first approach
Use previous-season information primarily as **prior features**, not as one giant merged rolling window.

This keeps the design simpler and more interpretable.

### Recommended previous-season usage
Examples:

- `prev_season_points_per90`
- `prev_season_minutes_per_match`
- `prev_season_goal_involvement_rate`
- `prev_season_team_attack_strength`
- `prev_season_team_defense_strength`

This allows current-season form and previous-season quality to remain separate signals.

## Why prior features are better than mixing seasons directly
If previous-season data is blended directly into current rolling windows:

- feature meaning becomes less clear
- early-season behavior can become inconsistent
- debugging gets harder
- model explanations become weaker

Using prior-style fields is cleaner.

---

## Prior vs aggregate vs lookup-table decision

## Option A — Prior features
Examples:
- `prev_season_points_per90`
- `prev_season_minutes_per_match`

### Pros
- simple to reason about
- easy to expose in feature exports
- easy to document

### Cons
- may require extra aggregation logic upstream

## Option B — Aggregates
Examples:
- season summary numbers per player/team

### Pros
- simple intermediate representation
- reusable in multiple pipelines

### Cons
- still needs join logic later

## Option C — Lookup table
Examples:
- `player_season_summary`
- `team_season_summary`

### Pros
- clean long-term architecture
- reusable for many models / dashboards

### Cons
- heavier implementation cost right now

## Recommended decision
For the next phase:

- use **prior features**
- optionally compute them from season aggregates
- do **not** overbuild a separate lookup-table system yet unless needed

This is the best balance for the current project stage.

---

## Season-aware rules for future feature engineering

### Rule 1 — GW is not enough by itself
Anywhere a workflow depends on GW, ask:

- which season does this GW belong to?

### Rule 2 — Rolling windows should default to same-season windows
Unless explicitly documented otherwise, rolling features should stay inside the same season.

### Rule 3 — Previous-season data should be explicit
If a feature uses previous-season information, it should be named clearly as a prior or previous-season field.

### Rule 4 — Metadata should record season scope
Feature snapshots, model metadata, and evaluation outputs should make season coverage visible.

---

## Expected future schema / metadata direction

## Near-term metadata additions
Without forcing immediate DB migrations everywhere, future artifacts should start recording fields such as:

- `season`
- `season_start`
- `season_end`
- `source_seasons`

depending on the artifact type.

## Likely future DB additions
The most likely future schema path is adding explicit `season` columns to:

- `fixtures`
- `player_gw_stats`
- `predictions`
- `match_predictions`

This should happen before serious multi-season backtesting is treated as production-like.

---

## Impact on backtesting
Multi-season backtesting should eventually support:

- filtering by explicit season
- comparing the same model across different seasons
- keeping training and evaluation windows season-aware
- using previous-season priors without leaking future information

At the moment, the platform should assume this is an important future requirement.

---

## Impact on season rollover
A clean season rollover path should eventually answer:

- how are new season fixtures ingested?
- how are season-specific stats separated?
- how are priors from the old season prepared?
- how are active prediction targets tied to the current season?
- how are archived predictions from the old season preserved?

This should be documented before the platform is treated as fully multi-season ready.

---

## Recommended next steps

### Day54
Build `player_features_v2` with explicit previous-season prior fields

### Day55
Build `match_features_v2` with team-strength and historical context fields

### After that
Decide whether DB schema updates are needed immediately for:
- `fixtures`
- `player_gw_stats`
- `predictions`
- `match_predictions`

### Before serious multi-season support
Add explicit season metadata to:
- feature snapshots
- model metadata
- evaluation outputs

---

## Summary
The project is moving toward:
- stronger feature engineering
- previous-season priors
- multi-season reuse
- season rollover support

To support that safely, the platform should treat `season` as an increasingly important piece of data rather than relying on GW alone.

The recommended short-term design is:

- keep current-season rolling windows explicit
- add previous-season information as prior features
- make artifacts and future schema work season-aware
- avoid overcomplicating the architecture too early