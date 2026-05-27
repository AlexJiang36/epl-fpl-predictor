# Feature v2 Plan

## Purpose
This document defines the first major feature-upgrade plan for the platform.

The goal is to improve prediction quality for both:
- player performance prediction
- match result prediction

while keeping the project:
- reproducible
- auditable
- season-aware
- compatible with the existing data / ML infra layers

---

## Why Feature v2 is needed
Current model behavior is still too dependent on simple rolling-history signals.

For example:
- player predictions can become too similar across future GWs
- target GW may act more like a write label than a real predictive context
- match predictions rely on limited team-form information
- future opponent context is still under-modeled

Feature v2 is meant to fix that by adding stronger football signals.

---

## Current feature gaps

## Player prediction gaps
Current player prediction logic is still missing several important predictive signals:

- fixture difficulty / opponent quality
- opponent defensive strength
- home vs away signal
- stronger team-level attacking context
- previous-season player prior
- longer-horizon player quality signal
- clearer separation between short-term form and long-term ability

## Match prediction gaps
Current match prediction logic is still missing:

- stronger team-strength proxy
- more explicit attack / defense signals
- goals scored / conceded form
- rest-day / schedule congestion context
- limited historical head-to-head context
- features that support future goals / scoreline prediction

---

## First-wave Feature v2 additions

## Player feature v2 candidates
The first wave of player feature upgrades should prioritize:

1. fixture difficulty proxy
2. opponent defensive strength
3. home / away flag
4. recent minutes stability
5. previous-season player priors
6. team attacking strength proxy

### Proposed player v2 fields
Possible first-wave player feature fields:

- `is_home`
- `opponent_team_id`
- `opponent_def_strength_recent`
- `opponent_goals_conceded_avg_recent`
- `fixture_difficulty_proxy`
- `recent_avg_minutes`
- `recent_mins_60_plus_count`
- `team_attack_strength_recent`
- `prev_season_points_per90`
- `prev_season_minutes_per_match`
- `prev_season_goal_involvement_rate`

### Player feature priority
Highest priority:
- `is_home`
- `opponent_def_strength_recent`
- `fixture_difficulty_proxy`
- `prev_season_points_per90`
- `recent_avg_minutes`

Second priority:
- `team_attack_strength_recent`
- `prev_season_minutes_per_match`
- `prev_season_goal_involvement_rate`

---

## Match feature v2 candidates
The first wave of match feature upgrades should prioritize:

1. historical team strength
2. recent goals scored / conceded
3. opponent-quality / fixture-difficulty context
4. rest days
5. limited H2H features
6. groundwork for goals prediction

### Proposed match v2 fields
Possible first-wave match feature fields:

- `home_team_strength_recent`
- `away_team_strength_recent`
- `home_goals_scored_avg_recent`
- `away_goals_scored_avg_recent`
- `home_goals_conceded_avg_recent`
- `away_goals_conceded_avg_recent`
- `strength_diff_recent`
- `rest_days_home`
- `rest_days_away`
- `limited_h2h_home_points_avg`
- `limited_h2h_goal_diff_avg`
- `fixture_difficulty_proxy_home`
- `fixture_difficulty_proxy_away`

### Match feature priority
Highest priority:
- `home_goals_scored_avg_recent`
- `away_goals_scored_avg_recent`
- `home_goals_conceded_avg_recent`
- `away_goals_conceded_avg_recent`
- `strength_diff_recent`
- `rest_days_home`
- `rest_days_away`

Second priority:
- `limited_h2h_home_points_avg`
- `limited_h2h_goal_diff_avg`
- explicit fixture difficulty proxy fields

---

## Guiding rules for new features

### Rule 1 — Prefer stable features first
Start with features that are:
- easy to compute
- reproducible
- explainable
- likely to generalize

### Rule 2 — Avoid overfitting to tiny samples
Head-to-head features should be:
- limited
- weakly weighted
- clearly documented as secondary signals

### Rule 3 — Separate short-term form from long-term prior
Use:
- rolling recent form
- previous-season prior

as separate concepts, not as one blended field.

### Rule 4 — Keep every feature version traceable
Every new export, model, and evaluation run should explicitly record which feature version it used.

---

## Feature version naming rules

## Dataset naming
Use explicit names such as:

- `player_features_v0`
- `player_features_v2`
- `match_features_v0`
- `match_features_v2`

## Model naming
Use model names that reflect meaningful version jumps, for example:

- `ridge_rollform_v1`
- `ridge_rollform_v2`
- `match_logreg_v0`
- `match_logreg_v1`
- `match_xgb_v1`

## Metadata expectations
For every meaningful model experiment, record:

- model name
- task type
- feature version
- training window
- evaluation range
- metrics summary
- notes

---

## Recommended implementation order

### Day54
Implement and export `player_features_v2`

### Day55
Implement and export `match_features_v2`

### Day56
Train / compare upgraded player models on `player_features_v2`

### Day57
Train / compare upgraded match models on `match_features_v2`

### Day58
Select the best small set of active models

---

## Active model philosophy
The platform should not keep too many active models.

Recommended steady-state:

### Player models
- 1 baseline model
- 1 stronger selected model

### Match models
- 1 baseline model
- 1 stronger selected model

Other models can remain:
- experimental
- archived in metadata
- documented in evaluation comparisons

---

## Summary
Feature v2 is intended to improve prediction quality by introducing:
- fixture difficulty
- opponent strength
- previous-season priors
- stronger team-level form signals
- limited historical context
- better support for future goals and scoreline prediction

This document defines the first-wave feature upgrade plan that later implementation work should follow.