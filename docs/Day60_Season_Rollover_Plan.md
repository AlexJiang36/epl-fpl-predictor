# Day60 Season Rollover Plan

## Goal
Make the platform reusable across seasons and able to use previous-season data cleanly, without making the project feel tied to one short live window.

---

## Why Day60 matters
So far, the project works well as a strong single-season system.

But several important next steps depend on proper season support:

- previous-season player priors
- previous-season team priors
- fair full-season backtests
- cleaner historical evaluation
- new-season refresh workflows
- future multi-season modeling and portfolio credibility

Day60 is the point where the project should stop feeling like:
- “a model for one current season only”

and start feeling like:
- “a reusable football prediction platform”

---

## Current problem
Right now, many parts of the system implicitly assume one active season.

That creates several risks:

- old and new season data may collide
- exports may become ambiguous
- predictions may not clearly belong to a specific season
- previous-season priors become awkward to join
- future backtests become harder to trust
- new-season onboarding becomes manual and error-prone

The system therefore needs explicit season-aware design.

---

## Day60 objectives
Day60 should achieve four things:

1. identify where the project is still single-season by assumption
2. decide where `season` must exist explicitly
3. define the first previous-season prior layer
4. document a clean season rollover path

---

# 1. Season-aware audit

## A. Ingest layer
Audit all ingest paths and identify where season is currently only implied.

### Questions to answer
- Are fixtures assumed to belong to only one season?
- Are gameweeks assumed to be unique globally?
- Is player GW data assumed to be unique only by `(player_id, gw)`?
- If data from a new season arrives, will it overwrite or conflict with old rows?
- Can new-season refresh commands run without mixing seasons?

### Main risk
If `gw` is treated as globally unique without `season`, then:
- GW1 from one season
- GW1 from another season

will collide logically even if not immediately in code.

---

## B. Evaluation layer
Audit all evaluation logic for hidden single-season assumptions.

### Questions to answer
- Do model evaluations assume only one season of rows exists?
- Are summaries labeled only by GW range instead of season + GW range?
- Are historical references and current-season results mixed without clear separation?
- Can full-season backtests be stored without season labels?

### Main risk
Evaluation outputs may become misleading if:
- two different seasons both produce `gw=23..27`
- but are later shown as if they are the same window

---

## C. Planning / recommendation layer
Audit recommendation logic such as:
- transfer suggestions
- captain recommendations
- chip planning
- squad planning

### Questions to answer
- Are recommendation snapshots tied to a season?
- Can saved recommendation outputs be traced back to a particular season?
- If a new season starts, will planning logic accidentally use stale prior rows?

### Main risk
Recommendations may silently combine incompatible season contexts if season is not explicit.

---

## D. Feature export / artifact layer
Audit all offline datasets and artifacts.

### Questions to answer
- Do export file names identify the season?
- Do feature snapshots identify the season?
- Do model metadata artifacts identify the training season(s)?
- Can the same feature version exist for two seasons without ambiguity?

### Main risk
Artifacts may look reproducible but still be ambiguous if season is missing.

---

# 2. Where `season` should be encoded explicitly

## A. Database tables
The following tables should become explicitly season-aware.

### High priority
- `fixtures`
- `player_gw_stats`
- `predictions`
- `match_predictions`

### Likely also needed
- `gameweeks`
- recommendation / planning snapshot tables
- future backtest result tables
- any precomputed prior summary tables

---

## Recommended season-aware uniqueness patterns

### Fixtures
Instead of relying only on:
- fixture id
- gw

the system should also conceptually support:
- `season`
- `gw`

### Player GW stats
Instead of only thinking in:
- `(player_id, gw)`

the system should support:
- `(season, player_id, gw)`

### Predictions
Instead of only:
- `(player_id, target_gw, model_name)`

the system should support:
- `(season, player_id, target_gw, model_name)`

### Match predictions
Instead of only:
- `(fixture_id, model_name)`

the system should support season traceability either through:
- season on fixture
- or direct season on prediction rows

---

## B. Metadata / artifact layer
The following should also be season-aware:

- feature snapshots
- model metadata
- validation snapshots
- refresh run snapshots
- backtest summaries
- evaluation exports

This does **not** always mean every artifact must store a single season only.
Some later artifacts may represent:
- one season
- or a training range across multiple seasons

But season context must always be explicit.

---

## C. File naming / export naming
Future exports should include season in the name.

### Current style
- `player_features_gw1_27_v2.csv`

### Recommended future style
- `player_features_2025_26_gw1_27_v2.csv`

### Current style
- `match_features_gw1_27_v2.csv`

### Recommended future style
- `match_features_2025_26_gw1_27_v2.csv`

This makes artifacts much easier to reason about.

---

# 3. First previous-season data layer to support

## Recommended first step
Do **not** start with full raw historical ingestion for many seasons.
That would be heavy and slow down the roadmap.

Instead, the first useful layer should be:

## Previous-season summary priors

### Player priors
Recommended first fields:
- `prev_season_points_per90`
- `prev_season_minutes_per_match`
- `prev_season_goal_involvement_rate`
- `prev_season_clean_sheet_rate` (for DEF/GK relevance later)
- `prev_season_team_id`
- optional `prev_season_cost_band` if historical price data becomes available later

### Team priors
Recommended first fields:
- `prev_season_team_rank`
- `prev_season_points_per_match`
- `prev_season_goal_diff_per_match`
- `prev_season_goals_scored_per_match`
- `prev_season_goals_conceded_per_match`

---

## Why start here
This is the highest-value first step because previous-season priors help most when:

- current-season sample size is still small
- the season is early
- rolling features are unstable
- player quality and team quality need a better starting prior

This is a much more controlled first move than trying to ingest many full historical seasons all at once.

---

# 4. How previous-season priors should coexist with current-season rolling features

## Core principle
Previous-season priors should be **supplementary**, not replacements.

### Current-season rolling features
These remain the main signal once enough games have been played.

Examples:
- recent points
- recent minutes
- recent team form
- recent goal-difference context
- recent opponent difficulty

### Previous-season priors
These provide a starting baseline, especially:
- early in the season
- for players / teams with limited current-season signal
- when trying to distinguish real ability from very short-term noise

---

## Intended interaction
The intended model behavior should be:

### Early season
- previous-season priors matter more
- rolling features are sparse and unstable

### Mid season
- both matter
- rolling form grows more informative

### Late season
- current-season rolling features dominate
- previous-season priors still help but should matter less

---

## Important design note
The project should **not** force a hard switch like:
- before GW5 use priors
- after GW5 ignore priors

Instead, both should coexist in the feature layer and let the model learn how much to use them.

---

# 5. Season-aware changes recommended now

## A. Metadata schema updates
Model metadata should eventually support season-aware fields such as:

- `training_season_start`
- `training_season_end`
- `evaluation_season`
- or a more general season range representation

This will matter later once training includes:
- one season only
- or multiple seasons

---

## B. Feature snapshot metadata updates
Feature snapshots should eventually record:
- export season
- GW range
- feature version
- source tables
- whether previous-season priors were included

---

## C. Validation / refresh snapshot updates
Refresh and validation artifacts should explicitly record:
- season
- target_gw
- model stack used
- relevant feature version(s)

This will make later debugging much easier.

---

# 6. Season rollover workflow to document

## What a future season rollover should look like
When a new season begins, the workflow should look something like:

1. define the new season identifier
2. ingest new season fixtures / teams / players / gameweeks
3. ensure tables accept new season rows without colliding with old season rows
4. generate previous-season prior tables from the just-finished season
5. refresh feature exports with current-season rows plus prior-season fields
6. run validation checks with season-aware logic
7. start the new season prediction cycle

---

## Documentation should answer
A future maintainer should be able to answer:

- what must be changed when a new season starts?
- what tables need season values?
- what artifacts should include season in their filenames?
- how are previous-season priors generated?
- how are full-season backtests run fairly?

---

# 7. What Day60 should actually implement now

## Must-do today
### Documentation and design
- complete season-aware audit
- decide where `season` belongs
- define first prior fields
- document rollover workflow

### Naming / artifact conventions
- update docs and conventions so exports / snapshots become season-aware

### Compatibility design
- confirm that future backtests should operate on:
  - season-aware windows
  - fair walk-forward logic
  - clearly labeled comparable ranges

---

## Good stretch goal for Day60
If time allows, start one of these:

- add season-aware fields to one or two metadata / snapshot structures
- create a stub or schema plan for previous-season prior tables
- define a canonical season format such as:
  - `2025_26`
  - or `2025-26`

---

## Not required for Day60
Day60 does **not** need to fully finish:
- multi-season raw ingest for many seasons
- full historical raw backfill
- fully trained multi-season models
- complete season-wide rolling backtests

Those are important, but they belong after the season-aware foundation is in place.

---

# 8. Recommended season identifier format

## Suggested format
Use a stable, filename-friendly format such as:

- `2025_26`

Why this is good:
- easy in filenames
- easy in metadata
- avoids slash issues
- visually clear

Examples:
- `player_features_2025_26_gw1_27_v2.csv`
- `match_features_2025_26_gw1_27_v2.csv`

---

# 9. Relationship to full-season backtesting
Day60 should prepare for the next important step:
- full-season rolling backtest / season review

That later backtest should use:
- clear season labels
- same-window model comparison
- rolling / walk-forward evaluation
- separation between comparable models and historical reference models

So Day60 is not the backtest day itself.
It is the day that makes the backtest trustworthy and scalable.

---

# 10. Recommended deliverables

## Documents
- `Day60_Season_Rollover_Plan.md`
- updated `season_data_audit.md`
- optional `previous_season_priors_design.md`

## Decisions
- season identifier format chosen
- tables needing season identified
- artifact naming rules decided
- first prior layer defined

## Optional first implementation
- metadata / snapshot structures begin storing season explicitly
- first prior-table schema draft created

---

## Final recommendation
The Day60 focus should be:

- make season handling explicit
- make rollover reproducible
- define previous-season priors cleanly
- prepare the system for fair full-season and multi-season evaluation later

The project does not need to become fully multi-season in one day.
It just needs to stop being architecturally single-season.
