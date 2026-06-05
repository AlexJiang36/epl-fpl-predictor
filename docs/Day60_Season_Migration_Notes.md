# Day60 Next Step: Season Migration Notes

## What this migration does
This migration adds a required `season` column to the core season-sensitive tables:

- `gameweeks`
- `fixtures`
- `player_gw_stats`
- `predictions`
- `match_predictions`

It also backfills all existing rows to:

- `2025_26`

and adds helpful season-aware indexes.

## Why this is the right next step
This is the first real infrastructure move after the Day60 season rollover plan.

It makes the database stop behaving like a purely single-season system, without yet taking on the risk of rewriting every uniqueness rule immediately.

## Important design choice
This migration **does not** add new unique constraints yet.

That is intentional.

Before adding constraints like:

- `(season, player_id, gw)`
- `(season, player_id, target_gw, model_name)`
- `(season, fixture_id, model_name)`

the app write paths and existing data should first be updated and audited.

This makes the rollout safer.

## What should happen right after this migration
After applying the migration, the next code updates should be:

1. update ORM models to include `season`
2. update all write paths so new rows always write `season`
3. update read/query paths to filter by `season`
4. update feature exports and snapshots to include `season`
5. only then consider a follow-up migration for season-aware uniqueness constraints

## Suggested Alembic filename
A good filename would be something like:

- `backend/alembic/versions/20260531_day60_add_season_columns.py`
