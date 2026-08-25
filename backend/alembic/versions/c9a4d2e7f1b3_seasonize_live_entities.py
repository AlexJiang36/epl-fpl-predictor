"""seasonize live team player and fixture identities

Revision ID: c9a4d2e7f1b3
Revises: b8f3a2c1d4e6
Create Date: 2026-08-25

The official FPL API reuses team, player/element, and fixture IDs across
seasons.  The previous schema treated those external IDs as globally unique,
which makes a live season rollover overwrite historical identities.

This migration:
- adds season to canonical teams and players and backfills 2025_26;
- changes team/player/fixture external-ID uniqueness to season-scoped;
- preserves all existing primary keys and foreign-key relationships.
"""

from typing import Iterable, List, Sequence

from alembic import op
import sqlalchemy as sa


revision = "c9a4d2e7f1b3"
down_revision = "b8f3a2c1d4e6"
branch_labels = None
depends_on = None

DEFAULT_SEASON = "2025_26"


def _column_names(value: Iterable[str]) -> List[str]:
    return [str(item) for item in (value or [])]


def _drop_single_column_global_uniqueness(table_name: str, column_name: str) -> None:
    """Drop any legacy one-column UNIQUE constraint/index for an external ID."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    constraints = inspector.get_unique_constraints(table_name)
    matching_constraints = [
        item
        for item in constraints
        if _column_names(item.get("column_names")) == [column_name]
        and item.get("name")
    ]
    matching_constraint_names = {str(item["name"]) for item in matching_constraints}

    indexes = inspector.get_indexes(table_name)
    matching_indexes = [
        item
        for item in indexes
        if bool(item.get("unique"))
        and _column_names(item.get("column_names")) == [column_name]
        and item.get("name")
        and item.get("duplicates_constraint") not in matching_constraint_names
    ]

    for item in matching_constraints:
        op.drop_constraint(str(item["name"]), table_name, type_="unique")

    for item in matching_indexes:
        op.drop_index(str(item["name"]), table_name=table_name)


def _index_names(table_name: str) -> Sequence[str]:
    inspector = sa.inspect(op.get_bind())
    return [str(item["name"]) for item in inspector.get_indexes(table_name) if item.get("name")]


def _constraint_names(table_name: str) -> Sequence[str]:
    inspector = sa.inspect(op.get_bind())
    return [
        str(item["name"])
        for item in inspector.get_unique_constraints(table_name)
        if item.get("name")
    ]


def _create_index_if_missing(name: str, table_name: str, columns: Sequence[str]) -> None:
    if name not in _index_names(table_name):
        op.create_index(name, table_name, list(columns), unique=False)


def _assert_downgrade_safe() -> None:
    bind = op.get_bind()
    checks = [
        ("teams", "season"),
        ("players", "season"),
        ("fixtures", "season"),
    ]
    for table_name, season_column in checks:
        count = bind.execute(
            sa.text(
                "SELECT COUNT(*) FROM %s WHERE %s <> :season"
                % (table_name, season_column)
            ),
            {"season": DEFAULT_SEASON},
        ).scalar()
        if int(count or 0) > 0:
            raise RuntimeError(
                "Refusing downgrade: %s contains non-%s rows. "
                "Downgrading would restore unsafe global external-ID uniqueness."
                % (table_name, DEFAULT_SEASON)
            )


def upgrade() -> None:
    # Add season to the two canonical entity tables that previously lacked it.
    op.add_column("teams", sa.Column("season", sa.String(length=16), nullable=True))
    op.add_column("players", sa.Column("season", sa.String(length=16), nullable=True))

    op.execute(
        "UPDATE teams SET season = '%s' WHERE season IS NULL" % DEFAULT_SEASON
    )
    op.execute(
        "UPDATE players SET season = '%s' WHERE season IS NULL" % DEFAULT_SEASON
    )

    op.alter_column(
        "teams",
        "season",
        existing_type=sa.String(length=16),
        nullable=False,
    )
    op.alter_column(
        "players",
        "season",
        existing_type=sa.String(length=16),
        nullable=False,
    )

    # The old schema made these FPL IDs globally unique. They are reused every
    # season, so remove that global uniqueness without changing primary keys.
    _drop_single_column_global_uniqueness("teams", "fpl_team_id")
    _drop_single_column_global_uniqueness("players", "fpl_player_id")
    _drop_single_column_global_uniqueness("fixtures", "fpl_fixture_id")

    _create_index_if_missing("ix_teams_season", "teams", ["season"])
    _create_index_if_missing("ix_players_season", "players", ["season"])
    _create_index_if_missing("ix_teams_fpl_team_id", "teams", ["fpl_team_id"])
    _create_index_if_missing("ix_players_fpl_player_id", "players", ["fpl_player_id"])
    _create_index_if_missing("ix_fixtures_fpl_fixture_id", "fixtures", ["fpl_fixture_id"])

    if "uq_teams_season_fpl_team_id" not in _constraint_names("teams"):
        op.create_unique_constraint(
            "uq_teams_season_fpl_team_id",
            "teams",
            ["season", "fpl_team_id"],
        )
    if "uq_players_season_fpl_player_id" not in _constraint_names("players"):
        op.create_unique_constraint(
            "uq_players_season_fpl_player_id",
            "players",
            ["season", "fpl_player_id"],
        )
    if "uq_fixtures_season_fpl_fixture_id" not in _constraint_names("fixtures"):
        op.create_unique_constraint(
            "uq_fixtures_season_fpl_fixture_id",
            "fixtures",
            ["season", "fpl_fixture_id"],
        )


def downgrade() -> None:
    _assert_downgrade_safe()

    op.drop_constraint(
        "uq_fixtures_season_fpl_fixture_id", "fixtures", type_="unique"
    )
    op.drop_constraint(
        "uq_players_season_fpl_player_id", "players", type_="unique"
    )
    op.drop_constraint(
        "uq_teams_season_fpl_team_id", "teams", type_="unique"
    )

    # Restore the legacy globally unique indexes only when no second-season
    # rows exist; _assert_downgrade_safe() guarantees that condition.
    existing_fixture_indexes = set(_index_names("fixtures"))
    if "ix_fixtures_fpl_fixture_id" in existing_fixture_indexes:
        op.drop_index("ix_fixtures_fpl_fixture_id", table_name="fixtures")
    op.create_index(
        "ix_fixtures_fpl_fixture_id",
        "fixtures",
        ["fpl_fixture_id"],
        unique=True,
    )

    existing_player_indexes = set(_index_names("players"))
    if "ix_players_fpl_player_id" in existing_player_indexes:
        op.drop_index("ix_players_fpl_player_id", table_name="players")
    op.create_index(
        "ix_players_fpl_player_id",
        "players",
        ["fpl_player_id"],
        unique=True,
    )

    existing_team_indexes = set(_index_names("teams"))
    if "ix_teams_fpl_team_id" in existing_team_indexes:
        op.drop_index("ix_teams_fpl_team_id", table_name="teams")
    op.create_index(
        "ix_teams_fpl_team_id",
        "teams",
        ["fpl_team_id"],
        unique=True,
    )

    if "ix_players_season" in _index_names("players"):
        op.drop_index("ix_players_season", table_name="players")
    if "ix_teams_season" in _index_names("teams"):
        op.drop_index("ix_teams_season", table_name="teams")

    op.drop_column("players", "season")
    op.drop_column("teams", "season")
