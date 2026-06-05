"""add season columns to core fact tables

Revision ID: day60_add_season_columns
Revises:
Create Date: 2026-05-31

Day60 phase 1 migration:
- add `season` to core season-sensitive tables
- backfill existing rows to a default season
- add indexes to support season-aware queries

Notes:
- this migration intentionally does NOT add new unique constraints yet
- add those only after application queries/writes are updated and duplicates are audited
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "day60_add_season_columns"
down_revision = "4ea18d2b0dde"
branch_labels = None
depends_on = None


DEFAULT_SEASON = "2025_26"


def upgrade() -> None:
    # 1) add nullable season columns first
    op.add_column("gameweeks", sa.Column("season", sa.String(length=16), nullable=True))
    op.add_column("fixtures", sa.Column("season", sa.String(length=16), nullable=True))
    op.add_column("player_gw_stats", sa.Column("season", sa.String(length=16), nullable=True))
    op.add_column("predictions", sa.Column("season", sa.String(length=16), nullable=True))
    op.add_column("match_predictions", sa.Column("season", sa.String(length=16), nullable=True))

    # 2) backfill current rows
    op.execute(f"UPDATE gameweeks SET season = '{DEFAULT_SEASON}' WHERE season IS NULL")
    op.execute(f"UPDATE fixtures SET season = '{DEFAULT_SEASON}' WHERE season IS NULL")
    op.execute(f"UPDATE player_gw_stats SET season = '{DEFAULT_SEASON}' WHERE season IS NULL")
    op.execute(f"UPDATE predictions SET season = '{DEFAULT_SEASON}' WHERE season IS NULL")
    op.execute(f"UPDATE match_predictions SET season = '{DEFAULT_SEASON}' WHERE season IS NULL")

    # 3) make season non-null once backfill is complete
    op.alter_column("gameweeks", "season", existing_type=sa.String(length=16), nullable=False)
    op.alter_column("fixtures", "season", existing_type=sa.String(length=16), nullable=False)
    op.alter_column("player_gw_stats", "season", existing_type=sa.String(length=16), nullable=False)
    op.alter_column("predictions", "season", existing_type=sa.String(length=16), nullable=False)
    op.alter_column("match_predictions", "season", existing_type=sa.String(length=16), nullable=False)

    # 4) add indexes for common season-aware query patterns
    op.create_index("ix_gameweeks_season_gw", "gameweeks", ["season", "gw"], unique=False)

    op.create_index("ix_fixtures_season_gw", "fixtures", ["season", "gw"], unique=False)
    op.create_index(
        "ix_fixtures_season_home_away_gw",
        "fixtures",
        ["season", "home_team_id", "away_team_id", "gw"],
        unique=False,
    )

    op.create_index(
        "ix_player_gw_stats_season_player_gw",
        "player_gw_stats",
        ["season", "player_id", "gw"],
        unique=False,
    )

    op.create_index(
        "ix_predictions_season_player_target_gw_model",
        "predictions",
        ["season", "player_id", "target_gw", "model_name"],
        unique=False,
    )

    op.create_index(
        "ix_match_predictions_season_fixture_model",
        "match_predictions",
        ["season", "fixture_id", "model_name"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_match_predictions_season_fixture_model", table_name="match_predictions")
    op.drop_index("ix_predictions_season_player_target_gw_model", table_name="predictions")
    op.drop_index("ix_player_gw_stats_season_player_gw", table_name="player_gw_stats")
    op.drop_index("ix_fixtures_season_home_away_gw", table_name="fixtures")
    op.drop_index("ix_fixtures_season_gw", table_name="fixtures")
    op.drop_index("ix_gameweeks_season_gw", table_name="gameweeks")

    op.drop_column("match_predictions", "season")
    op.drop_column("predictions", "season")
    op.drop_column("player_gw_stats", "season")
    op.drop_column("fixtures", "season")
    op.drop_column("gameweeks", "season")
