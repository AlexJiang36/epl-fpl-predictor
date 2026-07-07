"""create historical staging tables

Revision ID: b8f3a2c1d4e6
Revises: 5a76e5d27a59
Create Date: 2026-07-07
"""

from alembic import op
import sqlalchemy as sa


revision = "b8f3a2c1d4e6"
down_revision = "5a76e5d27a59"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "historical_teams",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("season", sa.String(), nullable=False),
        sa.Column("raw_team_id", sa.String(), nullable=False),
        sa.Column("raw_team_name", sa.String(), nullable=True),
        sa.Column("raw_team_short_name", sa.String(), nullable=True),
        sa.Column("canonical_team_id", sa.Integer(), nullable=True),
        sa.Column("canonical_team_name", sa.String(), nullable=True),
        sa.Column("mapping_status", sa.String(), nullable=True),
        sa.Column("mapping_confidence", sa.String(), nullable=True),
        sa.Column("notes", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["canonical_team_id"], ["teams.id"]),
        sa.UniqueConstraint("season", "raw_team_id", name="uq_historical_teams_season_raw_team_id"),
    )
    op.create_index(
        "ix_historical_teams_season",
        "historical_teams",
        ["season"],
        unique=False,
    )

    op.create_table(
        "historical_players",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("season", sa.String(), nullable=False),
        sa.Column("raw_player_id", sa.String(), nullable=False),
        sa.Column("raw_player_name", sa.String(), nullable=True),
        sa.Column("raw_team_id", sa.String(), nullable=True),
        sa.Column("raw_position", sa.String(), nullable=True),
        sa.Column("canonical_player_id", sa.Integer(), nullable=True),
        sa.Column("canonical_player_name", sa.String(), nullable=True),
        sa.Column("mapping_status", sa.String(), nullable=True),
        sa.Column("mapping_confidence", sa.String(), nullable=True),
        sa.Column("notes", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["canonical_player_id"], ["players.id"]),
        sa.UniqueConstraint("season", "raw_player_id", name="uq_historical_players_season_raw_player_id"),
    )
    op.create_index(
        "ix_historical_players_season",
        "historical_players",
        ["season"],
        unique=False,
    )

    op.create_table(
        "historical_fixtures",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("season", sa.String(), nullable=False),
        sa.Column("raw_fixture_id", sa.String(), nullable=True),
        sa.Column("gw", sa.Integer(), nullable=False),
        sa.Column("raw_home_team_id", sa.String(), nullable=False),
        sa.Column("raw_away_team_id", sa.String(), nullable=False),
        sa.Column("canonical_home_team_id", sa.Integer(), nullable=True),
        sa.Column("canonical_away_team_id", sa.Integer(), nullable=True),
        sa.Column("kickoff_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished", sa.Boolean(), nullable=True),
        sa.Column("home_score", sa.Integer(), nullable=True),
        sa.Column("away_score", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["canonical_home_team_id"], ["teams.id"]),
        sa.ForeignKeyConstraint(["canonical_away_team_id"], ["teams.id"]),
        sa.UniqueConstraint(
            "season",
            "gw",
            "raw_home_team_id",
            "raw_away_team_id",
            name="uq_historical_fixtures_season_gw_raw_teams",
        ),
    )
    op.create_index(
        "ix_historical_fixtures_season_gw",
        "historical_fixtures",
        ["season", "gw"],
        unique=False,
    )

    op.create_table(
        "historical_player_gw_stats",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("season", sa.String(), nullable=False),
        sa.Column("raw_player_id", sa.String(), nullable=False),
        sa.Column("gw", sa.Integer(), nullable=False),
        sa.Column("minutes", sa.Integer(), nullable=False),
        sa.Column("goals_scored", sa.Integer(), nullable=True),
        sa.Column("assists", sa.Integer(), nullable=True),
        sa.Column("clean_sheets", sa.Integer(), nullable=True),
        sa.Column("total_points", sa.Integer(), nullable=False),
        sa.Column("bonus", sa.Integer(), nullable=True),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("was_home", sa.Boolean(), nullable=True),
        sa.Column("raw_opponent_team_id", sa.String(), nullable=True),
        sa.Column("raw_fixture_ids", sa.String(), nullable=True),
        sa.UniqueConstraint(
            "season",
            "raw_player_id",
            "gw",
            name="uq_historical_player_gw_stats_season_raw_player_gw",
        ),
    )
    op.create_index(
        "ix_historical_player_gw_stats_season_gw",
        "historical_player_gw_stats",
        ["season", "gw"],
        unique=False,
    )
    op.create_index(
        "ix_historical_player_gw_stats_season_raw_player_id",
        "historical_player_gw_stats",
        ["season", "raw_player_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_historical_player_gw_stats_season_raw_player_id",
        table_name="historical_player_gw_stats",
    )
    op.drop_index(
        "ix_historical_player_gw_stats_season_gw",
        table_name="historical_player_gw_stats",
    )
    op.drop_table("historical_player_gw_stats")

    op.drop_index("ix_historical_fixtures_season_gw", table_name="historical_fixtures")
    op.drop_table("historical_fixtures")

    op.drop_index("ix_historical_players_season", table_name="historical_players")
    op.drop_table("historical_players")

    op.drop_index("ix_historical_teams_season", table_name="historical_teams")
    op.drop_table("historical_teams")
