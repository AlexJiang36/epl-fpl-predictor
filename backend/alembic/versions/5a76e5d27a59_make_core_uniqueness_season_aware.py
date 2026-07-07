"""make core uniqueness season aware

Revision ID: 5a76e5d27a59
Revises: day60_add_season_columns
Create Date: 2026-07-07 09:43:51.594704

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5a76e5d27a59'
down_revision: Union[str, Sequence[str], None] = 'day60_add_season_columns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop old single-season uniqueness.
    op.drop_index("ix_gameweeks_gw", table_name="gameweeks")
    op.drop_constraint(
        "uq_player_gw_stats_player_id_gw",
        "player_gw_stats",
        type_="unique",
    )
    op.drop_constraint(
        "uq_predictions_player_gw_model",
        "predictions",
        type_="unique",
    )
    op.drop_constraint(
        "uq_match_predictions_fixture_model",
        "match_predictions",
        type_="unique",
    )

    # Drop old non-unique season composite indexes before replacing them
    # with unique constraints on the same columns.
    op.drop_index("ix_gameweeks_season_gw", table_name="gameweeks")
    op.drop_index(
        "ix_player_gw_stats_season_player_gw",
        table_name="player_gw_stats",
    )
    op.drop_index(
        "ix_predictions_season_player_target_gw_model",
        table_name="predictions",
    )
    op.drop_index(
        "ix_match_predictions_season_fixture_model",
        table_name="match_predictions",
    )

    # Create season-aware uniqueness.
    op.create_unique_constraint(
        "uq_gameweeks_season_gw",
        "gameweeks",
        ["season", "gw"],
    )
    op.create_unique_constraint(
        "uq_player_gw_stats_season_player_id_gw",
        "player_gw_stats",
        ["season", "player_id", "gw"],
    )
    op.create_unique_constraint(
        "uq_predictions_season_player_gw_model",
        "predictions",
        ["season", "player_id", "target_gw", "model_name"],
    )
    op.create_unique_constraint(
        "uq_match_predictions_season_fixture_model",
        "match_predictions",
        ["season", "fixture_id", "model_name"],
    )


def downgrade() -> None:
    # Drop season-aware uniqueness.
    op.drop_constraint(
        "uq_match_predictions_season_fixture_model",
        "match_predictions",
        type_="unique",
    )
    op.drop_constraint(
        "uq_predictions_season_player_gw_model",
        "predictions",
        type_="unique",
    )
    op.drop_constraint(
        "uq_player_gw_stats_season_player_id_gw",
        "player_gw_stats",
        type_="unique",
    )
    op.drop_constraint(
        "uq_gameweeks_season_gw",
        "gameweeks",
        type_="unique",
    )

    # Restore previous non-unique season composite indexes.
    op.create_index(
        "ix_match_predictions_season_fixture_model",
        "match_predictions",
        ["season", "fixture_id", "model_name"],
        unique=False,
    )
    op.create_index(
        "ix_predictions_season_player_target_gw_model",
        "predictions",
        ["season", "player_id", "target_gw", "model_name"],
        unique=False,
    )
    op.create_index(
        "ix_player_gw_stats_season_player_gw",
        "player_gw_stats",
        ["season", "player_id", "gw"],
        unique=False,
    )
    op.create_index(
        "ix_gameweeks_season_gw",
        "gameweeks",
        ["season", "gw"],
        unique=False,
    )

    # Restore old single-season uniqueness.
    op.create_unique_constraint(
        "uq_match_predictions_fixture_model",
        "match_predictions",
        ["fixture_id", "model_name"],
    )
    op.create_unique_constraint(
        "uq_predictions_player_gw_model",
        "predictions",
        ["player_id", "target_gw", "model_name"],
    )
    op.create_unique_constraint(
        "uq_player_gw_stats_player_id_gw",
        "player_gw_stats",
        ["player_id", "gw"],
    )
    op.create_index(
        "ix_gameweeks_gw",
        "gameweeks",
        ["gw"],
        unique=True,
    )
