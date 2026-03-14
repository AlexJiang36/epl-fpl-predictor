"""add match_predictions table

Revision ID: d2c76c0c5236
Revises: 3d86053cdecc
Create Date: 2026-03-14 11:42:29.891916

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd2c76c0c5236'
down_revision: Union[str, Sequence[str], None] = '3d86053cdecc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        "match_predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("fixture_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=50), nullable=False),
        sa.Column("pred_home_win", sa.Float(), nullable=False),
        sa.Column("pred_draw", sa.Float(), nullable=False),
        sa.Column("pred_away_win", sa.Float(), nullable=False),
        sa.Column("pred_result", sa.String(length=10), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["fixture_id"], ["fixtures.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("fixture_id", "model_name", name="uq_match_predictions_fixture_model"),
    )

    op.create_index("ix_match_predictions_fixture_id", "match_predictions", ["fixture_id"])
    op.create_index("ix_match_predictions_model_name", "match_predictions", ["model_name"])


def downgrade():
    op.drop_index("ix_match_predictions_model_name", table_name="match_predictions")
    op.drop_index("ix_match_predictions_fixture_id", table_name="match_predictions")
    op.drop_table("match_predictions")
