"""add gw to fixtures

Revision ID: 4ea18d2b0dde
Revises: d2c76c0c5236
Create Date: 2026-03-14 11:53:00.160746

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4ea18d2b0dde'
down_revision: Union[str, Sequence[str], None] = 'd2c76c0c5236'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column("fixtures", sa.Column("gw", sa.Integer(), nullable=True))
    op.create_index("ix_fixtures_gw", "fixtures", ["gw"])

def downgrade():
    op.drop_index("ix_fixtures_gw", table_name="fixtures")
    op.drop_column("fixtures", "gw")
