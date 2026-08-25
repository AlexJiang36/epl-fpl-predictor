from sqlalchemy import Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # FPL team IDs are stable only within a season. They are reused across seasons.
    season: Mapped[str] = mapped_column(
        String(16), nullable=False, default="2025_26", index=True
    )
    fpl_team_id: Mapped[int] = mapped_column(Integer, index=True)

    name: Mapped[str] = mapped_column(String(100))
    short_name: Mapped[str] = mapped_column(String(10))

    __table_args__ = (
        UniqueConstraint(
            "season",
            "fpl_team_id",
            name="uq_teams_season_fpl_team_id",
        ),
    )
