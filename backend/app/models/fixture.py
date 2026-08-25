from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class Fixture(Base):
    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    season: Mapped[str] = mapped_column(
        String(16), nullable=False, default="2025_26", index=True
    )

    # FPL fixture IDs are reused across seasons, so season is part of identity.
    fpl_fixture_id: Mapped[int] = mapped_column(Integer, index=True)

    # Teams
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)

    gw = Column(Integer, nullable=True, index=True)

    # Kickoff time in UTC (from FPL)
    kickoff_time: Mapped[Optional[DateTime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Status
    finished: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Scores (nullable before match finishes)
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "season",
            "fpl_fixture_id",
            name="uq_fixtures_season_fpl_fixture_id",
        ),
    )
