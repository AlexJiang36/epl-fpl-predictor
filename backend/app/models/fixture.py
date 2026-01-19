from typing import Optional

from sqlalchemy import Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

class Fixture(Base):
    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # FPL fixture id (unique id from FPL)
    fpl_fixture_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)

    # Teams
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)

    # Kickoff time in UTC (from FPL)
    kickoff_time: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Status
    finished: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Scores (nullable before match finishes)
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)