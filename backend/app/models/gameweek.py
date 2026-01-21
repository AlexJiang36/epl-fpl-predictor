from typing import Optional
from sqlalchemy import Integer, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

class Gameweek(Base):
    __tablename__ = "gameweeks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # FPL event id (gameweek number)
    gw: Mapped[int] = mapped_column(Integer, unique=True, index=True)

    # deadline_time from FPL (UTC)
    deadline_time: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # status flags from FPL
    is_current: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_next: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_finished: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Optional: useful to store
    name: Mapped[Optional[str]] = mapped_column(nullable=True)