from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # FPL player/element IDs are stable only within a season. They are reused
    # for different players across seasons, so season is part of the identity.
    season: Mapped[str] = mapped_column(
        String(16), nullable=False, default="2025_26", index=True
    )
    fpl_player_id: Mapped[int] = mapped_column(Integer, index=True)

    first_name: Mapped[str] = mapped_column(String(50))
    second_name: Mapped[str] = mapped_column(String(50))
    web_name: Mapped[str] = mapped_column(String(50), index=True)

    # Link to the canonical Team row for the same season.
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)

    # GKP, DEF, MID, FWD
    position: Mapped[str] = mapped_column(String(3), index=True)

    # FPL uses integer cost = price * 10 (e.g. 75 => 7.5).
    now_cost: Mapped[int] = mapped_column(Integer)

    # Availability status (e.g. "a" available, "i" injured).
    status: Mapped[str] = mapped_column(String(1), index=True)

    team = relationship("Team")

    __table_args__ = (
        UniqueConstraint(
            "season",
            "fpl_player_id",
            name="uq_players_season_fpl_player_id",
        ),
    )
