from sqlalchemy import Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class PlayerGameweekStat(Base):
    __tablename__ = "player_gw_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # FK -> players.id
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)

    # Gameweek number (1..38)
    gw: Mapped[int] = mapped_column(Integer, index=True)

    # Minimal stats we need for baseline prediction
    minutes: Mapped[int] = mapped_column(Integer, default=0)
    goals_scored: Mapped[int] = mapped_column(Integer, default=0)
    assists: Mapped[int] = mapped_column(Integer, default=0)
    clean_sheets: Mapped[int] = mapped_column(Integer, default=0)
    total_points: Mapped[int] = mapped_column(Integer, default=0)

    # ORM convenience
    player = relationship("Player")

    __table_args__ = (
        UniqueConstraint("player_id", "gw", name="uq_player_gw_stats_player_id_gw"),
    )
