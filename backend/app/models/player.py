from sqlalchemy import String, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.db import Base

class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # FPL player id (stable unique id from the FPL dataset)
    fpl_player_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)

    first_name: Mapped[str] = mapped_column(String(50))
    second_name: Mapped[str] = mapped_column(String(50))
    web_name: Mapped[str] = mapped_column(String(50), index=True)

    # Link to teams table
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)

    # GKP, DEF, MID, FWD
    position: Mapped[str] = mapped_column(String(3), index=True)

    # FPL uses integer cose = price * 10 (e.g. 75 => 7.5)
    now_cost: Mapped[int] = mapped_column(Integer)

    # availability status (e.g. "a" available, "i" injured)
    status: Mapped[str] = mapped_column(String(1), index=True)

    team = relationship("Team")

