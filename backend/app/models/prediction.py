from datetime import datetime

from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, func, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base

class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("player_id", "target_gw", "model_name", name="uq_predictions_player_gw_model"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"),  index=True)

    target_gw: Mapped[int] = mapped_column(Integer, index=True)

    model_name: Mapped[str] = mapped_column(String(50), index=True)

    predicted_points: Mapped[float] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

