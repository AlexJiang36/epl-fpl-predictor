from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.sql import func

from app.core.db import Base


class MatchPrediction(Base):
    __tablename__ = "match_predictions"

    id = Column(Integer, primary_key=True, index=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(String(50), nullable=False)

    pred_home_win = Column(Float, nullable=False)
    pred_draw = Column(Float, nullable=False)
    pred_away_win = Column(Float, nullable=False)
    pred_result = Column(String(10), nullable=False)  # "H" | "D" | "A"

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("fixture_id", "model_name", name="uq_match_predictions_fixture_model"),
        Index("ix_match_predictions_fixture_id", "fixture_id"),
        Index("ix_match_predictions_model_name", "model_name"),
    )