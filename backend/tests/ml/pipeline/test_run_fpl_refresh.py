from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml.pipeline.run_fpl_refresh import (
    build_market_csv,
    build_top10_by_position,
)


class WeeklyRunnerPureTests(unittest.TestCase):
    def sample_predictions(self) -> pd.DataFrame:
        rows = []
        pid = 1
        for position in ("GKP", "DEF", "MID", "FWD"):
            for rank in range(12):
                rows.append(
                    {
                        "player_id": 1000 + pid,
                        "fpl_player_id": pid,
                        "web_name": "%s_%02d" % (position, rank),
                        "position": position,
                        "team_id": 1 + (rank % 4),
                        "team_name": "Team %s" % (1 + (rank % 4)),
                        "team_short_name": "T%s" % (1 + (rank % 4)),
                        "now_cost": 45 + rank,
                        "status": "a",
                        "chance_of_playing_next_round": None,
                        "predicted_points": float(rank),
                        "expected_minutes_total": 60.0,
                        "blended_start_probability": 0.7,
                        "blended_appearance_probability": 0.9,
                        "news": "",
                    }
                )
                pid += 1
        return pd.DataFrame(rows)

    def test_market_contract_is_exact_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "pred.csv"
            out = root / "market.csv"
            self.sample_predictions().to_csv(src, index=False)
            build_market_csv(src, out)
            market = pd.read_csv(out)
            self.assertEqual(
                list(market.columns),
                [
                    "player_id",
                    "fpl_player_id",
                    "web_name",
                    "position",
                    "team_id",
                    "team_name",
                    "team_short_name",
                    "now_cost",
                    "status",
                    "predicted_points",
                ],
            )
            self.assertEqual(len(market), 48)

    def test_top10_is_ten_per_position_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "pred.csv"
            self.sample_predictions().to_csv(src, index=False)
            _, _, _, grouped = build_top10_by_position(src, root, 10)
            self.assertEqual(set(grouped), {"GKP", "DEF", "MID", "FWD"})
            for position, rows in grouped.items():
                self.assertEqual(len(rows), 10)
                values = [float(row["predicted_points"]) for row in rows]
                self.assertEqual(values, sorted(values, reverse=True))


if __name__ == "__main__":
    unittest.main()
