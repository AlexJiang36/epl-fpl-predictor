import unittest

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base
from app.api.routes.ingest import ingest_bootstrap_document, ingest_fixture_document
from app.api.routes.ingest_gw_stats import ingest_one_gw_document
from app.models.fixture import Fixture
from app.models.player import Player
from app.models.player_gw_stat import PlayerGameweekStat
from app.models.team import Team


class SeasonizedLiveIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(
            self.engine,
            tables=[
                Team.__table__,
                Player.__table__,
                Fixture.__table__,
                PlayerGameweekStat.__table__,
            ],
        )
        Session = sessionmaker(bind=self.engine)
        self.db = Session()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    @staticmethod
    def bootstrap(team_name: str, player_name: str):
        return {
            "teams": [
                {
                    "id": 1,
                    "name": team_name,
                    "short_name": team_name[:3].upper(),
                }
            ],
            "elements": [
                {
                    "id": 1,
                    "first_name": player_name,
                    "second_name": "Example",
                    "web_name": player_name,
                    "team": 1,
                    "element_type": 3,
                    "now_cost": 50,
                    "status": "a",
                }
            ],
        }

    def test_same_external_ids_can_coexist_across_seasons(self) -> None:
        ingest_bootstrap_document(
            self.db,
            self.bootstrap("Old Team", "Old Player"),
            "2025_26",
        )
        ingest_bootstrap_document(
            self.db,
            self.bootstrap("New Team", "New Player"),
            "2026_27",
        )

        teams = self.db.execute(select(Team).order_by(Team.season)).scalars().all()
        players = self.db.execute(select(Player).order_by(Player.season)).scalars().all()

        self.assertEqual(2, len(teams))
        self.assertEqual(2, len(players))
        self.assertEqual(["Old Team", "New Team"], [row.name for row in teams])
        self.assertEqual(["Old Player", "New Player"], [row.web_name for row in players])
        self.assertNotEqual(teams[0].id, teams[1].id)
        self.assertNotEqual(players[0].id, players[1].id)
        self.assertEqual(teams[0].id, players[0].team_id)
        self.assertEqual(teams[1].id, players[1].team_id)

    def test_fixture_external_id_is_season_scoped(self) -> None:
        ingest_bootstrap_document(
            self.db,
            self.bootstrap("Old Team", "Old Player"),
            "2025_26",
        )
        ingest_bootstrap_document(
            self.db,
            self.bootstrap("New Team", "New Player"),
            "2026_27",
        )

        # Add a second team in each season so the fixture has valid endpoints.
        for season, name in [("2025_26", "Old Away"), ("2026_27", "New Away")]:
            self.db.add(
                Team(
                    season=season,
                    fpl_team_id=2,
                    name=name,
                    short_name="AWY",
                )
            )
        self.db.commit()

        fixture = {
            "id": 1,
            "team_h": 1,
            "team_a": 2,
            "event": 1,
            "kickoff_time": "2026-08-21T17:30:00Z",
            "finished": False,
            "team_h_score": None,
            "team_a_score": None,
        }
        ingest_fixture_document(self.db, [fixture], "2025_26")
        ingest_fixture_document(self.db, [fixture], "2026_27")

        rows = self.db.execute(
            select(Fixture).order_by(Fixture.season)
        ).scalars().all()
        self.assertEqual(2, len(rows))
        self.assertEqual([1, 1], [row.fpl_fixture_id for row in rows])
        self.assertNotEqual(rows[0].home_team_id, rows[1].home_team_id)

    def test_gw_actuals_map_to_player_in_requested_season(self) -> None:
        ingest_bootstrap_document(
            self.db,
            self.bootstrap("Old Team", "Old Player"),
            "2025_26",
        )
        ingest_bootstrap_document(
            self.db,
            self.bootstrap("New Team", "New Player"),
            "2026_27",
        )

        data = {
            "elements": [
                {
                    "id": 1,
                    "stats": {
                        "minutes": 90,
                        "goals_scored": 1,
                        "assists": 0,
                        "clean_sheets": 0,
                        "total_points": 8,
                    },
                }
            ]
        }
        result = ingest_one_gw_document(
            self.db,
            gw=1,
            season="2026_27",
            data=data,
        )

        self.assertEqual(1, result["inserted"])
        stat = self.db.execute(select(PlayerGameweekStat)).scalar_one()
        player = self.db.get(Player, stat.player_id)
        self.assertEqual("2026_27", stat.season)
        self.assertEqual("2026_27", player.season)
        self.assertEqual("New Player", player.web_name)


if __name__ == "__main__":
    unittest.main()
