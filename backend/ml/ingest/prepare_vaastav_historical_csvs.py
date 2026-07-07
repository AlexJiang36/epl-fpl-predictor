from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


PLAYER_GW_CANDIDATES = [
    "gws/merged_gw.csv",
    "gws/merged_gws.csv",
]

FIXTURES_CANDIDATES = [
    "fixtures.csv",
]

TEAMS_CANDIDATES = [
    "teams.csv",
]

PLAYERS_RAW_CANDIDATES = [
    "players_raw.csv",
    "cleaned_players.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare historical FPL CSVs from a local Vaastav Fantasy-Premier-League "
            "data directory. This command only writes normalized CSV files; it does not "
            "write to the application database."
        )
    )
    parser.add_argument(
        "--vaastav-data-dir",
        required=True,
        help="Path to Vaastav Fantasy-Premier-League/data directory.",
    )
    parser.add_argument(
        "--season",
        required=True,
        help="Canonical season key for this project, for example 2024_25.",
    )
    parser.add_argument(
        "--source-season-folder",
        default=None,
        help="Optional Vaastav folder name, for example 2024-25. Defaults from --season.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to data/raw/historical/{season}.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def season_to_vaastav_folder(season: str) -> str:
    return season.replace("_", "-")


def find_first_existing(base_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for rel_path in candidates:
        path = base_dir / rel_path
        if path.exists():
            return path
    return None


def normalize_bool_series(series: pd.Series) -> pd.Series:
    def _to_bool(value: Any) -> Optional[bool]:
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
        return None

    return series.apply(_to_bool)


def first_existing_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    lower_to_original = {col.lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]
    return None


def first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def last_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[-1]


def single_value_or_none(series: pd.Series) -> Any:
    non_null = series.dropna()
    unique_values = non_null.astype(str).unique()
    if len(unique_values) == 1:
        return non_null.iloc[0]
    return None


def unique_join(series: pd.Series) -> Optional[str]:
    non_null = series.dropna().astype(str)
    values = sorted(set(non_null.tolist()))
    if not values:
        return None
    return "|".join(values)


def prepare_player_gw(
    *,
    source_path: Path,
    season: str,
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Any]]:
    df = pd.read_csv(source_path)
    errors: List[str] = []
    warnings: List[str] = []
    diagnostics: Dict[str, Any] = {
        "source_row_count": int(len(df)),
        "duplicate_element_gw_rows_before_aggregation": 0,
        "aggregated_row_count": 0,
        "aggregation_notes": [],
    }

    element_col = first_existing_column(df, ["element", "player_id", "id"])
    gw_col = first_existing_column(df, ["round", "event", "gw"])
    minutes_col = first_existing_column(df, ["minutes"])
    total_points_col = first_existing_column(df, ["total_points", "points"])

    required_resolved = {
        "element": element_col,
        "gw": gw_col,
        "minutes": minutes_col,
        "total_points": total_points_col,
    }

    for logical_name, col in required_resolved.items():
        if col is None:
            errors.append(
                "player GW source is missing logical column %s. Available columns: %s"
                % (logical_name, list(df.columns))
            )

    if errors:
        return pd.DataFrame(), errors, warnings, diagnostics

    out = pd.DataFrame(index=df.index)
    out["season"] = season
    out["element"] = df[element_col]
    out["gw"] = df[gw_col]
    out["minutes"] = df[minutes_col]
    out["total_points"] = df[total_points_col]

    optional_direct = [
        "goals_scored",
        "assists",
        "clean_sheets",
        "bonus",
        "value",
        "was_home",
        "opponent_team",
    ]
    for col in optional_direct:
        source_col = first_existing_column(df, [col])
        if source_col is not None:
            out[col] = df[source_col]

    name_col = first_existing_column(df, ["name", "web_name", "player_name"])
    if name_col is not None:
        out["raw_player_name"] = df[name_col]

    fixture_col = first_existing_column(df, ["fixture", "fixture_id"])
    if fixture_col is not None:
        out["raw_fixture_id"] = df[fixture_col]

    if "was_home" in out.columns:
        out["was_home"] = normalize_bool_series(out["was_home"])

    for col in ["season", "element", "gw", "minutes", "total_points"]:
        null_count = int(out[col].isna().sum())
        if null_count > 0:
            errors.append("prepared player_gw_stats column %s has %s null values before aggregation." % (col, null_count))

    if errors:
        return pd.DataFrame(), errors, warnings, diagnostics

    duplicate_row_count = int(out.duplicated(subset=["element", "gw"]).sum())
    diagnostics["duplicate_element_gw_rows_before_aggregation"] = duplicate_row_count

    if duplicate_row_count > 0:
        warnings.append(
            "player GW source has %s duplicate element + gw rows. "
            "This is expected for double gameweeks; adapter will aggregate to one player-GW row."
            % duplicate_row_count
        )

    agg_spec: Dict[str, Any] = {
        "season": first_non_null,
        "minutes": "sum",
        "total_points": "sum",
    }

    for col in ["goals_scored", "assists", "clean_sheets", "bonus"]:
        if col in out.columns:
            agg_spec[col] = "sum"

    if "value" in out.columns:
        agg_spec["value"] = last_non_null

    if "was_home" in out.columns:
        agg_spec["was_home"] = single_value_or_none
        diagnostics["aggregation_notes"].append(
            "was_home is preserved only when all source rows for element+gw agree; otherwise blank."
        )

    if "opponent_team" in out.columns:
        agg_spec["opponent_team"] = single_value_or_none
        diagnostics["aggregation_notes"].append(
            "opponent_team is preserved only when all source rows for element+gw agree; otherwise blank."
        )

    if "raw_player_name" in out.columns:
        agg_spec["raw_player_name"] = first_non_null

    if "raw_fixture_id" in out.columns:
        agg_spec["raw_fixture_id"] = unique_join

    grouped = (
        out.groupby(["element", "gw"], as_index=False, dropna=False)
        .agg(agg_spec)
        .sort_values(["gw", "element"])
        .reset_index(drop=True)
    )

    ordered_cols = ["season", "element", "gw", "minutes", "total_points"]
    remaining_cols = [col for col in grouped.columns if col not in set(ordered_cols)]
    grouped = grouped[ordered_cols + remaining_cols]

    diagnostics["aggregated_row_count"] = int(len(grouped))

    duplicate_after = int(grouped.duplicated(subset=["element", "gw"]).sum())
    if duplicate_after > 0:
        errors.append("prepared player_gw_stats still has %s duplicate element + gw rows after aggregation." % duplicate_after)

    null_season_after = int(grouped["season"].isna().sum())
    if null_season_after > 0:
        errors.append("prepared player_gw_stats has %s null season values after aggregation." % null_season_after)

    return grouped, errors, warnings, diagnostics


def prepare_fixtures(
    *,
    source_path: Path,
    season: str,
) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Any]]:
    df = pd.read_csv(source_path)
    errors: List[str] = []
    warnings: List[str] = []
    diagnostics: Dict[str, Any] = {"source_row_count": int(len(df))}

    id_col = first_existing_column(df, ["id", "fixture_id", "fpl_fixture_id"])
    gw_col = first_existing_column(df, ["event", "gw", "round"])
    team_h_col = first_existing_column(df, ["team_h", "home_team_id", "home_team"])
    team_a_col = first_existing_column(df, ["team_a", "away_team_id", "away_team"])

    required_resolved = {
        "gw": gw_col,
        "team_h": team_h_col,
        "team_a": team_a_col,
    }

    for logical_name, col in required_resolved.items():
        if col is None:
            errors.append(
                "fixtures source is missing logical column %s. Available columns: %s"
                % (logical_name, list(df.columns))
            )

    if errors:
        return pd.DataFrame(), errors, warnings, diagnostics

    out = pd.DataFrame(index=df.index)
    out["season"] = season
    if id_col is not None:
        out["fpl_fixture_id"] = df[id_col]
    out["gw"] = df[gw_col]
    out["team_h"] = df[team_h_col]
    out["team_a"] = df[team_a_col]

    optional_map = {
        "kickoff_time": ["kickoff_time", "kickoff", "kickoff_date"],
        "team_h_score": ["team_h_score", "home_score", "home_goals"],
        "team_a_score": ["team_a_score", "away_score", "away_goals"],
        "finished": ["finished", "complete", "is_finished"],
    }

    for output_col, aliases in optional_map.items():
        source_col = first_existing_column(df, aliases)
        if source_col is not None:
            out[output_col] = df[source_col]
        else:
            warnings.append("fixtures source missing recommended column: %s" % output_col)

    if "finished" in out.columns:
        out["finished"] = normalize_bool_series(out["finished"])

    duplicate_count = int(out.duplicated(subset=["gw", "team_h", "team_a"]).sum())
    if duplicate_count > 0:
        errors.append("prepared fixtures has %s duplicate gw + team_h + team_a rows." % duplicate_count)

    same_team = out["team_h"].astype(str) == out["team_a"].astype(str)
    if bool(same_team.any()):
        errors.append("prepared fixtures has %s rows where team_h == team_a." % int(same_team.sum()))

    for col in ["season", "gw", "team_h", "team_a"]:
        null_count = int(out[col].isna().sum())
        if null_count > 0:
            errors.append("prepared fixtures column %s has %s null values." % (col, null_count))

    return out, errors, warnings, diagnostics


def prepare_team_mapping_template(
    *,
    teams_path: Optional[Path],
    fixtures_df: Optional[pd.DataFrame],
    season: str,
) -> pd.DataFrame:
    if teams_path and teams_path.exists():
        teams = pd.read_csv(teams_path)
        id_col = first_existing_column(teams, ["id", "team_id"])
        name_col = first_existing_column(teams, ["name", "team_name"])
        short_col = first_existing_column(teams, ["short_name", "shortname", "abbreviation"])

        out = pd.DataFrame(index=teams.index)
        out["season"] = season
        if id_col is not None:
            out["raw_team_id"] = teams[id_col]
        else:
            out["raw_team_id"] = None
        if name_col is not None:
            out["raw_team_name"] = teams[name_col]
        else:
            out["raw_team_name"] = None
        if short_col is not None:
            out["raw_team_short_name"] = teams[short_col]
        else:
            out["raw_team_short_name"] = None
    else:
        raw_ids = []
        if fixtures_df is not None and not fixtures_df.empty:
            raw_ids = sorted(
                set(fixtures_df["team_h"].dropna().astype(str).tolist())
                | set(fixtures_df["team_a"].dropna().astype(str).tolist())
            )
        out = pd.DataFrame({"season": season, "raw_team_id": raw_ids})
        out["raw_team_name"] = None
        out["raw_team_short_name"] = None

    out["canonical_team_id"] = None
    out["canonical_team_name"] = None
    out["mapping_confidence"] = None
    out["notes"] = None
    return out


def prepare_player_mapping_template(
    *,
    players_path: Optional[Path],
    player_gw_df: Optional[pd.DataFrame],
    season: str,
) -> pd.DataFrame:
    if players_path and players_path.exists():
        players = pd.read_csv(players_path)
        raw_col = first_existing_column(players, ["id", "element", "player_id", "fpl_player_id"])
        web_name_col = first_existing_column(players, ["web_name", "name", "player_name"])
        first_name_col = first_existing_column(players, ["first_name"])
        second_name_col = first_existing_column(players, ["second_name"])
        team_col = first_existing_column(players, ["team", "team_id"])
        position_col = first_existing_column(players, ["element_type", "position"])

        out = pd.DataFrame(index=players.index)
        out["season"] = season
        if raw_col is not None:
            out["raw_player_id"] = players[raw_col]
        else:
            out["raw_player_id"] = None

        if web_name_col is not None:
            out["raw_player_name"] = players[web_name_col]
        elif first_name_col is not None and second_name_col is not None:
            out["raw_player_name"] = (
                players[first_name_col].fillna("").astype(str).str.strip()
                + " "
                + players[second_name_col].fillna("").astype(str).str.strip()
            ).str.strip()
        else:
            out["raw_player_name"] = None

        if team_col is not None:
            out["raw_team_id"] = players[team_col]
        else:
            out["raw_team_id"] = None

        if position_col is not None:
            out["raw_position"] = players[position_col]
        else:
            out["raw_position"] = None
    else:
        if player_gw_df is not None and not player_gw_df.empty:
            temp = player_gw_df[["element"]].drop_duplicates().rename(columns={"element": "raw_player_id"})
            if "raw_player_name" in player_gw_df.columns:
                names = player_gw_df[["element", "raw_player_name"]].drop_duplicates(subset=["element"])
                temp = temp.merge(names, left_on="raw_player_id", right_on="element", how="left").drop(columns=["element"])
            else:
                temp["raw_player_name"] = None
            out = temp
        else:
            out = pd.DataFrame(columns=["raw_player_id", "raw_player_name"])

        out.insert(0, "season", season)
        out["raw_team_id"] = None
        out["raw_position"] = None

    out["canonical_player_id"] = None
    out["canonical_player_name"] = None
    out["mapping_confidence"] = None
    out["notes"] = None
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_report(report: Dict[str, Any], out_json: Optional[str]) -> None:
    if not out_json:
        return
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("saved_report:", path)


def print_summary(report: Dict[str, Any]) -> None:
    print("=== Vaastav Historical Source Adapter ===")
    print("season:", report["season"])
    print("source_season_folder:", report["source_season_folder"])
    print("passed:", report["passed"])
    print()

    if report["errors"]:
        print("Errors:")
        for error in report["errors"]:
            print("-", error)
        print()

    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print("-", warning)
        print()

    print("Source files:")
    for key, value in sorted(report["source_files"].items()):
        print("- %s: %s" % (key, value))
    print()

    print("Output files:")
    for key, value in sorted(report["output_files"].items()):
        print("- %s: %s" % (key, value))
    print()

    print("Row counts:")
    for key, value in sorted(report["row_counts"].items()):
        print("- %s: %s" % (key, value))
    print()

    print("Diagnostics:")
    for key, value in sorted(report["diagnostics"].items()):
        print("- %s: %s" % (key, value))
    print()


def main() -> None:
    args = parse_args()

    vaastav_data_dir = Path(args.vaastav_data_dir)
    source_season_folder = args.source_season_folder or season_to_vaastav_folder(args.season)
    season_dir = vaastav_data_dir / source_season_folder
    out_dir = Path(args.out_dir) if args.out_dir else Path("data/raw/historical") / args.season

    errors: List[str] = []
    warnings: List[str] = []
    diagnostics: Dict[str, Any] = {}

    if not vaastav_data_dir.exists():
        errors.append("vaastav_data_dir does not exist: %s" % vaastav_data_dir)

    if not season_dir.exists():
        errors.append("season directory does not exist: %s" % season_dir)

    player_gw_source = find_first_existing(season_dir, PLAYER_GW_CANDIDATES) if not errors else None
    fixtures_source = find_first_existing(season_dir, FIXTURES_CANDIDATES) if not errors else None
    teams_source = find_first_existing(season_dir, TEAMS_CANDIDATES) if not errors else None
    players_source = find_first_existing(season_dir, PLAYERS_RAW_CANDIDATES) if not errors else None

    if not errors and player_gw_source is None:
        errors.append("Could not find player GW file. Tried: %s" % PLAYER_GW_CANDIDATES)
    if not errors and fixtures_source is None:
        errors.append("Could not find fixtures file. Tried: %s" % FIXTURES_CANDIDATES)

    player_gw_df: Optional[pd.DataFrame] = None
    fixtures_df: Optional[pd.DataFrame] = None
    team_mapping_df = pd.DataFrame()
    player_mapping_df = pd.DataFrame()

    output_files = {
        "player_gw_stats": str(out_dir / "player_gw_stats.csv"),
        "fixtures": str(out_dir / "fixtures.csv"),
        "team_mapping_template": str(out_dir / "team_mapping_template.csv"),
        "player_mapping_template": str(out_dir / "player_mapping_template.csv"),
    }

    if not errors and player_gw_source is not None:
        player_gw_df, player_errors, player_warnings, player_diagnostics = prepare_player_gw(
            source_path=player_gw_source,
            season=args.season,
        )
        errors.extend(player_errors)
        warnings.extend(player_warnings)
        diagnostics["player_gw"] = player_diagnostics

    if not errors and fixtures_source is not None:
        fixtures_df, fixture_errors, fixture_warnings, fixture_diagnostics = prepare_fixtures(
            source_path=fixtures_source,
            season=args.season,
        )
        errors.extend(fixture_errors)
        warnings.extend(fixture_warnings)
        diagnostics["fixtures"] = fixture_diagnostics

    if not errors:
        team_mapping_df = prepare_team_mapping_template(
            teams_path=teams_source,
            fixtures_df=fixtures_df,
            season=args.season,
        )
        player_mapping_df = prepare_player_mapping_template(
            players_path=players_source,
            player_gw_df=player_gw_df,
            season=args.season,
        )

        write_csv(player_gw_df if player_gw_df is not None else pd.DataFrame(), out_dir / "player_gw_stats.csv")
        write_csv(fixtures_df if fixtures_df is not None else pd.DataFrame(), out_dir / "fixtures.csv")
        write_csv(team_mapping_df, out_dir / "team_mapping_template.csv")
        write_csv(player_mapping_df, out_dir / "player_mapping_template.csv")

    report = {
        "created_at": utc_now(),
        "season": args.season,
        "source_season_folder": source_season_folder,
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "source_files": {
            "player_gw_source": str(player_gw_source) if player_gw_source else None,
            "fixtures_source": str(fixtures_source) if fixtures_source else None,
            "teams_source": str(teams_source) if teams_source else None,
            "players_source": str(players_source) if players_source else None,
        },
        "output_files": output_files,
        "row_counts": {
            "player_gw_stats": int(len(player_gw_df)) if player_gw_df is not None else 0,
            "fixtures": int(len(fixtures_df)) if fixtures_df is not None else 0,
            "team_mapping_template": int(len(team_mapping_df)) if not team_mapping_df.empty else 0,
            "player_mapping_template": int(len(player_mapping_df)) if not player_mapping_df.empty else 0,
        },
        "diagnostics": diagnostics,
        "notes": [
            "This adapter writes normalized raw CSVs only. It does not write to the application database.",
            "Mapping templates intentionally leave canonical IDs blank. Fill or validate them before DB import.",
            "Player double-gameweek fixture rows are aggregated to one player-GW row for canonical player_gw_stats import.",
            "Use import_historical_season.py with --dry-run before any --write import.",
        ],
    }

    print_summary(report)
    save_report(report, args.out_json)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
