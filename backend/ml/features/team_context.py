import pandas as pd


def build_team_context_from_team_fixture(team_fixture: pd.DataFrame) -> pd.DataFrame:
    """
    Build reusable team-level, leakage-safe context features at the grain:
        (team_id, gw)

    Expected input columns in team_fixture:
        - gw
        - team_id
        - opponent_team_id
        - is_home
        - goals_for
        - goals_against

    Important assumptions:
        - team_fixture should contain ONE row per team per fixture
          (so a single real match contributes two rows: one for each team)
        - only finished fixtures should be included upstream
        - this function computes "before_gw" and recent features using shift(1),
          so each row for gw=t only uses information from gw < t
    """

    required = {
        "gw",
        "team_id",
        "opponent_team_id",
        "is_home",
        "goals_for",
        "goals_against",
    }
    missing = required - set(team_fixture.columns)
    if missing:
        raise RuntimeError(f"team_fixture is missing required columns: {sorted(missing)}")

    if team_fixture.empty:
        return pd.DataFrame(
            columns=[
                "team_id",
                "gw",
                "fixtures_in_gw",
                "home_ratio_gw",
                "team_goals_for_gw",
                "team_goals_against_gw",
                "team_goals_for_sum_gw",
                "team_goals_against_sum_gw",
                "team_points_gw",
                "team_attack_strength_recent",
                "team_defense_weakness_recent",
                "team_goals_scored_avg_recent",
                "team_goals_conceded_avg_recent",
                "team_points_before_gw",
                "team_matches_before_gw",
                "team_goal_diff_before_gw",
                "team_points_per_match_before_gw",
                "team_goal_diff_per_match_before_gw",
                "team_rank_before_gw",
            ]
        )

    tf = team_fixture.copy()
    tf = tf.sort_values(["team_id", "gw"]).reset_index(drop=True)

    # Points from each finished fixture row
    tf["points_gw_single"] = 0
    tf.loc[tf["goals_for"] > tf["goals_against"], "points_gw_single"] = 3
    tf.loc[tf["goals_for"] == tf["goals_against"], "points_gw_single"] = 1

    # Aggregate to one row per (team_id, gw)
    team_gw = (
        tf.groupby(["team_id", "gw"], as_index=False)
        .agg(
            fixtures_in_gw=("opponent_team_id", "count"),
            home_ratio_gw=("is_home", "mean"),
            team_goals_for_gw=("goals_for", "mean"),
            team_goals_against_gw=("goals_against", "mean"),
            team_goals_for_sum_gw=("goals_for", "sum"),
            team_goals_against_sum_gw=("goals_against", "sum"),
            team_points_gw=("points_gw_single", "sum"),
        )
        .sort_values(["team_id", "gw"])
        .reset_index(drop=True)
    )

    tg = team_gw.groupby("team_id", group_keys=False)

    # Recent rolling strength features (shifted by 1, so leakage-safe)
    team_gw["team_attack_strength_recent"] = (
        tg["team_goals_for_gw"].shift(1).rolling(5, min_periods=1).mean()
    )
    team_gw["team_defense_weakness_recent"] = (
        tg["team_goals_against_gw"].shift(1).rolling(5, min_periods=1).mean()
    )

    # Alias fields with clearer football meaning
    team_gw["team_goals_scored_avg_recent"] = team_gw["team_attack_strength_recent"]
    team_gw["team_goals_conceded_avg_recent"] = team_gw["team_defense_weakness_recent"]

    # Standings context BEFORE current gw
    team_gw["team_points_before_gw"] = tg["team_points_gw"].cumsum().shift(1)
    team_gw["team_matches_before_gw"] = tg["fixtures_in_gw"].cumsum().shift(1)
    team_gw["team_goal_diff_before_gw"] = (
        (tg["team_goals_for_sum_gw"].cumsum() - tg["team_goals_against_sum_gw"].cumsum()).shift(1)
    )

    team_gw["team_points_per_match_before_gw"] = (
        team_gw["team_points_before_gw"] / team_gw["team_matches_before_gw"]
    )
    team_gw["team_goal_diff_per_match_before_gw"] = (
        team_gw["team_goal_diff_before_gw"] / team_gw["team_matches_before_gw"]
    )

    # Rank inside each gw using only before-gw information
    rank_frame = team_gw[
        [
            "team_id",
            "gw",
            "team_points_per_match_before_gw",
            "team_goal_diff_per_match_before_gw",
            "team_points_before_gw",
        ]
    ].copy()

    rank_frame["team_points_per_match_before_gw_rank_sort"] = rank_frame[
        "team_points_per_match_before_gw"
    ].fillna(-9999.0)
    rank_frame["team_goal_diff_per_match_before_gw_rank_sort"] = rank_frame[
        "team_goal_diff_per_match_before_gw"
    ].fillna(-9999.0)
    rank_frame["team_points_before_gw_rank_sort"] = rank_frame[
        "team_points_before_gw"
    ].fillna(-9999.0)

    rank_frame = rank_frame.sort_values(
        [
            "gw",
            "team_points_per_match_before_gw_rank_sort",
            "team_goal_diff_per_match_before_gw_rank_sort",
            "team_points_before_gw_rank_sort",
            "team_id",
        ],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    rank_frame["team_rank_before_gw"] = rank_frame.groupby("gw").cumcount() + 1
    rank_frame = rank_frame[["team_id", "gw", "team_rank_before_gw"]]

    team_gw = team_gw.merge(rank_frame, on=["team_id", "gw"], how="left")

    return team_gw


def attach_opponent_context(
    team_fixture: pd.DataFrame,
    team_context: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach opponent-side context to team_fixture rows.

    Returns one row per team per fixture with additional columns such as:
        - opponent_rank_before_gw_single
        - opponent_points_per_match_before_gw_single
        - opponent_goal_diff_per_match_before_gw_single
        - opponent_goals_scored_avg_recent_single
        - opponent_goals_conceded_avg_recent_single
    """

    if team_fixture.empty:
        return team_fixture.copy()

    tf = team_fixture.copy()

    opp_context = team_context[
        [
            "team_id",
            "gw",
            "team_rank_before_gw",
            "team_points_per_match_before_gw",
            "team_goal_diff_per_match_before_gw",
            "team_goals_scored_avg_recent",
            "team_goals_conceded_avg_recent",
        ]
    ].rename(
        columns={
            "team_id": "opponent_team_id",
            "team_rank_before_gw": "opponent_rank_before_gw_single",
            "team_points_per_match_before_gw": "opponent_points_per_match_before_gw_single",
            "team_goal_diff_per_match_before_gw": "opponent_goal_diff_per_match_before_gw_single",
            "team_goals_scored_avg_recent": "opponent_goals_scored_avg_recent_single",
            "team_goals_conceded_avg_recent": "opponent_goals_conceded_avg_recent_single",
        }
    )

    return tf.merge(
        opp_context,
        on=["opponent_team_id", "gw"],
        how="left",
    )


def build_team_fixture_context(team_fixture: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience helper:
    build team_gw context, attach opponent context, and add common
    single-fixture difficulty signals.

    Returns one row per team per fixture.
    """

    if team_fixture.empty:
        return team_fixture.copy()

    team_context = build_team_context_from_team_fixture(team_fixture)
    tf = attach_opponent_context(team_fixture, team_context)

    # Keep legacy single proxy for continuity
    tf["fixture_difficulty_proxy_single"] = 1.0 / (
        tf["opponent_goals_conceded_avg_recent_single"].fillna(1.0) + 0.5
    )

    # Higher = more difficult
    tf["fixture_difficulty_for_attackers_single"] = 1.0 / (
        tf["opponent_goals_conceded_avg_recent_single"].fillna(1.0) + 0.5
    )
    tf["fixture_difficulty_for_defenders_single"] = (
        tf["opponent_goals_scored_avg_recent_single"].fillna(0.0)
    )

    return tf
