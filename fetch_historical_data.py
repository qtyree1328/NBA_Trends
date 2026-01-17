import json
import time
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import TeamGameLogs
from nba_api.stats.static import teams as nba_teams
from tqdm.auto import tqdm
from scipy.stats import pearsonr

# ----------------------------
# PARAMETERS
# ----------------------------
SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
]
WINDOWS = [10, 20, 30, 40, "full"]
SLEEP_BETWEEN_CALLS_SEC = 0.7  # Slightly higher for bulk requests
OUTPUT_FILE = "historical_trends_data.json"

# ----------------------------
# Helper Functions (from fetch_nba_data.py)
# ----------------------------
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")

def rugosity(y):
    """Calculate rugosity (arc-to-chord ratio) for a time series."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return 1.0

    # Path length: sum of distances between consecutive points
    path_length = 0.0
    for i in range(len(y) - 1):
        dx = 1.0
        dy = y[i + 1] - y[i]
        path_length += np.sqrt(dx**2 + dy**2)

    # Chord length: straight line from start to end
    total_dx = len(y) - 1
    total_dy = y[-1] - y[0]
    chord_length = np.sqrt(total_dx**2 + total_dy**2)

    if chord_length == 0:
        return 1.0

    return path_length / chord_length

def linear_regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0, 0.0, 0.0, []
    m, b = np.polyfit(x, y, 1)
    y_hat = m * x + b
    resid_std = float(np.std(y - y_hat))
    return float(m), float(b), resid_std, y_hat.tolist()

def rolling_mean(arr, window):
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().tolist()

# ----------------------------
# Calculate Metrics for a Window
# ----------------------------
def calculate_windowed_metrics(off_ratings, def_ratings, window):
    """Calculate rugosity, net rating, and trend score for a specific window."""
    if window == "full":
        off_slice = off_ratings
        def_slice = def_ratings
    else:
        n = int(window)
        off_slice = off_ratings[-n:] if len(off_ratings) >= n else off_ratings
        def_slice = def_ratings[-n:] if len(def_ratings) >= n else def_ratings

    if len(off_slice) < 2:
        return None

    off_rug = rugosity(off_slice)
    def_rug = rugosity(def_slice)
    combined_rug = off_rug + def_rug

    off_level = float(np.mean(off_slice))
    def_level = float(np.mean(def_slice))
    net_rating = off_level - def_level

    # Trend score (slope of net rating)
    x = list(range(len(off_slice)))
    net_series = [o - d for o, d in zip(off_slice, def_slice)]
    slope, _, _, _ = linear_regression(x, net_series)

    # Combined metrics
    adjusted_rating = net_rating / combined_rug if combined_rug > 0 else net_rating
    weighted_rating = net_rating - (combined_rug * 0.5)

    return {
        "rugosity": round(combined_rug, 4),
        "offRugosity": round(off_rug, 4),
        "defRugosity": round(def_rug, 4),
        "netRating": round(net_rating, 2),
        "trendScore": round(slope, 4),
        "offLevel": round(off_level, 2),
        "defLevel": round(def_level, 2),
        "adjustedRating": round(adjusted_rating, 4),
        "weightedRating": round(weighted_rating, 4),
        "games": len(off_slice)
    }

# ----------------------------
# Fetch Playoff Results
# ----------------------------
def fetch_playoff_results(season, team_list):
    """Fetch playoff wins for each team in a season."""
    playoff_wins = {}

    for t in tqdm(team_list, desc="  Playoffs", unit="team", leave=False):
        team_name = t["full_name"]
        team_id = t["id"]

        try:
            resp = TeamGameLogs(
                league_id_nullable="00",
                team_id_nullable=str(team_id),
                season_nullable=season,
                season_type_nullable="Playoffs"
            )
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

            df = resp.get_data_frames()[0]

            if df.empty:
                playoff_wins[team_name] = 0
            else:
                wl_col = pick_col(df, ["WL", "W_L"])
                wins = len(df[df[wl_col] == "W"])
                playoff_wins[team_name] = wins

        except Exception as e:
            playoff_wins[team_name] = 0

    return playoff_wins

# ----------------------------
# Fetch Regular Season Data
# ----------------------------
def fetch_season_data(season, team_list):
    """Fetch regular season data for all teams in a season."""
    teams_data = []

    for t in tqdm(team_list, desc="  Regular", unit="team", leave=False):
        team_name = t["full_name"]
        team_id = t["id"]

        try:
            resp = TeamGameLogs(
                league_id_nullable="00",
                team_id_nullable=str(team_id),
                season_nullable=season,
                season_type_nullable="Regular Season",
                measure_type_player_game_logs_nullable="Advanced"
            )
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

            df = resp.get_data_frames()[0]

            if df.empty:
                continue

            # Sort by date
            date_col = pick_col(df, ["GAME_DATE"])
            df = df.sort_values(date_col).reset_index(drop=True)

            # Get ratings
            off_col = pick_col(df, ["OFF_RATING", "OFF_RTG", "OFFRTG"])
            def_col = pick_col(df, ["DEF_RATING", "DEF_RTG", "DEFRTG"])

            off_ratings = df[off_col].tolist()
            def_ratings = df[def_col].tolist()
            games = len(off_ratings)

            if games < 10:  # Need at least 10 games for meaningful analysis
                continue

            # Calculate metrics for each window
            windows_data = {}
            for window in WINDOWS:
                metrics = calculate_windowed_metrics(off_ratings, def_ratings, window)
                if metrics:
                    windows_data[str(window)] = metrics

            teams_data.append({
                "name": team_name,
                "regularSeasonGames": games,
                "windows": windows_data
            })

        except Exception as e:
            tqdm.write(f"  Error fetching {team_name}: {e}")

    return teams_data

# ----------------------------
# Calculate Correlations
# ----------------------------
def calculate_correlations(all_seasons_data):
    """Calculate correlations between metrics and playoff wins for each window."""
    correlations = {}

    for window in WINDOWS:
        window_key = str(window)
        rugosities = []
        net_ratings = []
        trend_scores = []
        playoff_wins = []

        # For playoff teams only analysis
        playoff_rugosities = []
        playoff_net_ratings = []
        playoff_wins_only = []

        for season_data in all_seasons_data:
            for team in season_data["teams"]:
                if window_key in team["windows"]:
                    rug = team["windows"][window_key]["rugosity"]
                    net = team["windows"][window_key]["netRating"]
                    trend = team["windows"][window_key]["trendScore"]
                    wins = team["playoffWins"]

                    rugosities.append(rug)
                    net_ratings.append(net)
                    trend_scores.append(trend)
                    playoff_wins.append(wins)

                    # Track playoff teams separately
                    if wins > 0:
                        playoff_rugosities.append(rug)
                        playoff_net_ratings.append(net)
                        playoff_wins_only.append(wins)

        if len(rugosities) < 10:
            continue

        # Calculate basic correlations
        rug_corr, rug_p = pearsonr(rugosities, playoff_wins)
        net_corr, net_p = pearsonr(net_ratings, playoff_wins)
        trend_corr, trend_p = pearsonr(trend_scores, playoff_wins)

        # Combined metric: netRating adjusted by consistency
        # Higher net rating is better, lower rugosity is better
        # adjustedRating = netRating / rugosity (consistency-adjusted performance)
        adjusted_ratings = [net / rug if rug > 0 else net for net, rug in zip(net_ratings, rugosities)]
        adj_corr, adj_p = pearsonr(adjusted_ratings, playoff_wins)

        # Alternative: netRating - rugosity * weight (penalize inconsistency)
        # Use rugosity as a penalty (scaled to be comparable to net rating)
        weighted_ratings = [net - (rug * 0.5) for net, rug in zip(net_ratings, rugosities)]
        weighted_corr, weighted_p = pearsonr(weighted_ratings, playoff_wins)

        # Among playoff teams only: does consistency predict deeper runs?
        playoff_rug_corr, playoff_rug_p = (0, 1)
        playoff_net_corr, playoff_net_p = (0, 1)
        if len(playoff_rugosities) >= 10:
            playoff_rug_corr, playoff_rug_p = pearsonr(playoff_rugosities, playoff_wins_only)
            playoff_net_corr, playoff_net_p = pearsonr(playoff_net_ratings, playoff_wins_only)

        correlations[window_key] = {
            "rugosity_vs_playoffWins": {"r": round(rug_corr, 4), "p": round(rug_p, 6)},
            "netRating_vs_playoffWins": {"r": round(net_corr, 4), "p": round(net_p, 6)},
            "trendScore_vs_playoffWins": {"r": round(trend_corr, 4), "p": round(trend_p, 6)},
            "adjustedRating_vs_playoffWins": {"r": round(adj_corr, 4), "p": round(adj_p, 6)},
            "weightedRating_vs_playoffWins": {"r": round(weighted_corr, 4), "p": round(weighted_p, 6)},
            "playoffTeamsOnly": {
                "rugosity_vs_wins": {"r": round(playoff_rug_corr, 4), "p": round(playoff_rug_p, 6)},
                "netRating_vs_wins": {"r": round(playoff_net_corr, 4), "p": round(playoff_net_p, 6)},
                "sampleSize": len(playoff_rugosities)
            },
            "sampleSize": len(rugosities)
        }

    return correlations

# ----------------------------
# Main
# ----------------------------
def main():
    print("=" * 60)
    print("Historical NBA Trends Analysis")
    print(f"Analyzing seasons: {SEASONS[0]} to {SEASONS[-1]}")
    print("=" * 60)

    team_list = nba_teams.get_teams()
    all_seasons_data = []

    for season in tqdm(SEASONS, desc="Seasons", unit="season", position=0):
        tqdm.write(f"\n{season}")

        # Fetch regular season data
        teams_data = fetch_season_data(season, team_list)

        # Fetch playoff results
        playoff_results = fetch_playoff_results(season, team_list)

        # Merge playoff wins
        for team in teams_data:
            team["playoffWins"] = playoff_results.get(team["name"], 0)
            team["madePlayoffs"] = bool(team["playoffWins"] > 0)

        # Find champion (team with most wins, should be 16)
        champion = max(teams_data, key=lambda t: t["playoffWins"], default=None)
        if champion:
            tqdm.write(f"  Champion: {champion['name']} ({champion['playoffWins']} wins)")

        all_seasons_data.append({
            "season": season,
            "teams": teams_data
        })

    # Calculate correlations
    print("\n" + "=" * 40)
    print("Calculating correlations...")
    correlations = calculate_correlations(all_seasons_data)

    # Print correlation summary
    print("\nCorrelation Summary (with playoff wins):")
    print("-" * 70)
    print(f"  {'Window':>8}  {'Rugosity':>10}  {'NetRating':>10}  {'Adjusted':>10}  {'Weighted':>10}")
    print("-" * 70)
    for window in WINDOWS:
        w = str(window)
        if w in correlations:
            c = correlations[w]
            print(f"  {window:>8}  "
                  f"{c['rugosity_vs_playoffWins']['r']:+.3f}      "
                  f"{c['netRating_vs_playoffWins']['r']:+.3f}      "
                  f"{c['adjustedRating_vs_playoffWins']['r']:+.3f}      "
                  f"{c['weightedRating_vs_playoffWins']['r']:+.3f}")

    # Print playoff teams only analysis
    print("\nAmong Playoff Teams Only (predicting deeper runs):")
    print("-" * 50)
    for window in WINDOWS:
        w = str(window)
        if w in correlations:
            c = correlations[w]["playoffTeamsOnly"]
            print(f"  {window:>8}: Rugosity r={c['rugosity_vs_wins']['r']:+.3f}, "
                  f"NetRating r={c['netRating_vs_wins']['r']:+.3f} (n={c['sampleSize']})")

    # Find best window
    best_window = max(correlations.keys(),
                      key=lambda w: abs(correlations[w]["netRating_vs_playoffWins"]["r"]))

    # Summary statistics
    total_team_seasons = sum(len(s["teams"]) for s in all_seasons_data)
    playoff_teams_total = sum(1 for s in all_seasons_data for t in s["teams"] if t["madePlayoffs"])
    champions = [
        {"season": s["season"], "name": max(s["teams"], key=lambda t: t["playoffWins"])["name"]}
        for s in all_seasons_data if s["teams"]
    ]

    # Determine if rugosity predicts better
    full_corr = correlations.get("full", correlations.get("40", {}))
    rugosity_better = bool(abs(full_corr.get("rugosity_vs_playoffWins", {}).get("r", 0)) > \
                      abs(full_corr.get("netRating_vs_playoffWins", {}).get("r", 0)))

    # Build output
    output = {
        "generated": pd.Timestamp.now().isoformat(),
        "seasonsAnalyzed": SEASONS,
        "windows": [str(w) for w in WINDOWS],
        "seasons": all_seasons_data,
        "correlations": correlations,
        "summary": {
            "totalTeamSeasons": total_team_seasons,
            "playoffTeams": playoff_teams_total,
            "champions": champions,
            "bestCorrelationWindow": best_window,
            "rugosityPredictsBetter": rugosity_better
        }
    }

    # Save to file
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"  Total team-seasons: {total_team_seasons}")
    print(f"  Playoff appearances: {playoff_teams_total}")
    print(f"  Best prediction window: Last {best_window} games")
    print(f"  Rugosity predicts better than NetRating: {rugosity_better}")
    print("=" * 60)

if __name__ == "__main__":
    main()
