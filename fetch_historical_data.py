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

    # Rugosity (arc-to-chord ratio)
    off_rug = rugosity(off_slice)
    def_rug = rugosity(def_slice)
    combined_rug = off_rug + def_rug

    # Level (mean rating)
    off_level = float(np.mean(off_slice))
    def_level = float(np.mean(def_slice))
    net_rating = off_level - def_level

    # Standard deviation (alternative consistency metric)
    off_std = float(np.std(off_slice))
    def_std = float(np.std(def_slice))
    combined_std = off_std + def_std

    # Individual slopes (trends)
    x = list(range(len(off_slice)))
    off_slope, _, off_resid_std, _ = linear_regression(x, off_slice)
    def_slope, _, def_resid_std, _ = linear_regression(x, def_slice)

    # Net rating trend
    net_series = [o - d for o, d in zip(off_slice, def_slice)]
    net_slope, _, net_resid_std, _ = linear_regression(x, net_series)

    # Combined residual std (how far from trend line - alternative consistency)
    combined_resid_std = off_resid_std + def_resid_std

    # Combined metrics
    adjusted_rating = net_rating / combined_rug if combined_rug > 0 else net_rating
    weighted_rating = net_rating - (combined_rug * 0.5)

    # Std-based alternatives
    adjusted_by_std = net_rating / combined_std if combined_std > 0 else net_rating
    weighted_by_std = net_rating - (combined_std * 0.5)

    return {
        # Core metrics
        "netRating": round(net_rating, 2),
        "offLevel": round(off_level, 2),
        "defLevel": round(def_level, 2),

        # Rugosity (consistency)
        "rugosity": round(combined_rug, 4),
        "offRugosity": round(off_rug, 4),
        "defRugosity": round(def_rug, 4),

        # Standard deviation (alternative consistency)
        "stdDev": round(combined_std, 4),
        "offStdDev": round(off_std, 4),
        "defStdDev": round(def_std, 4),

        # Residual std (how well trend fits)
        "residualStd": round(combined_resid_std, 4),
        "offResidualStd": round(off_resid_std, 4),
        "defResidualStd": round(def_resid_std, 4),

        # Trends/slopes
        "trendScore": round(net_slope, 4),
        "offSlope": round(off_slope, 4),
        "defSlope": round(def_slope, 4),

        # Combined metrics (rugosity-based)
        "adjustedRating": round(adjusted_rating, 4),
        "weightedRating": round(weighted_rating, 4),

        # Combined metrics (std-based)
        "adjustedByStd": round(adjusted_by_std, 4),
        "weightedByStd": round(weighted_by_std, 4),

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
# Archetype Labels (from main page)
# ----------------------------
MATRIX_LABELS = {
    "1111": "Elite Machine",
    "1110": "Elite but Shaky D",
    "1101": "Streaky Juggernaut",
    "1100": "Talented Chaos",
    "1011": "Offensive Powerhouse",
    "1010": "Offensive but Shaky",
    "1001": "Streaky Offense",
    "1000": "Offensive Chaos",
    "0111": "Defensive Fortress",
    "0110": "Defense but Shaky",
    "0101": "Streaky Defense",
    "0100": "Defensive Chaos",
    "0011": "Consistently Mediocre",
    "0010": "Consistently Inconsistent",
    "0001": "Struggling but Steady D",
    "0000": "Full Rebuild Mode",
}

# ----------------------------
# Calculate Archetypes and Badges for a Season
# ----------------------------
def calculate_archetypes_and_badges(teams_data, window_key="full"):
    """Calculate archetypes based on league medians and identify badge winners."""
    if not teams_data:
        return {}, {}

    # Extract metrics for calculating medians
    metrics = []
    for team in teams_data:
        if window_key in team["windows"]:
            w = team["windows"][window_key]
            metrics.append({
                "name": team["name"],
                "offLevel": w["offLevel"],
                "defLevel": w["defLevel"],
                "offRugosity": w["offRugosity"],
                "defRugosity": w["defRugosity"],
                "rugosity": w["rugosity"],
                "netRating": w["netRating"],
                "trendScore": w["trendScore"],
                "offSlope": w["offSlope"],
                "defSlope": w["defSlope"],
            })

    if len(metrics) < 10:
        return {}, {}

    # Calculate league medians
    off_level_median = np.median([m["offLevel"] for m in metrics])
    def_level_median = np.median([m["defLevel"] for m in metrics])
    off_rug_median = np.median([m["offRugosity"] for m in metrics])
    def_rug_median = np.median([m["defRugosity"] for m in metrics])

    # Assign archetypes to each team
    archetypes = {}
    for m in metrics:
        off_high = 1 if m["offLevel"] >= off_level_median else 0
        def_high = 1 if m["defLevel"] <= def_level_median else 0  # Lower is better
        off_consistent = 1 if m["offRugosity"] <= off_rug_median else 0
        def_consistent = 1 if m["defRugosity"] <= def_rug_median else 0

        matrix_code = f"{off_high}{def_high}{off_consistent}{def_consistent}"
        archetypes[m["name"]] = {
            "code": matrix_code,
            "label": MATRIX_LABELS.get(matrix_code, "Unclassified"),
            "offHigh": bool(off_high),
            "defHigh": bool(def_high),
            "offConsistent": bool(off_consistent),
            "defConsistent": bool(def_consistent),
        }

    # Calculate badge winners
    badges = {}

    # Sort by net rating for top/bottom 10
    sorted_by_net = sorted(metrics, key=lambda t: t["netRating"], reverse=True)
    top_10 = sorted_by_net[:10]
    bottom_10 = sorted_by_net[-10:]

    # Consistently Good: Most consistent among top 10 net rating
    consistently_good = min(top_10, key=lambda t: t["rugosity"])
    badges["consistently_good"] = consistently_good["name"]

    # Consistently Bad: Most consistent among bottom 10 net rating
    consistently_bad = min(bottom_10, key=lambda t: t["rugosity"])
    badges["consistently_bad"] = consistently_bad["name"]

    # Best Offense: Highest off rating adjusted for consistency
    best_offense = max(metrics, key=lambda t: t["offLevel"] - (t["offRugosity"] * 0.5))
    badges["best_offense"] = best_offense["name"]

    # Best Defense: Lowest def rating adjusted for consistency
    best_defense = min(metrics, key=lambda t: t["defLevel"] + (t["defRugosity"] * 0.5))
    badges["best_defense"] = best_defense["name"]

    # On the Rise: Highest trend score (offSlope - defSlope)
    for m in metrics:
        m["recentTrendScore"] = m["offSlope"] - m["defSlope"]
    on_the_rise = max(metrics, key=lambda t: t["recentTrendScore"])
    badges["on_the_rise"] = on_the_rise["name"]

    # On the Decline: Lowest trend score
    on_the_decline = min(metrics, key=lambda t: t["recentTrendScore"])
    badges["on_the_decline"] = on_the_decline["name"]

    return archetypes, badges

# ----------------------------
# Calculate Correlations
# ----------------------------
def calculate_correlations(all_seasons_data):
    """Calculate correlations between ALL metrics and playoff wins for each window."""
    correlations = {}

    for window in WINDOWS:
        window_key = str(window)

        # Collect all metrics
        data = {
            "netRating": [], "offLevel": [], "defLevel": [],
            "rugosity": [], "offRugosity": [], "defRugosity": [],
            "stdDev": [], "offStdDev": [], "defStdDev": [],
            "residualStd": [], "offResidualStd": [], "defResidualStd": [],
            "trendScore": [], "offSlope": [], "defSlope": [],
            "adjustedRating": [], "weightedRating": [],
            "adjustedByStd": [], "weightedByStd": [],
            "playoffWins": [],
            "archetype": [], "archetypeCode": [],
        }

        # Playoff teams only
        playoff_data = {k: [] for k in data.keys()}

        for season_data in all_seasons_data:
            for team in season_data["teams"]:
                if window_key not in team["windows"]:
                    continue

                w = team["windows"][window_key]
                wins = team["playoffWins"]

                # Core metrics
                data["netRating"].append(w["netRating"])
                data["offLevel"].append(w["offLevel"])
                data["defLevel"].append(w["defLevel"])

                # Consistency metrics
                data["rugosity"].append(w["rugosity"])
                data["offRugosity"].append(w["offRugosity"])
                data["defRugosity"].append(w["defRugosity"])
                data["stdDev"].append(w["stdDev"])
                data["offStdDev"].append(w["offStdDev"])
                data["defStdDev"].append(w["defStdDev"])
                data["residualStd"].append(w["residualStd"])
                data["offResidualStd"].append(w["offResidualStd"])
                data["defResidualStd"].append(w["defResidualStd"])

                # Trend metrics
                data["trendScore"].append(w["trendScore"])
                data["offSlope"].append(w["offSlope"])
                data["defSlope"].append(w["defSlope"])

                # Combined metrics
                data["adjustedRating"].append(w["adjustedRating"])
                data["weightedRating"].append(w["weightedRating"])
                data["adjustedByStd"].append(w["adjustedByStd"])
                data["weightedByStd"].append(w["weightedByStd"])

                data["playoffWins"].append(wins)

                # Archetype (if available)
                if "archetype" in team:
                    data["archetype"].append(team["archetype"]["label"])
                    data["archetypeCode"].append(team["archetype"]["code"])

                # Playoff teams only
                if wins > 0:
                    for key in data.keys():
                        if key != "playoffWins" and len(data[key]) > 0:
                            playoff_data[key].append(data[key][-1])
                    playoff_data["playoffWins"].append(wins)

        if len(data["netRating"]) < 10:
            continue

        # Helper to calculate correlation safely
        def safe_corr(x, y):
            if len(x) < 10 or len(set(x)) < 2:
                return 0.0, 1.0
            try:
                r, p = pearsonr(x, y)
                return round(r, 4), round(p, 6)
            except:
                return 0.0, 1.0

        wins = data["playoffWins"]
        playoff_wins = playoff_data["playoffWins"]

        # Calculate all correlations
        corr_results = {}

        # Core metrics
        for metric in ["netRating", "offLevel", "defLevel"]:
            r, p = safe_corr(data[metric], wins)
            corr_results[f"{metric}_vs_playoffWins"] = {"r": r, "p": p}

        # Consistency metrics (rugosity, std, residual)
        for metric in ["rugosity", "offRugosity", "defRugosity",
                       "stdDev", "offStdDev", "defStdDev",
                       "residualStd", "offResidualStd", "defResidualStd"]:
            r, p = safe_corr(data[metric], wins)
            corr_results[f"{metric}_vs_playoffWins"] = {"r": r, "p": p}

        # Trend metrics
        for metric in ["trendScore", "offSlope", "defSlope"]:
            r, p = safe_corr(data[metric], wins)
            corr_results[f"{metric}_vs_playoffWins"] = {"r": r, "p": p}

        # Combined metrics
        for metric in ["adjustedRating", "weightedRating", "adjustedByStd", "weightedByStd"]:
            r, p = safe_corr(data[metric], wins)
            corr_results[f"{metric}_vs_playoffWins"] = {"r": r, "p": p}

        # Playoff teams only
        playoff_results = {}
        if len(playoff_wins) >= 10:
            for metric in ["netRating", "rugosity", "stdDev", "residualStd", "trendScore",
                           "offLevel", "defLevel", "offRugosity", "defRugosity"]:
                r, p = safe_corr(playoff_data[metric], playoff_wins)
                playoff_results[f"{metric}_vs_wins"] = {"r": r, "p": p}
            playoff_results["sampleSize"] = len(playoff_wins)
        else:
            playoff_results["sampleSize"] = len(playoff_wins)

        corr_results["playoffTeamsOnly"] = playoff_results
        corr_results["sampleSize"] = len(wins)

        correlations[window_key] = corr_results

    return correlations


# ----------------------------
# Analyze Badge Winners
# ----------------------------
def analyze_badge_performance(all_seasons_data):
    """Analyze how badge winners performed in playoffs."""
    badge_stats = {
        "consistently_good": {"wins": [], "made_playoffs": 0, "total": 0},
        "consistently_bad": {"wins": [], "made_playoffs": 0, "total": 0},
        "best_offense": {"wins": [], "made_playoffs": 0, "total": 0},
        "best_defense": {"wins": [], "made_playoffs": 0, "total": 0},
        "on_the_rise": {"wins": [], "made_playoffs": 0, "total": 0},
        "on_the_decline": {"wins": [], "made_playoffs": 0, "total": 0},
    }

    for season_data in all_seasons_data:
        badges = season_data.get("badges", {})
        teams_by_name = {t["name"]: t for t in season_data["teams"]}

        for badge_name, team_name in badges.items():
            if badge_name in badge_stats and team_name in teams_by_name:
                team = teams_by_name[team_name]
                wins = team["playoffWins"]
                badge_stats[badge_name]["wins"].append(wins)
                badge_stats[badge_name]["total"] += 1
                if wins > 0:
                    badge_stats[badge_name]["made_playoffs"] += 1

    # Calculate summary stats
    results = {}
    for badge, stats in badge_stats.items():
        if stats["total"] > 0:
            results[badge] = {
                "avgPlayoffWins": round(np.mean(stats["wins"]), 2) if stats["wins"] else 0,
                "playoffRate": round(stats["made_playoffs"] / stats["total"], 3),
                "totalSeasons": stats["total"],
                "madePlayoffs": stats["made_playoffs"],
            }

    return results


# ----------------------------
# Analyze Archetypes
# ----------------------------
def analyze_archetype_performance(all_seasons_data):
    """Analyze how each archetype performed in playoffs."""
    archetype_stats = {}

    for season_data in all_seasons_data:
        for team in season_data["teams"]:
            if "archetype" not in team:
                continue

            label = team["archetype"]["label"]
            code = team["archetype"]["code"]
            wins = team["playoffWins"]

            if label not in archetype_stats:
                archetype_stats[label] = {
                    "code": code,
                    "wins": [],
                    "made_playoffs": 0,
                    "total": 0,
                    "net_ratings": []
                }

            archetype_stats[label]["wins"].append(wins)
            archetype_stats[label]["total"] += 1
            if wins > 0:
                archetype_stats[label]["made_playoffs"] += 1
            if "full" in team["windows"]:
                archetype_stats[label]["net_ratings"].append(team["windows"]["full"]["netRating"])

    # Calculate summary stats
    results = {}
    for archetype, stats in archetype_stats.items():
        if stats["total"] >= 5:  # Need at least 5 samples
            results[archetype] = {
                "code": stats["code"],
                "avgPlayoffWins": round(np.mean(stats["wins"]), 2),
                "playoffRate": round(stats["made_playoffs"] / stats["total"], 3),
                "totalTeamSeasons": stats["total"],
                "madePlayoffs": stats["made_playoffs"],
                "avgNetRating": round(np.mean(stats["net_ratings"]), 2) if stats["net_ratings"] else 0,
            }

    # Sort by avg playoff wins
    results = dict(sorted(results.items(), key=lambda x: x[1]["avgPlayoffWins"], reverse=True))

    return results

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

        # Calculate archetypes and badges
        archetypes, badges = calculate_archetypes_and_badges(teams_data, window_key="full")

        # Assign archetypes to teams
        for team in teams_data:
            if team["name"] in archetypes:
                team["archetype"] = archetypes[team["name"]]

        # Find champion (team with most wins, should be 16)
        champion = max(teams_data, key=lambda t: t["playoffWins"], default=None)
        if champion:
            tqdm.write(f"  Champion: {champion['name']} ({champion['playoffWins']} wins)")

        all_seasons_data.append({
            "season": season,
            "teams": teams_data,
            "badges": badges
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
        if w in correlations and "rugosity_vs_wins" in correlations[w].get("playoffTeamsOnly", {}):
            c = correlations[w]["playoffTeamsOnly"]
            print(f"  {window:>8}: Rugosity r={c['rugosity_vs_wins']['r']:+.3f}, "
                  f"NetRating r={c['netRating_vs_wins']['r']:+.3f} (n={c['sampleSize']})")

    # Analyze badge winners
    print("\n" + "=" * 40)
    print("Badge Winner Performance:")
    print("-" * 50)
    badge_analysis = analyze_badge_performance(all_seasons_data)
    for badge, stats in badge_analysis.items():
        print(f"  {badge:20}: Avg {stats['avgPlayoffWins']:.1f} wins, "
              f"{stats['playoffRate']*100:.0f}% made playoffs ({stats['madePlayoffs']}/{stats['totalSeasons']})")

    # Analyze archetypes
    print("\n" + "=" * 40)
    print("Archetype Performance (sorted by avg playoff wins):")
    print("-" * 60)
    archetype_analysis = analyze_archetype_performance(all_seasons_data)
    for archetype, stats in list(archetype_analysis.items())[:8]:  # Top 8
        print(f"  {archetype:25}: Avg {stats['avgPlayoffWins']:.1f} wins, "
              f"{stats['playoffRate']*100:.0f}% playoffs (n={stats['totalTeamSeasons']})")

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
        "badgeAnalysis": badge_analysis,
        "archetypeAnalysis": archetype_analysis,
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
