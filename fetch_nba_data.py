# pip install nba_api pandas tqdm

import json
import time
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import TeamGameLogs
from nba_api.stats.static import teams as nba_teams
from tqdm.auto import tqdm

# ----------------------------
# PARAMETERS
# ----------------------------
SEASON = "2025-26"
WINDOW = 10
SEASON_TYPE = "Regular Season"
SLEEP_BETWEEN_CALLS_SEC = 0.65

OUTPUT_FILE = "nba_trends_data.json"

# ----------------------------
# Helpers
# ----------------------------
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")

def turning_points(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return 0
    d = np.diff(y)
    d = np.where(d == 0, 1e-12, d)
    return int(np.sum((d[:-1] * d[1:]) < 0))

def rugosity(y):
    """Calculate rugosity (arc-to-chord ratio) for a time series."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return 1.0
    
    # Path length: sum of distances between consecutive points
    # Using normalized x (each step = 1)
    path_length = 0.0
    for i in range(len(y) - 1):
        dx = 1.0  # normalized step
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
    m, b = np.polyfit(x, y, 1)
    y_hat = m * x + b
    resid_std = float(np.std(y - y_hat))
    return float(m), float(b), resid_std, y_hat.tolist()

def rolling_mean(arr, window):
    s = pd.Series(arr)
    return s.rolling(window, center=True, min_periods=1).mean().tolist()

def zscore(values):
    arr = np.array(values)
    std = arr.std()
    if std == 0:
        return [0.0] * len(values)
    return ((arr - arr.mean()) / std).tolist()

# ----------------------------
# Fetch all teams
# ----------------------------
print(f"Fetching NBA data for {SEASON} season...\n")

team_list = nba_teams.get_teams()
teams_data = []

pbar = tqdm(team_list, desc="Fetching teams", unit="team")

for t in pbar:
    team_name = t["full_name"]
    team_id = t["id"]
    pbar.set_postfix(team=team_name)

    try:
        resp = TeamGameLogs(
            league_id_nullable="00",
            team_id_nullable=str(team_id),
            season_nullable=SEASON,
            season_type_nullable=SEASON_TYPE,
            measure_type_player_game_logs_nullable="Advanced",
        )
        df = resp.get_data_frames()[0].copy()
        
        if df.empty:
            pbar.write(f"[SKIP] {team_name}: no games")
            continue

        off_col = pick_col(df, ["OFF_RATING", "OFF_RTG"])
        def_col = pick_col(df, ["DEF_RATING", "DEF_RTG"])

        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE").reset_index(drop=True)
        
        games = len(df)
        if games < WINDOW:
            pbar.write(f"[SKIP] {team_name}: only {games} games (need {WINDOW})")
            continue

        game_nums = list(range(1, games + 1))
        off_ratings = df[off_col].tolist()
        def_ratings = df[def_col].tolist()
        
        off_smooth = rolling_mean(off_ratings, WINDOW)
        def_smooth = rolling_mean(def_ratings, WINDOW)

        off_slope, off_b, off_resid_std, off_trend = linear_regression(game_nums, off_smooth)
        def_slope, def_b, def_resid_std, def_trend = linear_regression(game_nums, def_smooth)

        off_turns = turning_points(off_smooth)
        def_turns = turning_points(def_smooth)
        off_turns_norm = off_turns / max(1, len(off_smooth) - 2)
        def_turns_norm = def_turns / max(1, len(def_smooth) - 2)
        
        # Calculate rugosity for raw ratings (better measure of game-to-game volatility)
        off_rugosity = rugosity(off_ratings)
        def_rugosity = rugosity(def_ratings)

        teams_data.append({
            "name": team_name,
            "games": games,
            "gameNums": game_nums,
            "offRaw": [round(x, 2) for x in off_ratings],
            "defRaw": [round(x, 2) for x in def_ratings],
            "offSmooth": [round(x, 2) for x in off_smooth],
            "defSmooth": [round(x, 2) for x in def_smooth],
            "offTrend": [round(x, 2) for x in off_trend],
            "defTrend": [round(x, 2) for x in def_trend],
            "offLevel": round(float(np.mean(off_smooth)), 2),
            "defLevel": round(float(np.mean(def_smooth)), 2),
            "offSlope": round(off_slope, 4),
            "defSlope": round(def_slope, 4),
            "offResidStd": round(off_resid_std, 4),
            "defResidStd": round(def_resid_std, 4),
            "offTurnsNorm": round(off_turns_norm, 4),
            "defTurnsNorm": round(def_turns_norm, 4),
            "offRugosity": round(off_rugosity, 4),
            "defRugosity": round(def_rugosity, 4),
        })

    except Exception as e:
        pbar.write(f"[ERROR] {team_name}: {e}")

    time.sleep(SLEEP_BETWEEN_CALLS_SEC)

# ----------------------------
# Compute z-scores and metrics
# ----------------------------
print(f"\nProcessed {len(teams_data)} teams. Computing metrics...")

if len(teams_data) > 0:
    # Extract arrays for z-score calculation
    off_resid_stds = [t["offResidStd"] for t in teams_data]
    def_resid_stds = [t["defResidStd"] for t in teams_data]
    off_turns_norms = [t["offTurnsNorm"] for t in teams_data]
    def_turns_norms = [t["defTurnsNorm"] for t in teams_data]
    off_slopes = [t["offSlope"] for t in teams_data]
    def_slopes = [-t["defSlope"] for t in teams_data]  # Negative because lower def is better

    z_off_resid = zscore(off_resid_stds)
    z_def_resid = zscore(def_resid_stds)
    z_off_turns = zscore(off_turns_norms)
    z_def_turns = zscore(def_turns_norms)
    z_off_slope = zscore(off_slopes)
    z_def_slope = zscore(def_slopes)
    
    # Calculate percentiles for classification
    off_levels = [t["offLevel"] for t in teams_data]
    def_levels = [t["defLevel"] for t in teams_data]
    
    off_q25 = np.percentile(off_levels, 25)
    off_q75 = np.percentile(off_levels, 75)
    def_q25 = np.percentile(def_levels, 25)  # Lower is better for defense
    def_q75 = np.percentile(def_levels, 75)

    for i, team in enumerate(teams_data):
        team["offInconsistency"] = round(z_off_resid[i] + z_off_turns[i], 4)
        team["defInconsistency"] = round(z_def_resid[i] + z_def_turns[i], 4)
        team["overallInconsistency"] = round(z_off_resid[i] + z_off_turns[i] + z_def_resid[i] + z_def_turns[i], 4)
        team["overallImprovement"] = round(z_off_slope[i] + z_def_slope[i], 4)
        
        # Classify performance tiers
        off_level = team["offLevel"]
        def_level = team["defLevel"]
        off_rug = team["offRugosity"]
        def_rug = team["defRugosity"]
        off_slope = team["offSlope"]
        def_slope = team["defSlope"]
        
        # Performance classification
        off_tier = "elite" if off_level >= off_q75 else ("poor" if off_level <= off_q25 else "average")
        def_tier = "elite" if def_level <= def_q25 else ("poor" if def_level >= def_q75 else "average")
        
        # Rugosity classification (thresholds based on typical NBA variance)
        off_rug_tier = "consistent" if off_rug < 1.15 else ("volatile" if off_rug > 1.35 else "moderate")
        def_rug_tier = "consistent" if def_rug < 1.15 else ("volatile" if def_rug > 1.35 else "moderate")
        
        # Trend classification (per 10 games)
        off_trend_tier = "improving" if off_slope > 0.075 else ("declining" if off_slope < -0.075 else "stable")
        def_trend_tier = "improving" if def_slope < -0.075 else ("declining" if def_slope > 0.075 else "stable")
        
        team["offTier"] = off_tier
        team["defTier"] = def_tier
        team["offRugTier"] = off_rug_tier
        team["defRugTier"] = def_rug_tier
        team["offTrendTier"] = off_trend_tier
        team["defTrendTier"] = def_trend_tier
        
        # Assign archetype (priority order)
        archetype = "Unclassified"
        
        # 1. The Machine - Elite both ends, both consistent
        if off_tier == "elite" and def_tier == "elite" and off_rug_tier == "consistent" and def_rug_tier == "consistent":
            archetype = "The Machine"
        # 2. Streaky Contender - Elite both ends, either volatile
        elif off_tier == "elite" and def_tier == "elite" and (off_rug_tier == "volatile" or def_rug_tier == "volatile"):
            archetype = "Streaky Contender"
        # 3. Fireworks Show - Elite offense, poor defense, volatile offense
        elif off_tier == "elite" and def_tier == "poor" and off_rug_tier == "volatile":
            archetype = "Fireworks Show"
        # 4. Feast or Famine - Elite offense, volatile offense, consistent defense
        elif off_tier == "elite" and off_rug_tier == "volatile" and def_rug_tier == "consistent":
            archetype = "Feast or Famine"
        # 5. Defensive Shell - Poor offense, elite defense, both consistent
        elif off_tier == "poor" and def_tier == "elite" and off_rug_tier == "consistent" and def_rug_tier == "consistent":
            archetype = "Defensive Shell"
        # 6. Defensive Rollercoaster - Elite defense, volatile defense
        elif def_tier == "elite" and def_rug_tier == "volatile":
            archetype = "Defensive Rollercoaster"
        # 7. Defensive Fortress - Elite defense, average offense, consistent
        elif def_tier == "elite" and off_tier == "average" and off_rug_tier == "consistent":
            archetype = "Defensive Fortress"
        # 8. Steady Contender - Elite offense, average defense, consistent
        elif off_tier == "elite" and def_tier == "average" and off_rug_tier == "consistent":
            archetype = "Steady Contender"
        # 9. Jekyll & Hyde - Both volatile
        elif off_rug_tier == "volatile" and def_rug_tier == "volatile":
            archetype = "Jekyll & Hyde"
        # 10. Rising Tide - Both improving
        elif off_trend_tier == "improving" and def_trend_tier == "improving":
            archetype = "Rising Tide"
        # 11. Falling Apart - Both declining
        elif off_trend_tier == "declining" and def_trend_tier == "declining":
            archetype = "Falling Apart"
        # 12. One-Way Surge - Offense improving, defense stable
        elif off_trend_tier == "improving" and def_trend_tier == "stable":
            archetype = "One-Way Surge"
        # 13. Defensive Awakening - Defense improving, offense stable
        elif def_trend_tier == "improving" and off_trend_tier == "stable":
            archetype = "Defensive Awakening"
        # Fallback based on consistency
        elif off_rug_tier == "consistent" and def_rug_tier == "consistent":
            archetype = "Steady Mediocrity" if off_tier == "average" and def_tier == "average" else "Steady"
        elif off_rug_tier == "volatile" or def_rug_tier == "volatile":
            archetype = "Volatile Mediocrity" if off_tier == "average" and def_tier == "average" else "Volatile"
        
        team["archetype"] = archetype

# ----------------------------
# Output JSON
# ----------------------------
output = {
    "season": SEASON,
    "generated": pd.Timestamp.now().isoformat(),
    "window": WINDOW,
    "teams": teams_data
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Saved to {OUTPUT_FILE}")
print(f"  {len(teams_data)} teams, {sum(t['games'] for t in teams_data)} total games")
