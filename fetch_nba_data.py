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
WINDOW = 7
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

        # Calculate last 20 games trend (or all games if less than 20)
        last_n = min(20, games)
        recent_off = off_smooth[-last_n:]
        recent_def = def_smooth[-last_n:]
        recent_games = list(range(1, last_n + 1))

        recent_off_slope, _, _, _ = linear_regression(recent_games, recent_off)
        recent_def_slope, _, _, _ = linear_regression(recent_games, recent_def)
        # Combined trend: offense improving (positive) + defense improving (negative slope = good)
        recent_trend_score = recent_off_slope - recent_def_slope

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
            "recentOffSlope": round(recent_off_slope, 4),
            "recentDefSlope": round(recent_def_slope, 4),
            "recentTrendScore": round(recent_trend_score, 4),
        })

    except Exception as e:
        pbar.write(f"[ERROR] {team_name}: {e}")

    time.sleep(SLEEP_BETWEEN_CALLS_SEC)

# ----------------------------
# Compute z-scores and metrics
# ----------------------------
print(f"\nProcessed {len(teams_data)} teams. Computing metrics...")

# Initialize league medians for output
league_medians = {}

if len(teams_data) > 0:
    # Calculate net rating for each team
    for team in teams_data:
        team["netRating"] = round(team["offLevel"] - team["defLevel"], 2)
    
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
    
    # Extract arrays for classification
    off_levels = [t["offLevel"] for t in teams_data]
    def_levels = [t["defLevel"] for t in teams_data]
    net_ratings = [t["netRating"] for t in teams_data]
    off_rugosities = [t["offRugosity"] for t in teams_data]
    def_rugosities = [t["defRugosity"] for t in teams_data]
    off_slope_vals = [t["offSlope"] for t in teams_data]
    def_slope_vals = [t["defSlope"] for t in teams_data]

    # ===========================================
    # 4D MATRIX CLASSIFICATION SYSTEM
    # ===========================================
    # Dimensions: OffRating, DefRating, OffConsistency, DefConsistency
    # Each dimension is binary: above/below median
    # This creates 2^4 = 16 possible matrix cells
    
    # Calculate medians for each dimension
    off_median = np.median(off_levels)
    def_median = np.median(def_levels)
    off_rug_median = np.median(off_rugosities)
    def_rug_median = np.median(def_rugosities)
    off_slope_median = np.median(off_slope_vals)
    def_slope_median = np.median(def_slope_vals)
    
    # Store league medians for visualization
    league_medians = {
        "offLevel": round(float(off_median), 2),
        "defLevel": round(float(def_median), 2),
        "offRugosity": round(float(off_rug_median), 4),
        "defRugosity": round(float(def_rug_median), 4),
        "offSlope": round(float(off_slope_median), 4),
        "defSlope": round(float(def_slope_median), 4)
    }
    
    # Calculate standard deviations for trend significance
    off_slope_std = np.std(off_slope_vals)
    def_slope_std = np.std(def_slope_vals)
    
    # Trend significance thresholds (1.0 std = significant, 1.5 std = overwhelming)
    SIGNIFICANT_THRESHOLD = 1.0
    OVERWHELMING_THRESHOLD = 1.5
    
    # Define tier boundaries
    tier_definitions = [
        ("Contenders", 6.0),
        ("Legit Threats", 3.5),
        ("Dangerous", 1.5),
        ("Play-In", -1.5),
        ("Up-and-Coming", -4.0),
        ("Rebuild", -7.0),
        ("Tank", float('-inf'))
    ]

    for i, team in enumerate(teams_data):
        team["offInconsistency"] = round(z_off_resid[i] + z_off_turns[i], 4)
        team["defInconsistency"] = round(z_def_resid[i] + z_def_turns[i], 4)
        team["overallInconsistency"] = round(z_off_resid[i] + z_off_turns[i] + z_def_resid[i] + z_def_turns[i], 4)
        team["overallImprovement"] = round(z_off_slope[i] + z_def_slope[i], 4)
        
        off_level = team["offLevel"]
        def_level = team["defLevel"]
        off_rug = team["offRugosity"]
        def_rug = team["defRugosity"]
        off_slope = team["offSlope"]
        def_slope = team["defSlope"]
        net_rating = team["netRating"]
        
        # Assign tier based on net rating
        tier = "Tank"
        for tier_name, threshold in tier_definitions:
            if net_rating > threshold:
                tier = tier_name
                break
        team["tier"] = tier
        
        # ===========================================
        # 4D MATRIX POSITION (binary for each dimension)
        # ===========================================
        # Offense: 1 = above median (good), 0 = below median
        # Defense: 1 = below median (good, lower is better), 0 = above median
        # Off Consistency: 1 = below median rugosity (consistent), 0 = above (volatile)
        # Def Consistency: 1 = below median rugosity (consistent), 0 = above (volatile)
        
        off_high = 1 if off_level >= off_median else 0
        def_high = 1 if def_level <= def_median else 0  # Lower defense rating is better
        off_consistent = 1 if off_rug <= off_rug_median else 0  # Lower rugosity = more consistent
        def_consistent = 1 if def_rug <= def_rug_median else 0
        
        # Store matrix position as binary string (e.g., "1101")
        matrix_code = f"{off_high}{def_high}{off_consistent}{def_consistent}"
        team["matrixCode"] = matrix_code
        team["matrixPosition"] = {
            "offHigh": off_high,
            "defHigh": def_high,
            "offConsistent": off_consistent,
            "defConsistent": def_consistent
        }
        
        # ===========================================
        # TREND ANALYSIS (z-scores for significance)
        # ===========================================
        off_slope_z = (off_slope - off_slope_median) / off_slope_std if off_slope_std > 0 else 0
        def_slope_z = -(def_slope - def_slope_median) / def_slope_std if def_slope_std > 0 else 0  # Negative because lower is better
        
        team["offSlopeZ"] = round(off_slope_z, 3)
        team["defSlopeZ"] = round(def_slope_z, 3)
        
        # Determine trend significance
        off_trend_sig = "none"
        if abs(off_slope_z) >= OVERWHELMING_THRESHOLD:
            off_trend_sig = "overwhelming"
        elif abs(off_slope_z) >= SIGNIFICANT_THRESHOLD:
            off_trend_sig = "significant"
        
        def_trend_sig = "none"
        if abs(def_slope_z) >= OVERWHELMING_THRESHOLD:
            def_trend_sig = "overwhelming"
        elif abs(def_slope_z) >= SIGNIFICANT_THRESHOLD:
            def_trend_sig = "significant"
        
        off_improving = off_slope_z > 0
        def_improving = def_slope_z > 0
        
        team["offTrendSig"] = off_trend_sig
        team["defTrendSig"] = def_trend_sig
        team["offImproving"] = bool(off_improving)
        team["defImproving"] = bool(def_improving)
        
        # ===========================================
        # DYNAMIC LABEL GENERATION
        # ===========================================
        
        # Base labels from 4D matrix (16 combinations)
        MATRIX_LABELS = {
            # off_high, def_high, off_consistent, def_consistent
            "1111": "Elite Machine",           # Good offense, good defense, both consistent
            "1110": "Elite but Shaky D",       # Good O, good D, consistent O, volatile D
            "1101": "Streaky Juggernaut",      # Good O, good D, volatile O, consistent D
            "1100": "Talented Chaos",          # Good O, good D, both volatile
            "1011": "Offensive Engine",        # Good O, bad D, both consistent
            "1010": "Offensive Engine, Leaky", # Good O, bad D, consistent O, volatile D
            "1001": "Feast or Famine",         # Good O, bad D, volatile O, consistent D
            "1000": "Shootout Specialists",    # Good O, bad D, both volatile
            "0111": "Defensive Anchor",        # Bad O, good D, both consistent
            "0110": "Grinders",                # Bad O, good D, consistent O, volatile D  
            "0101": "Defensive Anchor, Streaky O", # Bad O, good D, volatile O, consistent D
            "0100": "Defensive Chaos",         # Bad O, good D, both volatile
            "0011": "Steady Mediocrity",       # Bad O, bad D, both consistent
            "0010": "Consistently Inconsistent", # Bad O, bad D, consistent O, volatile D
            "0001": "Struggling but Steady D", # Bad O, bad D, volatile O, consistent D
            "0000": "Full Rebuild Mode",       # Bad O, bad D, both volatile
        }
        
        base_label = MATRIX_LABELS.get(matrix_code, "Unclassified")
        
        # ===========================================
        # TREND MODIFIERS (can override or append)
        # ===========================================
        
        trend_modifier = ""
        trend_override = None
        
        # Check for overwhelming trends (these override the base label)
        if off_trend_sig == "overwhelming" and def_trend_sig == "overwhelming":
            if off_improving and def_improving:
                trend_override = "Rising Tide"
            elif not off_improving and not def_improving:
                trend_override = "Falling Apart"
            elif off_improving and not def_improving:
                trend_override = "Offensive Surge, Defensive Collapse"
            else:
                trend_override = "Defensive Surge, Offensive Collapse"
        elif off_trend_sig == "overwhelming":
            if off_improving:
                trend_override = "Offensive Breakout"
            else:
                trend_override = "Offensive Freefall"
        elif def_trend_sig == "overwhelming":
            if def_improving:
                trend_override = "Defensive Transformation"
            else:
                trend_override = "Defensive Meltdown"
        
        # Significant trends add modifiers (don't override)
        if trend_override is None:
            modifiers = []
            if off_trend_sig == "significant":
                if off_improving:
                    modifiers.append("↑O")
                else:
                    modifiers.append("↓O")
            if def_trend_sig == "significant":
                if def_improving:
                    modifiers.append("↑D")
                else:
                    modifiers.append("↓D")
            
            if modifiers:
                trend_modifier = " (" + ", ".join(modifiers) + ")"
        
        # Final archetype assignment
        if trend_override:
            archetype = trend_override
        else:
            archetype = base_label + trend_modifier
        
        team["archetype"] = archetype
        team["baseLabel"] = base_label
        team["trendModifier"] = trend_modifier
        team["trendOverride"] = trend_override is not None
        
        # Store tier classifications for reference
        team["offTier"] = "high" if off_high else "low"
        team["defTier"] = "high" if def_high else "low"
        team["offRugTier"] = "consistent" if off_consistent else "volatile"
        team["defRugTier"] = "consistent" if def_consistent else "volatile"
        team["offTrendTier"] = "improving" if off_improving else "declining"
        team["defTrendTier"] = "improving" if def_improving else "declining"

    # ----------------------------
    # Calculate League Awards
    # ----------------------------

    # Initialize awards for all teams
    for team in teams_data:
        team["awards"] = []

    # Most Consistent: Lowest total rugosity (offense + defense)
    most_consistent = min(teams_data, key=lambda t: t["offRugosity"] + t["defRugosity"])
    most_consistent["awards"].append("most_consistent")

    # Best Offense: Highest offensive rating adjusted by consistency
    # Higher off rating is better, lower rugosity is better (subtract it)
    best_offense = max(teams_data, key=lambda t: t["offLevel"] - (t["offRugosity"] * 0.5))
    best_offense["awards"].append("best_offense")

    # Best Defense: Lowest defensive rating adjusted by consistency
    # Lower def rating is better, lower rugosity is better (add it as penalty)
    best_defense = min(teams_data, key=lambda t: t["defLevel"] + (t["defRugosity"] * 0.5))
    best_defense["awards"].append("best_defense")

    # On the Rise: Highest recent trend score (last 20 games)
    on_the_rise = max(teams_data, key=lambda t: t["recentTrendScore"])
    on_the_rise["awards"].append("on_the_rise")

    # On the Decline: Lowest recent trend score (last 20 games)
    on_the_decline = min(teams_data, key=lambda t: t["recentTrendScore"])
    on_the_decline["awards"].append("on_the_decline")

    print(f"\n📊 League Awards:")
    print(f"   Most Consistent: {most_consistent['name']}")
    print(f"   Best Offense: {best_offense['name']}")
    print(f"   Best Defense: {best_defense['name']}")
    print(f"   On the Rise: {on_the_rise['name']}")
    print(f"   On the Decline: {on_the_decline['name']}")

# ----------------------------
# Output JSON
# ----------------------------
output = {
    "season": SEASON,
    "generated": pd.Timestamp.now().isoformat(),
    "window": WINDOW,
    "leagueMedians": league_medians,
    "trendThresholds": {
        "significant": 1.0,
        "overwhelming": 1.5
    },
    "teams": teams_data
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Saved to {OUTPUT_FILE}")
print(f"  {len(teams_data)} teams, {sum(t['games'] for t in teams_data)} total games")