import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ——— Configuration ———
fastf1.Cache.enable_cache('f1_cache')
TRACK_LENGTH_KM   = 6.174    # km
TIME_DELTA_THRESH = 5.0      # s gap ⇒ “clean air”
MAX_FRESH_LAPS    = 3        # first 3 laps after pit

# ——— 1) Load & preprocess ———
session = fastf1.get_session(2024, "Saudi Arabia", "R")
session.load()
laps = session.laps.dropna(
    subset=["LapTime","Sector1Time","Sector2Time","Sector3Time"]
).copy()

# Convert to seconds
for c in ["LapTime","Sector1Time","Sector2Time","Sector3Time"]:
    laps[f"{c}_s"] = laps[c].dt.total_seconds()

# Compute gap to leader at lap start
laps["TimeDelta"] = (
    laps["LapStartTime"]
    - laps.groupby("LapNumber")["LapStartTime"].transform("min")
).dt.total_seconds()

# ——— 2) Filter out non‑green, pit‑in/out, out‑laps, inaccurate laps ———
valid_laps = laps[
    (laps["LapNumber"] > 1) &
    laps["IsAccurate"] &
    (laps["TrackStatus"] == "1") &          # green‑flag laps only
    laps["PitInTime"].isna() &
    laps["PitOutTime"].isna()
]
print(f"Found {len(valid_laps)} valid green‑flag laps without pits.")

# ——— 3) Overall race pace (all compounds) ———
avg_all = valid_laps.groupby("Driver")["LapTime_s"].mean().sort_values()
norm_all = avg_all / TRACK_LENGTH_KM
overall_df = pd.DataFrame({
    "Average Race Pace (s)":       avg_all,
    "Normalized Pace (s/km)":      norm_all
})
print("\n=== Overall Race Pace (all compounds) ===")
print(overall_df.to_string())

# ——— 4) Auto‑select best compound (fresh+clean) ———
counts = {
    comp: (
        (valid_laps["Compound"] == comp) &
        valid_laps["FreshTyre"] &
        (valid_laps["TyreLife"] <= MAX_FRESH_LAPS) &
        (valid_laps["TimeDelta"] >= TIME_DELTA_THRESH)
    ).sum()
    for comp in valid_laps["Compound"].dropna().unique()
}
best_comp = max(counts, key=counts.get)
print(f"\nCounts per compound (fresh+clean): {counts}")
print(f"→ Best compound is '{best_comp}' with {counts[best_comp]} laps")

# ——— 5) Race pace on best compound ———
race_laps = valid_laps[
    (valid_laps["Compound"] == best_comp) &
    valid_laps["FreshTyre"] &
    (valid_laps["TyreLife"] <= MAX_FRESH_LAPS) &
    (valid_laps["TimeDelta"] >= TIME_DELTA_THRESH)
]
avg_best = race_laps.groupby("Driver")["LapTime_s"].mean().sort_values()
norm_best = avg_best / TRACK_LENGTH_KM
best_df = pd.DataFrame({
    "Average Race Pace (s)":       avg_best,
    "Normalized Pace (s/km)":      norm_best
})
print(f"\n=== Race Pace on {best_comp} (fresh+clean) ===")
print(best_df.to_string())

# ——— 6) Per‑compound breakdown (all valid laps) ———
comp_dfs = []
for comp in valid_laps["Compound"].unique():
    sel = valid_laps[valid_laps["Compound"] == comp]
    comp_avg = sel.groupby("Driver")["LapTime_s"].mean().rename(comp)
    comp_dfs.append(comp_avg)
comp_compare = pd.concat(comp_dfs, axis=1)
print("\n=== Avg Lap Times by Compound ===")
print(comp_compare.to_string())

# ——— 7) Plot overall vs best ———
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
overall_df["Normalized Pace (s/km)"].plot.bar()
plt.title("Overall Race Pace")
plt.ylabel("s per km")

plt.subplot(1, 2, 2)
best_df["Normalized Pace (s/km)"].plot.bar(color="C1")
plt.title(f"Pace on {best_comp}")

plt.tight_layout()
plt.show()
