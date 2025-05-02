# Race Pace Analysis README

A concise log of how the original “broader” `racepace.py` evolved into the final, self‑configuring script.

---

## 📜 Intro

This script computes “race pace” at the 2024 Saudi Arabian Grand Prix by:

1. Loading FastF1 lap data  
2. Filtering for valid, green‑flag laps  
3. Calculating per‑driver average lap times (all compounds)  
4. Automatically selecting the richest “fresh+clean” stint compound  
5. Producing a secondary table of per‑compound averages  

---

## 🔑 Key Changes

1. **Green‑Flag & No‑Pit Filtering**  
   - **Before:** Included all laps > 1, regardless of track status or pit events.  
   - **Now:** Excludes out‑laps/in‑laps, pit‑in/out laps and any non‑green‑flag or inaccurate timing.

2. **TimeDelta (“Clean Air”)**  
   - **Before:** Attempted to filter on a missing `TimeDelta` column.  
   - **Now:** Computed as:  
     ```python
     laps["LapStartTime"]  
       - laps.groupby("LapNumber")["LapStartTime"].transform("min")
     ```  
     ⇒ gap to lap leader at the start of each lap.

3. **Dynamic Compound Selection**  
   - **Before:** Hard‑coded to “Soft” (with manual fallbacks).  
   - **Now:**  
     - Prints all compounds present.  
     - Counts “fresh+clean” laps (first 3 laps post‑pit, `TimeDelta ≥ 5 s`) per compound.  
     - Automatically picks the compound with the most such laps for the primary metric.

4. **Overall vs. Best‑Compound Pace**  
   - **Overall Pace:** Mean lap time across **all** valid green‑flag laps (all compounds).  
   - **Best‑Compound Pace:** Mean lap time on the automatically chosen compound in fresh+clean stints.

5. **Comprehensive Compound Breakdown**  
   - **Before:** Only showed drivers with fresh+clean stints on each tire → many NaNs.  
   - **Now:** Builds per‑compound averages from **all** valid laps (Lap > 1 & accurate) so every driver appears for each tire they ran.

---

## 🚀 Usage

1. Clone or drop into your project: `racepace.py`  
2. Install dependencies:  
   ```bash
   pip install fastf1 pandas numpy matplotlib
