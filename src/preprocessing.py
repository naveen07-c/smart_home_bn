"""
Preprocessing and Feature Engineering Pipeline for CASAS Datasets.
Covers Phases 2 - 15:
- Raw cleaning and timestamp parsing
- 5-second tumbling window aggregation
- 18 observable evidence variables
- Distribution analysis and discretization
- 6 behavioral weak labels
- Modeling dataset generation and train/validation/test chronological splitting
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
PROCESSED_DIR = "data/processed"
METRICS_DIR = "outputs/metrics"
FIGURES_DIR = "outputs/figures"

HOMES = ["aruba", "cairo", "milan"]

# Exact 11 room locations defined across the homes
ALL_ROOM_LOCATIONS = [
    "Bedroom", "Bathroom", "DiningRoom", "GuestRoom", "Kitchen",
    "LivingRoom", "LoungeChair", "OtherRoom", "OutsideDoor", "WorkArea", "Hall"
]

# Native locations per home as verified from Phase 1 data inspection
HOME_SENSORS = {
    "Aruba": [
        "Bedroom", "Bathroom", "DiningRoom", "GuestRoom", "Kitchen",
        "LivingRoom", "LoungeChair", "OtherRoom", "OutsideDoor", "WorkArea"
    ],
    "Cairo": [
        "Bedroom", "GuestRoom", "Hall", "Kitchen",
        "LivingRoom", "OtherRoom", "OutsideDoor", "WorkArea"
    ],
    "Milan": [
        "Bathroom", "Bedroom", "DiningRoom", "Kitchen",
        "LivingRoom", "LoungeChair", "OtherRoom", "OutsideDoor", "WorkArea"
    ]
}

HOME_DOOR_SEMANTICS = {
    "Aruba": True,   # Explicit OPEN / CLOSE events exist
    "Cairo": False,  # Only ON / OFF events exist
    "Milan": True    # Explicit OPEN / CLOSE events exist
}


def load_and_clean_raw_data(home_id: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Phase 3: Raw Event Cleaning
    1. Read without assuming header.
    2. Assign date, time, location, event.
    3. Construct timestamp from date + time.
    4. Remove rows with invalid timestamps.
    5. Strip whitespace from location and event.
    6. Sort chronologically.
    7. Remove exact duplicate records.
    """
    filepath = os.path.join(data_dir, f"{home_id.lower()}.csv")
    df = pd.read_csv(filepath, header=None, names=["date", "time", "location", "event"], dtype=str)
    
    # Strip whitespace
    df["location"] = df["location"].astype(str).str.strip()
    df["event"] = df["event"].astype(str).str.strip()
    
    # Construct timestamp
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], format="mixed", errors="coerce")
    
    # Drop invalid timestamps
    df = df.dropna(subset=["timestamp"])
    
    # Drop exact duplicates
    df = df.drop_duplicates(subset=["date", "time", "location", "event"])
    
    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["home_id"] = home_id.capitalize()
    return df


def aggregate_5s_windows(df_clean: pd.DataFrame, home_name: str) -> pd.DataFrame:
    """
    Phases 4-9: 5-Second Windowing and Observable Feature Construction.
    Aggregates events into tumbling 5-second windows:
    window_start = timestamp.floor("5s")
    """
    df = df_clean.copy()
    df["window_start"] = df["timestamp"].dt.floor("5s")
    
    native_locations = set(HOME_SENSORS[home_name])
    has_explicit_doors = HOME_DOOR_SEMANTICS[home_name]
    
    # Group by window_start
    windows = []
    
    # Fast vectorized pivots
    # 1. Location activity counts (number of events in location within window)
    loc_pivot = pd.crosstab(df["window_start"], df["location"])
    
    # 2. Door event flags
    if has_explicit_doors:
        door_open_series = df[(df["location"] == "OutsideDoor") & (df["event"] == "OPEN")].groupby("window_start").size()
        door_close_series = df[(df["location"] == "OutsideDoor") & (df["event"] == "CLOSE")].groupby("window_start").size()
    else:
        door_open_series = pd.Series(dtype=int)
        door_close_series = pd.Series(dtype=int)
        
    unique_windows = pd.Series(df["window_start"].unique()).sort_values().reset_index(drop=True)
    features_df = pd.DataFrame({"window_start": unique_windows})
    features_df["home_id"] = home_name
    
    # 11 Room Activity variables (numerical counts)
    for loc in ALL_ROOM_LOCATIONS:
        col_name = f"{loc}_Activity"
        if loc in native_locations:
            if loc in loc_pivot.columns:
                features_df[col_name] = features_df["window_start"].map(loc_pivot[loc]).fillna(0).astype(int)
            else:
                features_df[col_name] = 0
        else:
            # Mark sensor as non-existent in this home: -1 sentinel for unavailable
            features_df[col_name] = -1
            
    # Door Open / Close features
    if has_explicit_doors:
        features_df["OutsideDoor_Open"] = features_df["window_start"].map(door_open_series).fillna(0).astype(int)
        features_df["OutsideDoor_Close"] = features_df["window_start"].map(door_close_series).fillna(0).astype(int)
        features_df["Total_Contact_Events"] = features_df["OutsideDoor_Open"] + features_df["OutsideDoor_Close"]
    else:
        # Cairo: OutsideDoor ON/OFF events are door contact events, but explicit OPEN/CLOSE are unavailable
        features_df["OutsideDoor_Open"] = -1
        features_df["OutsideDoor_Close"] = -1
        # OutsideDoor ON/OFF counts as contact events
        if "OutsideDoor" in loc_pivot.columns:
            features_df["Total_Contact_Events"] = features_df["window_start"].map(loc_pivot["OutsideDoor"]).fillna(0).astype(int)
        else:
            features_df["Total_Contact_Events"] = 0
            
    # Temporal variables (Phase 8)
    dt_series = features_df["window_start"]
    hour = dt_series.dt.hour
    dayofweek = dt_series.dt.dayofweek
    
    # Time_Of_Day: Night (22-06), Morning (06-12), Afternoon (12-18), Evening (18-22)
    conditions_tod = [
        (hour >= 22) | (hour < 6),
        (hour >= 6) & (hour < 12),
        (hour >= 12) & (hour < 18),
        (hour >= 18) & (hour < 22)
    ]
    choices_tod = ["Night", "Morning", "Afternoon", "Evening"]
    features_df["Time_Of_Day"] = np.select(conditions_tod, choices_tod, default="Night")
    
    # Day_Of_Week: Weekday (0-4), Weekend (5-6)
    features_df["Day_Of_Week"] = np.where(dayofweek < 5, "Weekday", "Weekend")
    
    # Is_Night: Yes if hour >= 22 or hour < 6, else No
    features_df["Is_Night"] = np.where((hour >= 22) | (hour < 6), "Yes", "No")
    
    # Total_Activity (Phase 9): sum of active room counts for native locations
    native_cols = [f"{loc}_Activity" for loc in native_locations]
    features_df["Total_Activity"] = features_df[native_cols].sum(axis=1)
    
    return features_df


def analyze_distributions(df_all: pd.DataFrame, output_metrics_dir: str = METRICS_DIR, output_figures_dir: str = FIGURES_DIR):
    """
    Phase 10: Feature Distribution Analysis.
    Calculates summary percentiles and generates distribution plots.
    """
    os.makedirs(output_metrics_dir, exist_ok=True)
    os.makedirs(output_figures_dir, exist_ok=True)
    
    numerical_cols = [f"{loc}_Activity" for loc in ALL_ROOM_LOCATIONS] + ["Total_Activity", "Total_Contact_Events"]
    stats_list = []
    
    for col in numerical_cols:
        series = df_all[df_all[col] >= 0][col]  # Exclude -1 (unavailable)
        if series.empty:
            continue
        p = np.percentile(series, [0, 25, 50, 75, 90, 95, 99, 100])
        stats_list.append({
            "feature": col,
            "count": len(series),
            "min": float(p[0]),
            "p25": float(p[1]),
            "median": float(p[2]),
            "mean": float(series.mean()),
            "p75": float(p[3]),
            "p90": float(p[4]),
            "p95": float(p[5]),
            "p99": float(p[6]),
            "max": float(p[7])
        })
        
    df_stats = pd.DataFrame(stats_list)
    stats_path = os.path.join(output_metrics_dir, "feature_distributions.csv")
    df_stats.to_csv(stats_path, index=False)
    
    # Generate distribution figures
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(data=df_all[df_all["Total_Activity"] >= 0], x="Total_Activity", bins=30, ax=axes[0, 0], color="royalblue")
    axes[0, 0].set_title("Total Activity Distribution per 5s Window")
    axes[0, 0].set_yscale("log")
    
    sns.histplot(data=df_all[df_all["Bedroom_Activity"] >= 0], x="Bedroom_Activity", bins=20, ax=axes[0, 1], color="salmon")
    axes[0, 1].set_title("Bedroom Activity Count Distribution")
    axes[0, 1].set_yscale("log")
    
    sns.histplot(data=df_all[df_all["Kitchen_Activity"] >= 0], x="Kitchen_Activity", bins=20, ax=axes[1, 0], color="mediumseagreen")
    axes[1, 0].set_title("Kitchen Activity Count Distribution")
    axes[1, 0].set_yscale("log")
    
    sns.countplot(data=df_all, x="Time_Of_Day", order=["Night", "Morning", "Afternoon", "Evening"], ax=axes[1, 1], hue="Time_Of_Day", legend=False, palette="Set2")
    axes[1, 1].set_title("Events by Time of Day")
    
    plt.tight_layout()
    fig_path = os.path.join(output_figures_dir, "activity_distributions.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    print(f"Distribution analysis saved to {stats_path} and {fig_path}")
    return df_stats


def get_discretization_rules():
    """
    Phase 11: Documented Discretization Rules.
    Distinguishes:
    - sensor does not exist in home: 'Unavailable'
    - sensor exists and 0 activity: 'None'
    - sensor exists and low activity (1-2 events): 'Low'
    - sensor exists and high activity (>=3 events): 'High'
    """
    rules = {
        "room_activity": {
            "unavailable": "Unavailable",
            "none": {"condition": "count == 0", "state": "None"},
            "low": {"condition": "1 <= count <= 2", "state": "Low"},
            "high": {"condition": "count >= 3", "state": "High"}
        },
        "total_activity": {
            "none": {"condition": "count == 0", "state": "None"},
            "low": {"condition": "1 <= count <= 2", "state": "Low"},
            "medium": {"condition": "3 <= count <= 5", "state": "Medium"},
            "high": {"condition": "count >= 6", "state": "High"}
        },
        "door_open_close": {
            "unavailable": "Unavailable",
            "no": {"condition": "count == 0", "state": "No"},
            "yes": {"condition": "count > 0", "state": "Yes"}
        },
        "total_contact": {
            "none": {"condition": "count == 0", "state": "None"},
            "contact": {"condition": "count > 0", "state": "Yes"}
        }
    }
    return rules


def discretize_evidence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 11: Discretization into Bayesian categorical states.
    Ensures exact compliance with project state schema.
    """
    df_disc = df.copy()
    
    # 1. Room Activities: Unavailable, None, Low, High
    for loc in ALL_ROOM_LOCATIONS:
        col = f"{loc}_Activity"
        val = df_disc[col]
        conditions = [
            val < 0,
            val == 0,
            (val >= 1) & (val <= 2),
            val >= 3
        ]
        choices = ["Unavailable", "None", "Low", "High"]
        df_disc[col] = np.select(conditions, choices, default="None")
        
    # 2. OutsideDoor_Open and OutsideDoor_Close
    for door_col in ["OutsideDoor_Open", "OutsideDoor_Close"]:
        val = df_disc[door_col]
        conditions = [
            val < 0,
            val == 0,
            val > 0
        ]
        choices = ["Unavailable", "No", "Yes"]
        df_disc[door_col] = np.select(conditions, choices, default="No")
        
    # 3. Total_Activity: None, Low, Medium, High
    tot = df_disc["Total_Activity"]
    conditions_tot = [
        tot == 0,
        (tot >= 1) & (tot <= 2),
        (tot >= 3) & (tot <= 5),
        tot >= 6
    ]
    choices_tot = ["None", "Low", "Medium", "High"]
    df_disc["Total_Activity"] = np.select(conditions_tot, choices_tot, default="Low")
    
    # 4. Total_Contact_Events: No, Yes
    cnt = df_disc["Total_Contact_Events"]
    df_disc["Total_Contact_Events"] = np.where(cnt > 0, "Yes", "No")
    
    return df_disc


def generate_behavioral_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phases 12 & 13: Six Behavioral Variables (Inferred / Weak Labels)
    19. Home_Occupancy: [Empty, Occupied]
    20. Resident_Activity: [Inactive, Low, Moderate, High]
    21. Resident_Sleep: [Awake, Sleeping]
    22. Leaving_Home: [No, Yes]
    23. Returning_Home: [No, Yes]
    24. Unusual_Activity: [Normal, Unusual]
    """
    df_out = df.copy()
    
    # Compute time delta to previous and next window
    time_series = df_out["window_start"]
    prev_gap = time_series.diff().dt.total_seconds().fillna(5.0)
    next_gap = (-time_series.diff(-1)).dt.total_seconds().fillna(5.0)
    
    is_door_active = (df_out["OutsideDoor_Activity"].isin(["Low", "High"])) | (df_out["Total_Contact_Events"] == "Yes")
    total_act = df_out["Total_Activity"]
    is_night = df_out["Is_Night"] == "Yes"
    bedroom_act = df_out["Bedroom_Activity"].isin(["Low", "High"])
    kitchen_act = df_out["Kitchen_Activity"].isin(["Low", "High"])
    living_act = df_out["LivingRoom_Activity"].isin(["Low", "High"])
    
    # 1. Returning_Home:
    # Occurs when there was a long prior absence gap (> 15 minutes = 900s) and door is activated followed by activity
    returning_cond = (prev_gap >= 600) & (is_door_active | (total_act.isin(["Low", "Medium", "High"])))
    df_out["Returning_Home"] = np.where(returning_cond, "Yes", "No")
    
    # 2. Leaving_Home:
    # Occurs when door is activated and followed by prolonged silence (> 10 minutes = 600s gap)
    leaving_cond = is_door_active & (next_gap >= 600)
    df_out["Leaving_Home"] = np.where(leaving_cond, "Yes", "No")
    
    # 3. Home_Occupancy:
    # Since windows exist because events occurred, occupancy is 'Occupied' during bursts.
    # It transitions to 'Empty' if a leaving event just finalized or isolated door trigger without indoor follow-up.
    df_out["Home_Occupancy"] = np.where(df_out["Leaving_Home"] == "Yes", "Empty", "Occupied")
    
    # 4. Resident_Activity: [Inactive, Low, Moderate, High]
    cond_act = [
        (df_out["Home_Occupancy"] == "Empty") | (total_act == "None"),
        total_act == "Low",
        total_act == "Medium",
        total_act == "High"
    ]
    choice_act = ["Inactive", "Low", "Moderate", "High"]
    df_out["Resident_Activity"] = np.select(cond_act, choice_act, default="Low")
    
    # 5. Resident_Sleep: [Awake, Sleeping]
    # Inferred when: Is_Night is Yes, in Bedroom (or inactive at night), no other room activity (kitchen, living, work), Occupied
    sleep_cond = (
        is_night & 
        (df_out["Home_Occupancy"] == "Occupied") & 
        (~kitchen_act) & 
        (~living_act) & 
        (df_out["Resident_Activity"].isin(["Inactive", "Low"]))
    )
    df_out["Resident_Sleep"] = np.where(sleep_cond, "Sleeping", "Awake")
    
    # 6. Unusual_Activity: [Normal, Unusual]
    # Behavioral deviation / anomaly:
    # a) High non-bedroom activity at night (e.g. Kitchen or LivingRoom High at 2AM-5AM)
    # b) Outside door activity in the middle of the night (Is_Night == Yes)
    # c) Active movement when occupancy would otherwise be Empty
    unusual_cond = (
        (is_night & is_door_active) |
        (is_night & (total_act == "High") & (~bedroom_act)) |
        ((df_out["Home_Occupancy"] == "Empty") & (total_act.isin(["Medium", "High"])))
    )
    df_out["Unusual_Activity"] = np.where(unusual_cond, "Unusual", "Normal")
    
    return df_out


def build_modeling_dataset(save_intermediate: bool = True):
    """
    Executes Phases 2 - 15:
    Loads all 3 homes, constructs 5s windows, 18 evidence variables, 6 behavioral labels,
    performs chronological 70/15/15 train/val/test split per home, and saves datasets.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    home_dfs = []
    
    for home in HOMES:
        home_cap = home.capitalize()
        print(f"Processing raw data for {home_cap}...")
        df_clean = load_and_clean_raw_data(home_cap)
        df_feat = aggregate_5s_windows(df_clean, home_cap)
        home_dfs.append(df_feat)
        
    df_all_raw = pd.concat(home_dfs, ignore_index=True)
    
    # Phase 10: Distributions
    print("Running distribution analysis...")
    analyze_distributions(df_all_raw)
    
    # Save discretization rules
    rules = get_discretization_rules()
    with open(os.path.join(METRICS_DIR, "discretization_rules.json"), "w") as f:
        json.dump(rules, f, indent=2)
        
    # Phase 11: Discretize
    print("Discretizing evidence features...")
    df_all_disc = discretize_evidence_features(df_all_raw)
    
    # Phase 12-13: Behavioral weak labels (per home to maintain chronological sequence)
    print("Generating 6 behavioral weak labels...")
    labeled_dfs = []
    for home in ["Aruba", "Cairo", "Milan"]:
        df_h = df_all_disc[df_all_disc["home_id"] == home].sort_values("window_start").reset_index(drop=True)
        df_labeled = generate_behavioral_weak_labels(df_h)
        labeled_dfs.append(df_labeled)
        
    df_modeling = pd.concat(labeled_dfs, ignore_index=True)
    
    # 24 core modeling columns (18 evidence + 6 behavioral)
    evidence_cols = [
        "Bedroom_Activity", "Bathroom_Activity", "DiningRoom_Activity", "GuestRoom_Activity",
        "Kitchen_Activity", "LivingRoom_Activity", "LoungeChair_Activity", "OtherRoom_Activity",
        "OutsideDoor_Activity", "WorkArea_Activity", "Hall_Activity",
        "OutsideDoor_Open", "OutsideDoor_Close",
        "Time_Of_Day", "Day_Of_Week", "Is_Night",
        "Total_Activity", "Total_Contact_Events"
    ]
    behavioral_cols = [
        "Home_Occupancy", "Resident_Activity", "Resident_Sleep",
        "Leaving_Home", "Returning_Home", "Unusual_Activity"
    ]
    
    core_cols = ["home_id", "window_start"] + evidence_cols + behavioral_cols
    df_modeling = df_modeling[core_cols]
    
    # Phase 14: Save modeling dataset
    modeling_path = os.path.join(PROCESSED_DIR, "modeling_dataset.csv")
    df_modeling.to_csv(modeling_path, index=False)
    print(f"Saved complete modeling dataset to {modeling_path} (shape: {df_modeling.shape})")
    
    # Phase 15: Chronological Split per home: 70% train, 15% val, 15% test
    print("Performing chronological train / validation / test split...")
    train_list, val_list, test_list = [], [], []
    split_meta = {}
    
    for home in ["Aruba", "Cairo", "Milan"]:
        df_h = df_modeling[df_modeling["home_id"] == home].sort_values("window_start").reset_index(drop=True)
        n = len(df_h)
        idx_train = int(n * 0.70)
        idx_val = int(n * 0.85)
        
        train_h = df_h.iloc[:idx_train]
        val_h = df_h.iloc[idx_train:idx_val]
        test_h = df_h.iloc[idx_val:]
        
        train_list.append(train_h)
        val_list.append(val_h)
        test_list.append(test_h)
        
        split_meta[home] = {
            "total_windows": n,
            "train_windows": len(train_h),
            "train_start": str(train_h["window_start"].iloc[0]),
            "train_end": str(train_h["window_start"].iloc[-1]),
            "val_windows": len(val_h),
            "val_start": str(val_h["window_start"].iloc[0]),
            "val_end": str(val_h["window_start"].iloc[-1]),
            "test_windows": len(test_h),
            "test_start": str(test_h["window_start"].iloc[0]),
            "test_end": str(test_h["window_start"].iloc[-1])
        }
        
    df_train = pd.concat(train_list, ignore_index=True)
    df_val = pd.concat(val_list, ignore_index=True)
    df_test = pd.concat(test_list, ignore_index=True)
    
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "validation.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    split_meta_path = os.path.join(METRICS_DIR, "split_metadata.json")
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    with open(split_meta_path, "w") as f:
        json.dump(split_meta, f, indent=2)
        
    print(f"Saved {train_path} ({len(df_train)} rows)")
    print(f"Saved {val_path} ({len(df_val)} rows)")
    print(f"Saved {test_path} ({len(df_test)} rows)")
    print(f"Saved split metadata to {split_meta_path}")
    
    return df_modeling, df_train, df_val, df_test


if __name__ == "__main__":
    build_modeling_dataset()
