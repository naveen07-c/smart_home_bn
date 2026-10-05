"""
Dataset inspection module for CASAS smart home datasets.
Generates dataset_summary.csv, location_summary.csv, and event_summary.csv.
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPUTS_METRICS_DIR = "outputs/metrics"
HOMES = ["aruba", "cairo", "milan"]

def inspect_datasets():
    os.makedirs(OUTPUTS_METRICS_DIR, exist_ok=True)
    
    dataset_summaries = []
    location_records = []
    event_records = []
    
    for home in HOMES:
        filepath = os.path.join(DATA_DIR, f"{home}.csv")
        # CASAS raw format: date, time, location, event (no header)
        df_raw = pd.read_csv(filepath, header=None, names=["date", "time", "location", "event"], dtype=str)
        
        n_rows = len(df_raw)
        n_cols = len(df_raw.columns)
        col_names = ";".join(df_raw.columns)
        dtypes = ";".join([str(t) for t in df_raw.dtypes])
        
        missing_values = int(df_raw.isnull().sum().sum())
        duplicate_rows = int(df_raw.duplicated().sum())
        
        # Clean whitespace for analysis
        df_clean = df_raw.copy()
        df_clean["location"] = df_clean["location"].str.strip()
        df_clean["event"] = df_clean["event"].str.strip()
        
        # Timestamps
        dt_series = pd.to_datetime(df_clean["date"] + " " + df_clean["time"], format="mixed", errors="coerce")
        invalid_timestamps = int(dt_series.isnull().sum())
        valid_dt = dt_series.dropna()
        min_ts = str(valid_dt.min()) if not valid_dt.empty else "N/A"
        max_ts = str(valid_dt.max()) if not valid_dt.empty else "N/A"
        
        unique_locs = sorted(df_clean["location"].dropna().unique().tolist())
        unique_evts = sorted(df_clean["event"].dropna().unique().tolist())
        
        dataset_summaries.append({
            "home_id": home.capitalize(),
            "rows": n_rows,
            "columns": n_cols,
            "column_names": col_names,
            "data_types": dtypes,
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "invalid_timestamps": invalid_timestamps,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
            "unique_locations_count": len(unique_locs),
            "unique_locations": ";".join(unique_locs),
            "unique_events_count": len(unique_evts),
            "unique_events": ";".join(unique_evts),
        })
        
        # Location summary
        loc_counts = df_clean["location"].value_counts()
        for loc, count in loc_counts.items():
            location_records.append({
                "home_id": home.capitalize(),
                "location": loc,
                "event_count": int(count),
                "percentage": float(count / n_rows * 100)
            })
            
        # Event summary
        evt_counts = df_clean["event"].value_counts()
        for evt, count in evt_counts.items():
            event_records.append({
                "home_id": home.capitalize(),
                "event": evt,
                "event_count": int(count),
                "percentage": float(count / n_rows * 100)
            })
            
    df_dataset_summary = pd.DataFrame(dataset_summaries)
    df_loc_summary = pd.DataFrame(location_records)
    df_evt_summary = pd.DataFrame(event_records)
    
    ds_path = os.path.join(OUTPUTS_METRICS_DIR, "dataset_summary.csv")
    loc_path = os.path.join(OUTPUTS_METRICS_DIR, "location_summary.csv")
    evt_path = os.path.join(OUTPUTS_METRICS_DIR, "event_summary.csv")
    
    df_dataset_summary.to_csv(ds_path, index=False)
    df_loc_summary.to_csv(loc_path, index=False)
    df_evt_summary.to_csv(evt_path, index=False)
    
    print(f"Generated {ds_path}")
    print(f"Generated {loc_path}")
    print(f"Generated {evt_path}")
    return df_dataset_summary, df_loc_summary, df_evt_summary

if __name__ == "__main__":
    df_ds, df_loc, df_evt = inspect_datasets()
    print("\nDataset Summary:")
    print(df_ds.to_string(index=False))
