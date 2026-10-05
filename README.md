# Probabilistic Smart Home Decision Network

A complete semester-level implementation of a **26-node Bayesian / Decision Network** applied to three CASAS smart home datasets (Aruba, Cairo, Milan).  
The system performs probabilistic inference over resident behavioral states and selects home-automation actions using **Maximum Expected Utility (MEU)**.

---

## Dataset Description

| Home  | Raw Events | Date Range | Unique Locations | Event Types |
|-------|-----------|------------|-----------------|-------------|
| Aruba | 1,602,821 | 2010-11-04 → 2011-06-11 | 10 | ON, OFF, OPEN, CLOSE |
| Cairo |   647,487 | 2009-06-10 → 2009-08-05 |  8 | ON, OFF |
| Milan |   421,392 | 2009-10-16 → 2010-01-06 |  9 | ON, OFF, OPEN, CLOSE |

**Important notes:**
- Cairo uses `ON`/`OFF` for OutsideDoor — not `OPEN`/`CLOSE`. This is preserved, not converted.
- Missing sensors (e.g. Cairo has no Bathroom) are represented as `Unavailable`, not zero.
- No variables were fabricated: temperature, humidity, electricity, HVAC, intrusion ground-truth do NOT exist in CASAS and were not invented.

---

## Installation

```bash
# 1. Clone / enter the project directory
cd SMART_HOME_BN

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
# Activate venv first
source .venv/bin/activate

# Run the complete pipeline
python main.py

# Run unit tests
pytest tests/test_pipeline.py -v
```

---

## Project Structure

```
SMART_HOME_BN/
├── data/
│   ├── aruba.csv
│   ├── cairo.csv
│   ├── milan.csv
│   └── processed/
│       ├── modeling_dataset.csv   (1,078,797 × 26)
│       ├── train.csv              (755,156 rows)
│       ├── validation.csv         (161,820 rows)
│       └── test.csv               (161,821 rows)
│
├── src/
│   ├── inspect_datasets.py    Phase 1:  Dataset inspection
│   ├── preprocessing.py       Phases 2–15: Cleaning, windowing, features, labels
│   ├── model.py               Phases 16–20: 26-node DAG, CPT learning, inference
│   ├── decision.py            Phases 21–24: Decision node, utility, MEU, baseline
│   └── evaluation.py          Phases 25–30: Metrics, latency, cross-home, figures
│
├── outputs/
│   ├── FINAL_RESULTS.md
│   ├── figures/               (10 PNG files)
│   ├── metrics/               (CSV / JSON summaries)
│   └── models/                (bayesian_network.pkl)
│
├── tests/
│   └── test_pipeline.py       10 unit tests (all pass)
│
├── main.py                    Single entry point
├── requirements.txt
├── TRACKER.txt                Phase-by-phase status tracker
└── project_plan.md            Original project specification
```

---

## Preprocessing Pipeline

Each of the three homes is processed by a single generic `CASASPreprocessor`:

1. **Raw Cleaning** — parse `date`+`time` → `timestamp` (sub-second precision), strip whitespace, drop exact duplicates, sort chronologically.
2. **5-Second Windowing** — non-overlapping tumbling windows: `window_start = timestamp.floor("5s")`
3. **18 Observable Evidence Variables** constructed per window:
   - 11 room activity counts (native locations only; absent sensors = `Unavailable`)
   - 2 door variables (OPEN/CLOSE where available; `Unavailable` for Cairo)
   - 3 temporal (Time_Of_Day, Day_Of_Week, Is_Night)
   - 2 aggregates (Total_Activity, Total_Contact_Events)
4. **Discretization** — into categorical Bayesian states using documented thresholds (see `outputs/metrics/discretization_rules.json`)
5. **6 Behavioral Weak Labels** — transparent heuristic rules documented in `src/preprocessing.py`
6. **Chronological Split** — 70%/15%/15% per home; no shuffling

---

## 26-Node Network Architecture

```
Evidence (18)           Behavioral (6)         Decision (1)   Utility (1)
─────────────           ──────────────         ────────────   ─────────
Room Activity ×11  →    Home_Occupancy    →    Smart_Home  →  Homeowner
Door Events ×2     →    Resident_Activity      _Action        _Utility
Temporal ×3        →    Resident_Sleep
Aggregates ×2      →    Leaving_Home
                        Returning_Home
                        Unusual_Activity
```

- **39 directed edges**, max parent set = 4, acyclic (validated)
- **Node schema contract:** implemented exactly as specified — no silent redesign

---

## Bayesian Model

| Parameter | Value |
|-----------|-------|
| Library | pgmpy 1.1.2 `DiscreteBayesianNetwork` |
| Parameter Estimator | `DiscreteBayesianEstimator` |
| Prior | BDeu (equivalent sample size = 5) |
| CPTs Learned | 24 variables |
| Inference Algorithm | Variable Elimination (exact) |
| Avg Inference Latency | **0.984 ms per window** |

---

## Decision Model

**5 Available Actions:** `No_Action`, `Monitor`, `Notify_Resident`, `Silent_Alert`, `Local_Alert`

**MEU Formula:**
```
EU(action) = Σ P(state | evidence) × U(action, state)
best_action = argmax EU(action)
```

Utility values account for: safety, resident disruption, false-alert cost, and response appropriateness.  
Full table at `outputs/metrics/utility_table.csv`.

**Rule-Based Baseline:**
```
IF Unusual_Activity = Unusual  → Local_Alert
ELIF Returning_Home = Yes      → Monitor
ELIF Leaving_Home = Yes        → Monitor
ELSE                           → No_Action
```

---

## Results Summary

### Behavioral Inference (Test Set, n=3,000)

| Variable | Accuracy | Weighted F1 |
|----------|----------|------------|
| Home_Occupancy | **99.97%** | 0.9995 |
| Resident_Activity | **99.97%** | 0.9995 |
| Resident_Sleep | 98.10% | 0.9796 |
| Leaving_Home | **99.97%** | 0.9995 |
| Returning_Home | 99.33% | 0.9900 |
| Unusual_Activity | 99.87% | 0.9985 |

### Decision Comparison

| System | Mean Utility | False-Alert Rate |
|--------|-------------|-----------------|
| Bayesian Decision Network | **10.173** | **0.0%** |
| Rule-Based Baseline | 10.173 | 0.0% |

### Cross-Home Generalization

| Experiment | Occupancy Acc | Sleep Acc |
|-----------|--------------|-----------|
| Aruba → Aruba | 100% | 96.0% |
| Aruba → Milan | 100% | 97.8% |
| Aruba+Milan → Cairo | 100% | 98.1% |

### Inference Latency
- Mean: **0.984 ms** | Median: **0.946 ms** | P95: **1.170 ms**

---

## Limitations

1. CASAS is an activity-event dataset — no temperature, humidity, HVAC, or intrusion ground truth.
2. Six behavioral variables are **weakly labeled** — not human-annotated ground truth.
3. `Unusual_Activity` is behavioral deviation, not verified intrusion detection.
4. Sensor availability differs across homes — handled via `Unavailable` state.
5. Cairo lacks explicit OPEN/CLOSE events — preserved as-is, not converted.
6. Utility values are project-defined assumptions, not validated homeowner preferences.

---

## Figures

All 10 figures in `outputs/figures/`:
`dataset_event_distribution.png`, `activity_distributions.png`, `activity_over_time.png`,
`network_dag.png`, `example_posterior_output.png`, `action_distribution.png`,
`utility_comparison.png`, `confusion_matrices.png`, `cross_home_performance.png`,
`latency_distribution.png`
# smart_home_bn
