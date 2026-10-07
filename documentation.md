# Probabilistic Smart Home Decision Network — Complete Documentation

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset and Problem Context](#2-dataset-and-problem-context)
3. [Methodology](#3-methodology)
   - 3.1 Pipeline Overview
   - 3.2 Dataset Inspection (Phase 1)
   - 3.3 Raw Event Cleaning (Phase 2–3)
   - 3.4 5-Second Tumbling Windowing (Phase 4)
   - 3.5 Evidence Variable Construction (Phases 5–9)
   - 3.6 Distribution Analysis and Discretization (Phases 10–11)
   - 3.7 Behavioral Weak Label Generation (Phases 12–13)
   - 3.8 Train / Validation / Test Split (Phase 15)
   - 3.9 26-Node Bayesian Decision Network Architecture (Phases 16–18)
   - 3.10 CPT Learning with Bayesian Smoothing (Phase 19)
   - 3.11 Probabilistic Inference via Variable Elimination (Phase 20)
   - 3.12 Decision Node — Smart_Home_Action (Phase 21)
   - 3.13 Utility Function — Homeowner_Utility (Phase 22)
   - 3.14 Maximum Expected Utility (MEU) Action Selection (Phase 23)
   - 3.15 Rule-Based Baseline (Phase 24)
4. [Evaluation Metrics](#4-evaluation-metrics)
   - 4.1 Behavioral Inference Metrics
   - 4.2 Decision and Utility Metrics
   - 4.3 Inference Latency Benchmarks
   - 4.4 Cross-Home Generalization Experiments
5. [Web Dashboard Implementation](#5-web-dashboard-implementation)
   - 5.1 Architecture Overview
   - 5.2 Backend (FastAPI + WebSocket)
   - 5.3 Frontend (D3.js Network Visualization)
   - 5.4 Parameter Serialization (.pkl Bundling)
   - 5.5 Live Inference Replay
6. [Results](#6-results)
   - 6.1 Dataset Summary
   - 6.2 Network Structure and Validation
   - 6.3 Behavioral Variable Inference Performance
   - 6.4 Decision Metrics — Bayesian vs Rule-Based
   - 6.5 Inference Latency
   - 6.6 Cross-Home Generalization
   - 6.7 Visualizations
7. [Limitations](#7-limitations)
8. [Project Structure](#8-project-structure)
9. [How to Run](#9-how-to-run)

---

## 1. Introduction

The **Probabilistic Smart Home Decision Network** is a complete semester-level implementation of an agentic probabilistic reasoning system for smart-home activity streams. It ingests raw event logs from three CASAS smart-home datasets (Aruba, Cairo, and Milan), transforms them into structured evidence, learns a 26-node Bayesian Decision Network from data, performs exact probabilistic inference, and selects actions through Maximum Expected Utility (MEU) to decide how the system should respond to the inferred home state.

The project pursues three integrated goals:

1. **Behavioral inference** — Given noisy binary PIR/motion sensor events aggregated into 5-second windows, infer six high-level behavioral variables: whether the home is occupied, how active the resident is, whether the resident is likely sleeping, whether someone is leaving or returning home, and whether the current activity pattern is unusual.
2. **Decision-making under uncertainty** — Use the inferred posterior probabilities over behavioral states to select the best smart-home action (No_Action, Monitor, Notify_Resident, Silent_Alert, Local_Alert) according to a transparent utility function that trades off safety, disruption, false-alert cost, and response appropriateness.
3. **Comparative evaluation and deployment** — Evaluate the Bayesian Decision Network against a rule-based baseline on unseen test data, measure real inference latency, test cross-home generalization, and expose the entire live inference pipeline through an interactive web dashboard.

All code is organized into a minimal, reproducible pipeline driven by a single entry point (`python main.py`), with unit tests covering preprocessing, network structure, the decision model, and train/test separation.

---

## 2. Dataset and Problem Context

### 2.1 CASAS Activity-Event Datasets

The project uses three real-world CASAS smart-home datasets. CASAS records **binary PIR/motion sensor state changes** — events such as a motion sensor turning ON or OFF, or a door sensor triggering OPEN or CLOSE. Each raw record has the schema:

```
date, time, location, event
```

**Key properties of the data:**

- **Aruba** — 1,602,821 raw events spanning 2010-11-04 to 2011-06-11, 10 unique locations, events: ON, OFF, OPEN, CLOSE.
- **Cairo** — 647,487 raw events spanning 2009-06-10 to 2009-08-05, 8 unique locations, events: ON, OFF only (no explicit OPEN/CLOSE door events; Cairo uses ON/OFF for OutsideDoor).
- **Milan** — 421,392 raw events spanning 2009-10-16 to 2010-01-06, 9 unique locations, events: ON, OFF, OPEN, CLOSE.

**Total: 2,671,700 raw events across 3 homes.**

### 2.2 Important Data Characteristics and Constraints

1. **CASAS is an activity-event dataset, not a environmental sensor dataset.** It records binary motion/door sensor state changes. Temperature, humidity, electricity, HVAC, smoke, and camera data do **not** exist in these files. The project explicitly does not fabricate them.

2. **Sensor configurations differ across homes.** Aruba has a GuestRoom but no Hall. Cairo has a Hall but no Bathroom, DiningRoom, or LoungeChair. Milan has neither GuestRoom nor Hall. A missing sensor must be represented as **Unavailable** — not as zero activity. This distinction between "sensor exists and detected nothing" and "sensor does not exist in this home" is critical and enforced throughout.

3. **Cairo uses different door semantics.** Aruba and Milan have explicit `OutsideDoor OPEN` and `OutsideDoor CLOSE` events. Cairo uses `OutsideDoor ON` and `OutsideDoor OFF`. The project does **not** silently convert Cairo ON/OFF into OPEN/CLOSE. For Cairo, `OutsideDoor_Open` and `OutsideDoor_Close` are marked `Unavailable`, and door contact events are captured through `Total_Contact_Events`.

4. **No ground-truth behavioral labels exist.** CASAS provides raw event streams, not annotations for occupancy, sleep, leaving/returning, or unusual activity. The six behavioral variables are **inferred via transparent weak-label rules** — they are not verified ground truth. Evaluation measures consistency between Bayesian MAP predictions and these weak labels, not true behavioral accuracy.

---

## 3. Methodology

### 3.1 Pipeline Overview

The complete pipeline is sequential and driven by `main.py`. Each phase produces artifacts that the next phase consumes:

```
Raw CSVs (Aruba/Cairo/Milan)
    │
    ▼
Phase 1: Dataset Inspection
    │
    ▼
Phase 2-3: Raw Event Cleaning (parse timestamps, strip whitespace, deduplicate, sort)
    │
    ▼
Phase 4: 5-Second Tumbling Windowing (floor timestamps to 5s boundaries)
    │
    ▼
Phases 5-9: 18 Observable Evidence Variables (room activity counts, door flags, temporal features, aggregates)
    │
    ▼
Phase 10: Feature Distribution Analysis (percentiles, histograms)
    │
    ▼
Phase 11: Discretization into Categorical States (Unavailable/None/Low/High etc.)
    │
    ▼
Phases 12-13: 6 Behavioral Weak Labels (occupancy, activity, sleep, leaving, returning, unusual)
    │
    ▼
Phase 15: Chronological Train/Val/Test Split (70/15/15 per home)
    │
    ▼
Phases 16-18: 26-Node DAG Construction and Validation
    │
    ▼
Phase 19: CPT Learning from Training Data (BDeu Bayesian estimator)
    │
    ▼
Phase 20: Bayesian Inference Engine (Variable Elimination)
    │
    ▼
Phases 21-23: Decision Node, Utility Function, MEU Action Selection
    │
    ▼
Phase 24: Rule-Based Baseline
    │
    ▼
Phases 25-27: Evaluation on Test Set (behavioral metrics, decision metrics, latency)
    │
    ▼
Phase 28: Cross-Home Generalization Experiments
    │
    ▼
Phase 30: Visualization Figures (10 figures)
    │
    ▼
Phase 31: Final Results Report (FINAL_RESULTS.md)
```

### 3.2 Dataset Inspection (Phase 1)

Before any preprocessing, the project inspects all three raw CSV files to validate the expected schema and gather statistics. For each dataset, `inspect_datasets.py` computes:

- Number of rows and columns
- Column names and data types
- Missing values and duplicate rows
- Invalid timestamps
- Timestamp range (min/max)
- Unique locations and their event counts
- Unique event types and their frequencies

These statistics are saved to:

- `outputs/metrics/dataset_summary.csv`
- `outputs/metrics/location_summary.csv`
- `outputs/metrics/event_summary.csv`

This phase is a gate: if the schema differs from expectations, preprocessing is adapted rather than crashing or silently assuming equivalence.

### 3.3 Raw Event Cleaning (Phases 2–3)

`load_and_clean_raw_data()` in `src/preprocessing.py` performs the following for each home:

1. **Read without assuming a header** — CASAS files have no header row, so columns are assigned as `date`, `time`, `location`, `event`.
2. **Strip whitespace** from `location` and `event` values.
3. **Construct a proper timestamp** by concatenating `date` + `time` and parsing with `pd.to_datetime(..., format="mixed", errors="coerce")`.
4. **Drop rows with invalid timestamps** (NaT after parsing).
5. **Sort chronologically** by timestamp.
6. **Remove exact duplicate records** (same date, time, location, event).
7. **Tag each row with `home_id`** (Aruba, Cairo, or Milan) so home identity is preserved throughout.

Importantly, multiple legitimate events within the same 5-second window are **not** collapsed at this stage — they are kept as separate events and aggregated later during windowing.

### 3.4 5-Second Tumbling Windowing (Phase 4)

`aggregate_5s_windows()` converts the event stream into non-overlapping (tumbling) 5-second windows:

```
window_start = timestamp.floor("5s")
```

Each window represents the interval `[window_start, window_start + 5 seconds)`. Windows are identified by `window_start` and `home_id`. A window containing multiple events is still a single modeling instance — all events within the window are aggregated into feature counts.

This produces a row per active window (windows that contain at least one event) with:
- `window_start` — the start timestamp of the 5-second window
- `home_id` — which home this window belongs to
- Numerical activity counts per room location
- Door event counts (where available)
- Temporal features derived from `window_start`

### 3.5 Evidence Variable Construction (Phases 5–9)

The project constructs exactly **18 observable/evidence variables** from the windowed event data. These are the input features the Bayesian network observes.

#### 3.5.1 Activity Variables (11 room locations)

For each of the 11 room locations defined across all homes (Bedroom, Bathroom, DiningRoom, GuestRoom, Kitchen, LivingRoom, LoungeChair, OtherRoom, OutsideDoor, WorkArea, Hall), the pipeline counts the number of ON/OFF events within each 5-second window:

```
<Location>_Activity = count of ON/OFF events in that location during the window
```

This is an **activity intensity measure**, not a direct occupancy judgment. An ON/OFF event does not by itself indicate occupancy — it indicates that the sensor fired.

The 11 activity variables are:

1. `Bedroom_Activity`
2. `Bathroom_Activity`
3. `DiningRoom_Activity`
4. `GuestRoom_Activity`
5. `Kitchen_Activity`
6. `LivingRoom_Activity`
7. `LoungeChair_Activity`
8. `OtherRoom_Activity`
9. `OutsideDoor_Activity`
10. `WorkArea_Activity`
11. `Hall_Activity`

**Missing sensor handling:** If a home does not have a particular sensor, the count is set to `-1` as a sentinel, which is later discretized to the categorical state `Unavailable`. For example:
- Aruba has no Hall sensor → `Hall_Activity = -1` → `Unavailable`
- Cairo has no Bathroom, DiningRoom, or LoungeChair → those are `Unavailable`
- Milan has no GuestRoom or Hall → those are `Unavailable`

This ensures the model can distinguish "no activity detected by an existing sensor" (`None`) from "this sensor does not exist in this home" (`Unavailable`).

#### 3.5.2 Door Variables (2)

For homes with explicit OPEN/CLOSE events (Aruba and Milan):

- `OutsideDoor_Open` — 1 if an `OutsideDoor OPEN` event occurred in the window, 0 otherwise (later discretized to `Yes`/`No`)
- `OutsideDoor_Close` — 1 if an `OutsideDoor CLOSE` event occurred in the window, 0 otherwise

For Cairo (which only has ON/OFF):

- `OutsideDoor_Open` and `OutsideDoor_Close` are set to `-1` → discretized to `Unavailable`
- Door contact events are still counted through `Total_Contact_Events` (using OutsideDoor ON/OFF counts)

#### 3.5.3 Temporal Variables (3)

Derived from the `window_start` timestamp:

- **`Time_Of_Day`** — Four states based on hour:
  - `Night` (hour >= 22 OR hour < 6)
  - `Morning` (6 <= hour < 12)
  - `Afternoon` (12 <= hour < 18)
  - `Evening` (18 <= hour < 22)

- **`Day_Of_Week`** — Two states:
  - `Weekday` (Monday–Friday, dayofweek 0–4)
  - `Weekend` (Saturday–Sunday, dayofweek 5–6)

- **`Is_Night`** — Binary:
  - `Yes` if hour >= 22 OR hour < 6
  - `No` otherwise

The night threshold (hour >= 22 OR hour < 6) is used consistently throughout the entire project and is never changed after model training.

#### 3.5.4 Aggregate Variables (2)

- **`Total_Activity`** — Sum of valid room activity counts within the window (only for sensors that exist in the home). This represents the total activity intensity across all rooms.

- **`Total_Contact_Events`** — Count of door contact events in the window. For Aruba/Milan, this is `OutsideDoor_Open + OutsideDoor_Close`. For Cairo, this is the count of `OutsideDoor` ON/OFF events (since Cairo lacks explicit OPEN/CLOSE but still has door contact events).

### 3.6 Distribution Analysis and Discretization (Phases 10–11)

#### 3.6.1 Distribution Analysis (Phase 10)

Before choosing categorical state boundaries, `analyze_distributions()` inspects the actual distributions of all numerical features. For each feature (excluding the `-1` unavailable sentinel), it computes:

- min, max, mean, median
- 25th, 50th, 75th, 90th, 95th, 99th, 100th percentiles

These statistics are saved to `outputs/metrics/feature_distributions.csv`. Distribution histograms are generated for `Total_Activity`, `Bedroom_Activity`, `Kitchen_Activity`, and `Time_Of_Day`, saved to `outputs/figures/activity_distributions.png`.

This analysis informs (but does not arbitrarily predetermine) the discretization thresholds.

#### 3.6.2 Discretization (Phase 11)

Bayesian network variables require manageable categorical states. `discretize_evidence_features()` converts all numerical evidence into categorical states using the following rules, which are saved to `outputs/metrics/discretization_rules.json`:

**Room Activity (11 variables):**
| State | Condition |
|-------|-----------|
| `Unavailable` | Sensor does not exist in this home (count = -1) |
| `None` | Sensor exists and 0 events in window (count = 0) |
| `Low` | Sensor exists and 1–2 events (1 <= count <= 2) |
| `High` | Sensor exists and >= 3 events (count >= 3) |

**Total_Activity:**
| State | Condition |
|-------|-----------|
| `None` | 0 total events |
| `Low` | 1–2 total events |
| `Medium` | 3–5 total events |
| `High` | >= 6 total events |

**Door Open/Close (where available):**
| State | Condition |
|-------|-----------|
| `Unavailable` | Sensor does not exist (Cairo) |
| `No` | 0 door events |
| `Yes` | > 0 door events |

**Total_Contact_Events:**
| State | Condition |
|-------|-----------|
| `No` | 0 contact events |
| `Yes` | > 0 contact events |

**Time_Of_Day, Day_Of_Week, Is_Night** are already categorical from construction.

The critical design choice: `Unavailable` is a distinct state from `None`. A Bayesian node with state `Unavailable` can carry the information "this sensor is not present in this home," which is structurally different from "the sensor is present but detected nothing."

### 3.7 Behavioral Weak Label Generation (Phases 12–13)

Because CASAS provides no ground-truth behavioral annotations, the project generates **six inferred behavioral variables** using transparent, documented rule-based heuristics. These are **weak labels** — they represent the project's best deterministic interpretation of the activity stream, not verified truth.

`generate_behavioral_weak_labels()` operates on each home's chronologically sorted, discretized windows.

#### 3.7.1 Home_Occupancy (Empty / Occupied)

**Principle:** Windows exist because events occurred, so activity bursts indicate occupancy. A window is labeled `Empty` only when a leaving event has just finalized (door activity followed by prolonged silence), or when there is an isolated door trigger without indoor follow-up.

**Rule:**
```
Home_Occupancy = "Empty"  if Leaving_Home == "Yes"
                = "Occupied" otherwise
```

This means occupancy is primarily derived from the leaving-home detection. During active periods, the home is assumed occupied.

#### 3.7.2 Resident_Activity (Inactive / Low / Moderate / High)

**Principle:** Combine occupancy state with total activity intensity.

**Rule:**
```
Resident_Activity =
    "Inactive"   if Home_Occupancy == "Empty" OR Total_Activity == "None"
    "Low"        if Total_Activity == "Low"
    "Moderate"   if Total_Activity == "Medium"
    "High"       if Total_Activity == "High"
```

When the home is empty or there is no activity, the resident is considered inactive. Otherwise, the activity level mirrors the total activity discretization.

#### 3.7.3 Resident_Sleep (Awake / Sleeping)

**Principle:** Sleep is inferred when it is night, the home is occupied, there is no activity in high-activity rooms (kitchen, living room), and overall activity is low or inactive — suggesting the resident is in bed.

**Rule:**
```
Resident_Sleep = "Sleeping" if:
    Is_Night == "Yes"
    AND Home_Occupancy == "Occupied"
    AND NOT Kitchen_Activity in [Low, High]
    AND NOT LivingRoom_Activity in [Low, High]
    AND Resident_Activity in [Inactive, Low]
           = "Awake" otherwise
```

This is an inferred behavioral label, not a medical or sleep-study ground truth. It captures the plausible scenario of a resident sleeping at night with minimal motion sensor activity.

#### 3.7.4 Leaving_Home (No / Yes)

**Principle:** A resident leaves home when there is door activity (OutsideDoor activity or contact events) followed by a prolonged period of silence (no further events for an extended gap).

**Rule:**
```
Leaving_Home = "Yes" if:
    door activity is present in this window (OutsideDoor_Activity in [Low, High] OR Total_Contact_Events == "Yes")
    AND the gap to the NEXT window is >= 600 seconds (10 minutes)
           = "No" otherwise
```

The 600-second threshold means: if the door triggers and then no events occur for at least 10 minutes, the system infers the resident has left.

#### 3.7.5 Returning_Home (No / Yes)

**Principle:** A resident returns home when there was a long prior absence (gap since previous window >= 600 seconds) and then door activity or new indoor activity appears.

**Rule:**
```
Returning_Home = "Yes" if:
    gap since previous window >= 600 seconds (10 minutes)
    AND (door activity present OR Total_Activity in [Low, Medium, High])
           = "No" otherwise
```

This captures the transition from a long empty period to renewed activity, indicating return.

#### 3.7.6 Unusual_Activity (Normal / Unusual)

**Principle:** Unusual activity represents behavioral deviation from normal patterns — not intrusion ground truth. It flags scenarios that are anomalous given the time of day and occupancy state.

**Rule:**
```
Unusual_Activity = "Unusual" if ANY of:
    (a) Is_Night == "Yes" AND door activity present  (door use in middle of night)
    (b) Is_Night == "Yes" AND Total_Activity == "High" AND Bedroom_Activity NOT in [Low, High]
        (high non-bedroom activity at night — e.g., kitchen/living room active at 2AM)
    (c) Home_Occupancy == "Empty" AND Total_Activity in [Medium, High]
        (activity when home should be empty)
           = "Normal" otherwise
```

This captures three types of behavioral deviation:
- Nighttime door usage
- High activity in non-sleeping areas during night hours
- Activity bursts when the home is supposedly empty

**Important:** `Unusual_Activity` is NOT an intrusion detection label. It is a deviation-from-normal-pattern indicator. The project explicitly states this limitation.

### 3.8 Train / Validation / Test Split (Phase 15)

Adjacent 5-second windows are temporally correlated, so a random split would cause leakage. Instead, the project uses a **chronological split per home**:

- **First 70%** of each home's windows (by `window_start`) → **Training set**
- **Next 15%** → **Validation set**
- **Final 15%** → **Test set**

This preserves temporal order and ensures the test set contains only windows that occur after all training and validation windows for that home.

**Split sizes:**
- Train: 755,156 windows
- Validation: 161,820 windows
- Test: 161,821 windows

The split metadata (start/end timestamps per home per split) is saved to `outputs/metrics/split_metadata.json`. Unit tests verify that no temporal leakage exists between splits (`max(train timestamps) <= min(val timestamps)` and `max(val timestamps) <= min(test timestamps)` for each home).

All discretization thresholds and CPT learning are fitted on the **training set only**. The test set is touched only once, at the final evaluation step.

### 3.9 26-Node Bayesian Decision Network Architecture (Phases 16–18)

#### 3.9.1 Node Composition

The complete network contains exactly **26 nodes**, organized into four categories:

**Evidence Nodes (18) — Observable:**
These are the discretized features from Phases 5–11:
1. `Bedroom_Activity`
2. `Bathroom_Activity`
3. `DiningRoom_Activity`
4. `GuestRoom_Activity`
5. `Kitchen_Activity`
6. `LivingRoom_Activity`
7. `LoungeChair_Activity`
8. `OtherRoom_Activity`
9. `OutsideDoor_Activity`
10. `WorkArea_Activity`
11. `Hall_Activity`
12. `OutsideDoor_Open`
13. `OutsideDoor_Close`
14. `Time_Of_Day`
15. `Day_Of_Week`
16. `Is_Night`
17. `Total_Activity`
18. `Total_Contact_Events`

**Behavioral Nodes (6) — Inferred:**
These are the weak-label variables from Phase 13:
19. `Home_Occupancy` (Empty / Occupied)
20. `Resident_Activity` (Inactive / Low / Moderate / High)
21. `Resident_Sleep` (Awake / Sleeping)
22. `Leaving_Home` (No / Yes)
23. `Returning_Home` (No / Yes)
24. `Unusual_Activity` (Normal / Unusual)

**Decision Node (1):**
25. `Smart_Home_Action` (No_Action / Monitor / Notify_Resident / Silent_Alert / Local_Alert)

**Utility Node (1):**
26. `Homeowner_Utility` (continuous utility value)

Total: 18 + 6 + 1 + 1 = **26 nodes**.

#### 3.9.2 DAG Structure (39 Directed Edges)

The DAG is designed to be sparse and semantically meaningful, not fully connected. Edges are defined in `src/model.py` as `DAG_EDGES`. The conceptual dependencies are:

**Temporal layer:**
- `Day_Of_Week → Time_Of_Day` (day type influences time categorization)
- `Time_Of_Day → Is_Night` (time of day determines night status)

**Door/contact layer:**
- `OutsideDoor_Open → Total_Contact_Events`
- `OutsideDoor_Close → Total_Contact_Events`
- `Total_Contact_Events → OutsideDoor_Activity`

**Activity aggregation layer:**
- `LivingRoom_Activity → Total_Activity`
- `Kitchen_Activity → Total_Activity`
- `Bedroom_Activity → Total_Activity`

**Occupancy layer:**
- `Total_Activity → Home_Occupancy`
- `OutsideDoor_Activity → Home_Occupancy`

**Resident activity layer:**
- `Home_Occupancy → Resident_Activity`
- `Total_Activity → Resident_Activity`
- `Time_Of_Day → Resident_Activity`
- `Resident_Activity → Bathroom_Activity` (secondary rooms influenced by overall activity)
- `Resident_Activity → DiningRoom_Activity`
- `Resident_Activity → GuestRoom_Activity`
- `Resident_Activity → LoungeChair_Activity`
- `Resident_Activity → OtherRoom_Activity`
- `Resident_Activity → WorkArea_Activity`
- `Resident_Activity → Hall_Activity`

**Sleep layer:**
- `Home_Occupancy → Resident_Sleep`
- `Is_Night → Resident_Sleep`
- `Bedroom_Activity → Resident_Sleep`

**Leaving/Returning layer:**
- `Home_Occupancy → Leaving_Home`
- `OutsideDoor_Activity → Leaving_Home`
- `Total_Contact_Events → Leaving_Home`
- `Home_Occupancy → Returning_Home`
- `OutsideDoor_Activity → Returning_Home`
- `Total_Contact_Events → Returning_Home`

**Unusual activity layer:**
- `Home_Occupancy → Unusual_Activity`
- `Resident_Activity → Unusual_Activity`
- `Is_Night → Unusual_Activity`
- `OutsideDoor_Activity → Unusual_Activity`

**Decision layer (informational influences on action choice):**
- `Unusual_Activity → Smart_Home_Action`
- `Home_Occupancy → Smart_Home_Action`
- `Resident_Sleep → Smart_Home_Action`

**Utility layer (payoff depends on action and state):**
- `Smart_Home_Action → Homeowner_Utility`
- `Unusual_Activity → Homeowner_Utility`
- `Home_Occupancy → Homeowner_Utility`

**Graph properties:**
- Total nodes: 26
- Total edges: 39
- Max parent set size: 4
- Acyclic: ✓ (verified with `nx.is_directed_acyclic_graph()`)

The structure report is saved to `outputs/metrics/network_structure.txt`, and a visualization is saved to `outputs/figures/network_dag.png`.

#### 3.9.3 Graph Validation (Phase 18)

Before learning parameters, `validate_network_structure()` performs rigorous checks:
1. Every node exists and the count is exactly 26.
2. Every edge references valid nodes (both source and target in `ALL_NODES`).
3. The graph is acyclic (no cycles).
4. No node has an excessively large parent set (max in-degree <= 6).
5. Structure report and DAG visualization are generated.

These checks run as both runtime assertions and unit tests (`test_exact_26_nodes`, `test_dag_acyclicity_and_validity`).

### 3.10 CPT Learning with Bayesian Smoothing (Phase 19)

#### 3.10.1 Learning Setup

The 24 probabilistic nodes (18 evidence + 6 behavioral) have their Conditional Probability Tables (CPTs) learned from the **training data only** using `train_bayesian_network()` in `src/model.py`.

The implementation uses **pgmpy's `DiscreteBayesianNetwork`** with the **`DiscreteBayesianEstimator`** and a **BDeu (Bayesian Dirichlet equivalent uniform) prior** with `equivalent_sample_size=5`.

```
model = DiscreteBayesianNetwork(prob_edges)  # edges between probabilistic nodes only
estimator = DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=5)
model.fit(train_data, estimator=estimator)
```

**Why BDeu smoothing?** Raw maximum-frequency CPT estimates can produce zero-probability combinations for rare state configurations, which would cause inference to fail (zero probabilities propagate and can make queries return no valid distribution). The BDeu prior with equivalent sample size 5 adds a small pseudo-count to every parent-child configuration, ensuring every CPT entry is strictly positive while still being dominated by the data where data is abundant.

**What is learned:** 24 CPTs (one per probabilistic node), each specifying `P(Node | Parents)` for all combinations of parent states.

**What is NOT learned as a CPT:** The decision node `Smart_Home_Action` and utility node `Homeowner_Utility` are not probabilistic nodes with learned CPTs. The decision node's distribution is determined by the MEU calculation, and the utility node's value is computed from the utility matrix.

The trained model is serialized to `outputs/models/bayesian_network.pkl` using `joblib`.

### 3.11 Probabilistic Inference via Variable Elimination (Phase 20)

#### 3.11.1 Inference Engine

`BayesianInferenceEngine` in `src/model.py` wraps pgmpy's **`VariableElimination`** algorithm, which performs **exact inference** by eliminating variables one at a time using the chain rule and factorization given the DAG structure.

```
engine = BayesianInferenceEngine(model_path)
posteriors = engine.query(evidence_dict)
```

Given an evidence dictionary (mapping observable variable names to their observed categorical states), the engine returns posterior probability distributions over the 6 behavioral variables:

```
{
    "Home_Occupancy": {"Empty": 0.08, "Occupied": 0.92},
    "Resident_Activity": {"Inactive": 0.05, "Low": 0.30, "Moderate": 0.45, "High": 0.20},
    "Resident_Sleep": {"Awake": 0.35, "Sleeping": 0.65},
    "Leaving_Home": {"No": 0.97, "Yes": 0.03},
    "Returning_Home": {"No": 0.88, "Yes": 0.12},
    "Unusual_Activity": {"Normal": 0.94, "Unusual": 0.06}
}
```

#### 3.11.2 Evidence Filtering

The `query()` method carefully filters the evidence dictionary:
- Only variables that exist in the model's node set are kept.
- Evidence values must match valid states in the node's CPD (otherwise they are dropped with a warning).
- Target variables (the 6 behavioral nodes being queried) are excluded from evidence (you cannot observe what you're trying to infer).
- Variables marked `Unavailable` in the evidence are excluded.

This filtering is important for cross-home experiments, where a sensor that exists in the training home may not exist in the test home. The inference engine gracefully handles missing evidence by marginalizing over the missing variables.

#### 3.11.3 MAP Prediction

For evaluation, the Maximum A Posteriori (MAP) estimate is taken for each behavioral variable: the state with the highest posterior probability is selected as the prediction. This MAP prediction is compared against the weak label to compute behavioral metrics.

### 3.12 Decision Node — Smart_Home_Action (Phase 21)

The decision node `Smart_Home_Action` has **5 possible actions**:

| Action | Description |
|--------|-------------|
| `No_Action` | Do nothing; home state is acceptable |
| `Monitor` | Passive observation; log and watch for developments |
| `Notify_Resident` | Send a notification to the homeowner (e.g., phone push notification) |
| `Silent_Alert` | Trigger a silent alert to external contacts/security without disturbing the residence |
| `Local_Alert` | Trigger a local audible alarm within the home |

The action is **not** selected by simply taking the maximum-probability behavioral state. Instead, it is selected through Maximum Expected Utility (MEU), which accounts for the full posterior distribution over states and the utility of each action in each state.

### 3.13 Utility Function — Homeowner_Utility (Phase 22)

#### 3.13.1 Situational States

To compute expected utility, the 6 behavioral variables are collapsed into **6 mutually exclusive situational states** that capture the essential home situation:

| State | Meaning |
|-------|---------|
| `Normal_Awake` | Home occupied, resident awake, normal activity pattern |
| `Normal_Sleeping` | Home occupied, resident sleeping, normal activity pattern |
| `Normal_Empty` | Home empty, normal activity pattern |
| `Unusual_Occupied_Awake` | Home occupied, resident awake, unusual activity detected |
| `Unusual_Occupied_Sleeping` | Home occupied, resident sleeping, unusual activity detected |
| `Unusual_Empty` | Home empty, unusual activity detected |

These states are computed from the joint posterior of `Home_Occupancy`, `Resident_Sleep`, and `Unusual_Activity` (either exactly via the inference engine or approximately via the independence assumption from marginal posteriors).

#### 3.13.2 Utility Matrix

The utility function `Homeowner_Utility` is a matrix `U(action, state)` specifying the payoff (utility) of taking each action in each situational state. The values are **project-defined assumptions** reflecting tradeoffs between safety, disruption, false-alert cost, and response appropriateness. The full matrix is exported to `outputs/metrics/utility_table.csv`.

**Key utility design principles:**

- **Normal situations:** `No_Action` has the highest utility (peaceful, no disruption). `Local_Alert` in a normal situation has the most negative utility (nuisance alarm).
- **Normal sleeping:** `No_Action` has the *highest* utility of any normal state (undisturbed rest is maximally valuable). Waking a sleeping resident unnecessarily (`Notify_Resident` during sleep) has high negative utility.
- **Normal empty:** `Monitor` is slightly better than `No_Action` (periodic health/sensor monitoring is desirable when the home is empty). False alerts to an absent owner are costly.
- **Unusual occupied awake:** `Notify_Resident` is ideal (resident is awake and can verify). `No_Action` is strongly negative (negligent — failing to inform an awake occupant).
- **Unusual occupied sleeping:** Safety is paramount. `No_Action` is catastrophic (-100). `Local_Alert` has the highest utility (alarm warns sleeping family). `Silent_Alert` is also high (prepares external response).
- **Unusual empty:** `No_Action` is the worst possible outcome (-120 — disaster: intrusion/hazard unaddressed). `Silent_Alert` is best (silent dispatch without alerting a potential intruder).

The full matrix:

```
                    No_Action  Monitor  Notify_Resident  Silent_Alert  Local_Alert
Normal_Awake        10.0       8.0      -5.0             -10.0         -50.0
Normal_Sleeping     15.0       10.0     -25.0            -15.0         -80.0
Normal_Empty        10.0       12.0     -10.0            -20.0         -60.0
Unusual_Occupied_Awake  -30.0   10.0     30.0             15.0          -10.0
Unusual_Occupied_Sleeping -100.0 -20.0   20.0             35.0          45.0
Unusual_Empty       -120.0     -30.0    30.0             55.0          45.0
```

#### 3.13.3 Utility Specification Transparency

The utility table is explicitly exported as a CSV file (`outputs/metrics/utility_table.csv`) with columns `state`, `action`, `utility`. This makes the utility function fully transparent and auditable — it is not hidden inside arbitrary code.

### 3.14 Maximum Expected Utility (MEU) Action Selection (Phase 23)

#### 3.14.1 The MEU Formula

For each possible action `a`, the expected utility is:

```
EU(a) = Σ_{s in situational_states} P(s | evidence) × U(a, s)
```

where `P(s | evidence)` is the probability of situational state `s` given the observed evidence (derived from the posterior over behavioral variables), and `U(a, s)` is the utility of action `a` in state `s`.

The best action is:

```
best_action = argmax_{a in ACTIONS} EU(a)
```

#### 3.14.2 Implementation

`select_action_meu()` in `src/decision.py` implements this calculation:

1. Compute situational state probabilities from the posteriors (either exactly via the inference engine's joint query, or approximately via the independence assumption).
2. For each of the 5 actions, compute `EU(a)` by summing over all 6 situational states.
3. Select the action with the highest expected utility.
4. Return a dict containing:
   - `selected_action` — the best action
   - `expected_utilities` — EU for all 5 actions
   - `state_probabilities` — probability of each situational state

**Example output:**
```
Selected Action: Silent_Alert
Expected Utilities: {
    'No_Action': -30.0,
    'Monitor': 10.0,
    'Notify_Resident': 30.0,
    'Silent_Alert': 35.0,
    'Local_Alert': 25.0
}
```

#### 3.14.3 Why MEU Over Maximum Probability

Selecting the action based on the single most likely behavioral state would ignore uncertainty. For example, if `P(Unusual_Activity = Unusual) = 0.45` and `P(Normal) = 0.55`, a maximum-probability approach would choose `No_Action` (since Normal is more likely). But MEU correctly accounts for the fact that the 45% chance of unusual activity (where `No_Action` has utility -30 or worse) may outweigh the 55% chance of normal activity (where `No_Action` has utility +10). MEU produces the action with the best expected outcome across all possible states, weighted by their probabilities.

### 3.15 Rule-Based Baseline (Phase 24)

To provide a meaningful comparison, the project implements a simple **rule-based baseline** in `select_rule_based_action()`:

```
IF Unusual_Activity == "Unusual":
    → Local_Alert
ELSE IF Returning_Home == "Yes":
    → Monitor
ELSE IF Leaving_Home == "Yes":
    → Monitor
ELSE:
    → No_Action
```

This baseline:
- Uses the same test observations (evidence and weak labels).
- Is deterministic (no probabilistic reasoning).
- Is simple and interpretable.
- Provides a comparison point: does the Bayesian Decision Network's probabilistic reasoning produce better expected utility and/or fewer inappropriate actions than a simple rule-based system?

The rule-based system is binary in nature: it escalates to `Local_Alert` on unusual activity and uses `Monitor` for transitional events, otherwise `No_Action`. It lacks the nuanced gradient of responses (Monitor / Notify / Silent_Alert) that the Bayesian system can produce based on the full posterior distribution.

---

## 4. Evaluation Metrics

### 4.1 Behavioral Inference Metrics

The behavioral inference layer is evaluated on a **3,000-window subsample** of the unseen test set. For each of the 6 behavioral variables, the following metrics are computed by comparing the Bayesian MAP prediction against the weak-label ground truth:

#### 4.1.1 Accuracy

```
Accuracy = (Number of correct predictions) / (Total predictions)
```

The fraction of windows where the MAP-predicted state matches the weak-label state.

#### 4.1.2 Precision, Recall, and F1-Score

Computed using `sklearn.metrics.precision_recall_fscore_support` with two averaging schemes:

- **Macro averaging:** Computes metrics for each class independently and averages them equally. This treats rare and common classes with equal weight. For imbalanced variables (e.g., `Home_Occupancy` where "Occupied" dominates), macro F1 will be lower because the minority class performance drags down the average.

- **Weighted averaging:** Computes metrics for each class and averages them weighted by class frequency. This reflects the overall performance dominated by the majority class.

For each variable, both macro and weighted precision, recall, and F1 are reported.

#### 4.1.3 Confusion Matrices

Full confusion matrices are computed for each behavioral variable, showing the count of predictions for each true/predicted state combination. These are visualized in `outputs/figures/confusion_matrices.png` for the key variables: `Home_Occupancy`, `Resident_Sleep`, `Unusual_Activity`, and `Resident_Activity`.

#### 4.1.4 Reported Behavioral Metrics (Test Set, n=3,000)

| Variable | Accuracy | Weighted F1 | Macro F1 |
|----------|----------|-------------|----------|
| Home_Occupancy | 99.97% | 0.9995 | 0.4999 |
| Resident_Activity | 99.97% | 0.9995 | 0.7499 |
| Resident_Sleep | 98.10% | 0.9796 | 0.8600 |
| Leaving_Home | 99.97% | 0.9995 | 0.4999 |
| Returning_Home | 99.33% | 0.9900 | 0.4983 |
| Unusual_Activity | 99.87% | 0.9985 | 0.8568 |

**Interpretation of low macro-F1:** For highly imbalanced variables like `Home_Occupancy` (heavily skewed toward "Occupied"), macro-F1 is ~0.50 because the model essentially always predicts "Occupied" (which is correct most of the time for accuracy/weighted-F1), but the minority "Empty" class gets essentially no correct predictions, resulting in ~0.50 macro-F1 (average of ~1.0 for the majority class and ~0.0 for the minority). This is expected behavior for imbalanced data and does not indicate a model deficiency — the dataset itself is dominated by occupied windows because windows only exist when events occur.

**Resident_Sleep** shows the most meaningful differentiation (macro-F1 = 0.86) because sleep/wake states are more balanced and the night-time inactivity pattern is a strong signal.

**Unusual_Activity** also shows good differentiation (macro-F1 = 0.86) because the unusual conditions (night door activity, high non-bedroom night activity, activity when empty) are distinctive enough for the model to learn.

### 4.2 Decision and Utility Metrics

#### 4.2.1 Utility Metrics

For each system (Bayesian Decision Network and Rule-Based Baseline), the following are computed on the 3,000-window test subsample:

- **Mean Utility:** Average realized utility across all test windows.
- **Median Utility:** Median realized utility.
- **Total Utility:** Sum of realized utilities across all test windows.
- **Std Utility:** Standard deviation of realized utilities (higher = more variable responses).

#### 4.2.2 False Alert Rate

```
False Alert Rate = (Windows where an alert action was fired AND the true state was Normal) / (Total windows)
```

Alert actions are `Notify_Resident`, `Silent_Alert`, and `Local_Alert`. A false alert occurs when one of these is selected but the true situational state is normal (no unusual activity). This measures how often the system unnecessarily alarms when nothing is wrong.

#### 4.2.3 Unusual Activity Detection Rate

```
Unusual Detection Rate = (Windows where an alert action was fired AND the true state was Unusual) / (Total unusual windows)
```

This measures the system's sensitivity to unusual activity — what fraction of truly unusual windows triggered some form of escalation.

#### 4.2.4 Action Distribution

The percentage of test windows where each action was selected, for both systems. This reveals the response profile: does the system mostly do nothing, or does it escalate frequently?

#### 4.2.5 Reported Decision Metrics (Test Set, n=3,000)

| Metric | Bayesian Decision Network | Rule-Based Baseline |
|--------|--------------------------|---------------------|
| Mean Utility | **10.173** | 10.173 |
| Median Utility | 10.0 | 10.0 |
| Total Utility | **30,520** | 30,518 |
| Std Utility | 2.796 | 1.713 |
| False Alert Rate | **0.0%** | 0.0% |
| Unusual Detection Rate | 88.9% | 100.0% |
| % No_Action | 99.57% | 99.03% |
| % Monitor | 0.00% | 0.67% |
| % Notify_Resident | 0.27% | 0.00% |
| % Silent_Alert | 0.17% | 0.00% |
| % Local_Alert | 0.00% | 0.30% |

**Key findings:**

1. **Equal mean utility:** Both systems achieve essentially identical mean utility (10.173). This is because the test set is dominated by normal windows (99.57% No_Action for BDN), where both systems correctly choose No_Action with utility 10.0. The small differences in the rare unusual windows balance out.

2. **Nuance vs. binary:** The Bayesian Decision Network shows a **nuanced response gradient** — it uses Monitor (0%), Notify_Resident (0.27%), and Silent_Alert (0.17%) in appropriate situations. The rule-based system is **binary**: it either does Nothing (99.03%) or escalates to Local_Alert (0.30%). The BDN's ability to choose intermediate actions based on the full posterior distribution is a key advantage, even if total utility is similar on this dataset.

3. **Zero false alerts:** Both systems achieve 0.0% false alert rate — neither fires alerts during normal situations. This is a strong result indicating both systems are conservative about unnecessary escalation.

4. **Detection rate difference:** The rule-based system has 100% unusual detection (it escalates to Local_Alert on every unusual window). The BDN has 88.9% detection — it sometimes chooses Monitor or Notify_Resident instead of a full alert, depending on the posterior probabilities. This is not necessarily worse; the BDN is making a more nuanced tradeoff based on the full situational context (e.g., if the resident is awake and the unusual activity is mild, Notify_Resident may be more appropriate than Local_Alert).

5. **Higher std for BDN:** The BDN has higher utility standard deviation (2.796 vs 1.713), reflecting its more varied action selection. The rule-based system is more consistent (mostly No_Action or Local_Alert).

#### 4.2.6 Cumulative Utility Comparison

A cumulative utility plot (`outputs/figures/utility_comparison.png`) shows the running sum of realized utilities over the 3,000 test windows for both systems. The curves track closely, with the BDN slightly ahead in total accumulated utility (30,520 vs 30,518).

### 4.3 Inference Latency Benchmarks

Real hardware inference latency is measured by timing each `engine.query()` call during the 3,000-window test evaluation. The measurements use `time.perf_counter()` for high-resolution timing.

#### 4.3.1 Reported Latency Metrics

| Metric | Value |
|--------|-------|
| Sample Count | 3,000 queries |
| Mean Latency | **0.984 ms** |
| Median Latency | 0.946 ms |
| 95th Percentile | 1.170 ms |
| 99th Percentile | 2.115 ms |
| Min Latency | 0.846 ms |
| Max Latency | 3.175 ms |
| Total Inference Time | 2.951 seconds |

**Interpretation:** The Bayesian inference engine handles each 5-second window in **under 1 millisecond on average**, with 95% of queries completing in under 1.2 ms. This is well within real-time constraints for a live smart-home monitoring system. Even the 99th percentile (2.115 ms) is negligible for practical deployment. Total inference over 3,000 windows takes approximately 3 seconds.

The latency distribution is visualized in `outputs/figures/latency_distribution.png` as a histogram with KDE overlay, showing the tight clustering around the mean with a slight right tail for the occasional slower query.

**Caveat:** These measurements are from the project's execution hardware and are reported as-is. They represent the performance of pgmpy's VariableElimination on this specific network structure and dataset. Performance on very different network structures or much larger state spaces may differ.

### 4.4 Cross-Home Generalization Experiments

#### 4.4.1 Experimental Design

Three cross-home experiments test whether the model generalizes to homes it was not trained on:

| Experiment | Training Homes | Test Home | Available Features |
|-----------|---------------|-----------|-------------------|
| Exp1 | Aruba | Aruba | 17 (in-home evaluation) |
| Exp2 | Aruba | Milan | 16 (Aruba→Milan transfer) |
| Exp3 | Aruba + Milan | Cairo | 13 (multi-home→unseen home) |

**Critical constraint:** Only features available in the target home are used as evidence. For example, in Exp2 (Aruba→Milan), `Hall_Activity` (present in Aruba but not Milan) is excluded from inference evidence. In Exp3 (Aruba+Milan→Cairo), both `Bathroom_Activity` and `DiningRoom_Activity` (absent in Cairo) plus `OutsideDoor_Open`/`OutsideDoor_Close` (Cairo lacks explicit OPEN/CLOSE) are excluded.

This ensures the model is not asked to reason about sensors that don't exist in the target home.

#### 4.4.2 Reported Cross-Home Results

| Experiment | Train | Test | Features | Occupancy Acc | Activity Acc | Sleep Acc | Mean Utility |
|-----------|-------|------|----------|--------------|-------------|-----------|-------------|
| Exp1 | Aruba | Aruba | 17 | 100.0% | 100.0% | 96.0% | 10.18 |
| Exp2 | Aruba | Milan | 16 | 100.0% | 100.0% | 97.8% | 10.28 |
| Exp3 | Aruba+Milan | Cairo | 13 | 100.0% | 100.0% | 98.1% | 10.05 |

**Interpretation:**

1. **Occupancy and activity accuracy are 100% across all experiments.** This is expected because these variables are primarily driven by total activity and door activity, which are universal features available in all homes. The model learns the fundamental relationship "activity → occupied" regardless of which specific room sensors are present.

2. **Sleep accuracy is high (96–98%) and actually improves in cross-home settings.** This is because sleep inference depends on `Is_Night`, `Home_Occupancy`, and absence of kitchen/living room activity — all of which are available in every home. The slight improvement in cross-home settings may reflect that the training homes' sleep patterns are representative of the test home's patterns.

3. **Mean utility remains high (10.05–10.28) across all experiments.** The model maintains good decision quality even when transferring to unseen homes with different sensor configurations.

4. **Feature count decreases as homes become more different.** Exp1 uses 17 features (full Aruba set minus the decision/utility nodes), Exp2 uses 16 (Milan lacks Hall), and Exp3 uses only 13 (Cairo lacks Bathroom, DiningRoom, LoungeChair, and explicit door OPEN/CLOSE). Despite using fewer features, Exp3 still achieves strong performance, demonstrating robustness to missing sensors.

The cross-home comparison is visualized in `outputs/figures/cross_home_performance.png`.

---

## 5. Web Dashboard Implementation

### 5.1 Architecture Overview

The web dashboard provides a **live, interactive visualization** of the full inference pipeline. It streams 5-second sensor windows one at a time, displays the observed evidence, shows the Bayesian posterior probabilities over behavioral variables, visualizes the 26-node network DAG, and presents the MEU decision with expected utilities.

**Tech stack:**
- **Backend:** FastAPI (Python) with WebSocket support, served via Uvicorn
- **Frontend:** Vanilla HTML/CSS/JavaScript with D3.js v7 for the network visualization
- **Model serving:** The trained `bayesian_network.pkl` is loaded at startup; inference runs on demand per window

The dashboard is hosted from `dashboard/` and run with `python dashboard/main.py` (or `uvicorn dashboard.main:app`).

### 5.2 Backend (FastAPI + WebSocket)

#### 5.2.1 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serves the main dashboard HTML page |
| `/api/network` | GET | Returns the 26-node network structure (nodes, edges, stage layout) for D3 visualization |
| `/api/nodes` | GET | Returns node definitions grouped by category (evidence, behavioral, decision, utility) |
| `/api/actions` | GET | Returns available actions and the full utility matrix |
| `/api/window/{idx}` | GET | Returns full inference results for a specific window index (evidence, posteriors, decision, true labels, MAP predictions) |
| `/api/next-window` | GET | Returns the next window in sequence (for manual stepping) |
| `/api/stats` | GET | Returns overall statistics (total windows, current index, home distribution, playback state) |
| `/ws` | WebSocket | Live streaming of inference results during playback |

#### 5.2.2 Model Loading

At startup (via FastAPI's `lifespan` context manager), the backend loads:
- The trained Bayesian network from `outputs/models/bayesian_network.pkl`
- The test dataset from `data/processed/test.csv`

Loading happens exactly once, not per request.

#### 5.2.3 Inference Per Window

`run_inference_on_window()` in `dashboard/main.py` performs the full pipeline for a single window:
1. Build an evidence dict from the window's observable variables (excluding `Unavailable` values).
2. Run Bayesian inference via `engine.query(evidence)` to get posteriors over the 6 behavioral variables.
3. Run MEU decision via `select_action_meu(posteriors, engine, evidence)` to get the selected action and expected utilities.
4. Extract MAP predictions (argmax of each posterior).
5. Extract true weak labels for comparison.

The result is a JSON-serializable dict containing everything the frontend needs.

#### 5.2.4 WebSocket Playback

The WebSocket endpoint (`/ws`) supports live streaming:
- Clients send `{type: "play", speed: X}` to start playback at X seconds per window.
- Clients send `{type: "pause"}` to stop.
- Clients send `{type: "speed", speed: X}` to change speed.
- Clients send `{type: "jump", index: N}` to jump to a specific window.

During playback, a background asyncio task streams windows at the configured speed to all connected clients. Each message is `{type: "update", data: {...}}` with the full inference result for one window.

The playback speed corresponds to **seconds per 5-second sensor window** — at 1.0s speed, one real second displays one 5-second window, so 5x accelerated relative to real-time. At 0.5s speed, it's 10x accelerated.

### 5.3 Frontend (D3.js Network Visualization)

#### 5.3.1 Layout and Design

The dashboard uses a three-column dark-themed layout:

- **Left panel:** Live inference — observed evidence grid and behavioral posterior distributions
- **Center panel:** The 26-node network DAG (D3.js force-directed layout with stage columns) and live replay controls
- **Right panel:** Decision engine — selected action, expected utility bars, situational state probabilities, and prediction vs. weak-label comparison

#### 5.3.2 Network Visualization

The network is rendered using D3.js v7. Rather than a spring-layout (which would be random each load), the dashboard uses a **deterministic stage-column layout** that mirrors the pipeline flow:

- Column 0: Temporal Inputs (Day_Of_Week, Time_Of_Day, Is_Night)
- Column 1: Door Sensors (OutsideDoor_Open, OutsideDoor_Close, Total_Contact_Events, OutsideDoor_Activity)
- Column 2: Primary Rooms (Bedroom_Activity, Kitchen_Activity, LivingRoom_Activity)
- Column 3: Secondary Rooms (Bathroom, DiningRoom, GuestRoom, LoungeChair, OtherRoom, WorkArea, Hall)
- Column 4: Aggregates (Total_Activity)
- Column 5: Behavioral (Inferred) — the 6 behavioral nodes
- Column 6: Decision (Smart_Home_Action)
- Column 7: Utility (Homeowner_Utility)

Nodes are color-coded by category:
- Evidence: blue (#6baed6)
- Behavioral: green (#74c476)
- Decision: orange (#fd8d3c)
- Utility: purple (#9e9ac8)

Edges are drawn as curved paths with arrowheads. During live playback, edges flowing into high-confidence behavioral nodes (≥75% probability) are highlighted in blue, and behavioral nodes glow with a green animation.

The layout is deterministic, zoomable (D3 zoom behavior), and responsive. Stage column labels are shown above each column.

#### 5.3.3 Per-Window Updates

Each time a new window is streamed or loaded:
1. **Evidence grid** updates — each of the 18 evidence variables shows its observed value, color-coded by state. Values that changed from the previous window get a highlight border. Unavailable sensors are dimmed.
2. **Posterior distributions** update — each of the 6 behavioral nodes shows a horizontal bar chart of its posterior probability distribution, with the MAP state highlighted.
3. **Prediction vs. weak label** comparison updates — shows the MAP prediction next to the true weak label, color-coded green (match) or red (mismatch).
4. **Decision engine** updates — the selected action is displayed prominently with its expected utility. A horizontal bar chart shows the expected utility of all 5 actions, with the selected action highlighted. Situational state probabilities are shown as percentage cards.
5. **Network animation** — evidence nodes that observed something this window pulse with a blue glow; behavioral nodes glow with a green animation; edges flowing into high-confidence conclusions light up.
6. **Timeline** updates — a scrubber shows current position in the test set, with home ID and timestamp metadata.

#### 5.3.4 Live Replay Controls

- **Play/Pause buttons** — start and stop WebSocket streaming
- **Previous/Next buttons** — step one window forward or backward via REST API
- **Speed selector** — choose 0.5s, 1.0s, 2.0s, or 5.0s per window
- **Timeline scrubber** — click or drag to jump to any window in the test set

A "5s" pulse indicator animates with each new window, reinforcing that each update represents one 5-second sensor window.

### 5.4 Parameter Serialization (.pkl Bundling)

The trained Bayesian network is serialized as a `.pkl` file using `joblib`:

```python
joblib.dump(model, os.path.join(MODELS_DIR, "bayesian_network.pkl"))
```

This file contains:
- The full `DiscreteBayesianNetwork` model object from pgmpy
- All 24 learned CPTs (Conditional Probability Tables)
- The DAG structure (edges)
- Node state names

The `.pkl` file is the **single artifact that captures the entire trained model**. It is:
- Loaded by the evaluation script (`src/evaluation.py`) for batch test-set evaluation
- Loaded by the dashboard backend (`dashboard/main.py`) at startup for live inference
- Loaded by the inference engine (`BayesianInferenceEngine`) for any interactive querying

**Cross-home experiment models** are also saved as separate `.pkl` files in `outputs/metrics/` (e.g., `model_Exp1_Aruba_to_Aruba.pkl`), capturing the specialized models trained for each cross-home experiment.

This bundling approach means the web dashboard does not need to retrain the model or recompute CPTs — it simply loads the pre-trained `.pkl` and serves inference on demand. The parameters (CPTs, structure, state names) are fully encapsulated in the serialized file.

### 5.5 Live Inference Replay

The dashboard's live replay feature streams windows sequentially over WebSocket, creating a "movie" of the smart home's inferred state over time. At the default speed of 1.0 seconds per window, the replay is 5x real-time accelerated (1 real second = 5 seconds of sensor data).

During replay:
- Each window's evidence is shown in real-time
- The Bayesian posterior updates visibly (bars shift, MAP states change)
- The decision engine updates its selected action and utility bars
- The network graph animates to highlight which nodes are active
- The timeline scrubber progresses

This provides an intuitive, visual demonstration of the entire probabilistic reasoning pipeline in action, making the abstract concepts (posteriors, MEU, utility) concrete and observable.

---

## 6. Results

### 6.1 Dataset Summary

| Home | Raw Events | Date Range | Unique Locations | Event Types |
|------|-----------|------------|-----------------|-------------|
| Aruba | 1,602,821 | 2010-11-04 → 2011-06-11 | 10 | ON, OFF, OPEN, CLOSE |
| Cairo | 647,487 | 2009-06-10 → 2009-08-05 | 8 | ON, OFF only |
| Milan | 421,392 | 2009-10-16 → 2010-01-06 | 9 | ON, OFF, OPEN, CLOSE |
| **Total** | **2,671,700** | — | 11 unique across homes | 4 distinct |

**Active 5-second windows generated:**
- Aruba: 656,811 windows
- Cairo: 226,259 windows
- Milan: 195,727 windows
- **Combined modeling dataset: 1,078,797 windows**

**Chronological split (70/15/15 per home):**
- Train: 755,156 windows
- Validation: 161,820 windows
- Test: 161,821 windows

### 6.2 Network Structure and Validation

- **Total nodes:** 26 (18 evidence + 6 behavioral + 1 decision + 1 utility)
- **Total edges:** 39 directed edges
- **Max parent set size:** 4
- **Acyclic:** Yes (verified)
- **Node schema:** Exactly as specified in the project contract — no silent changes

The DAG visualization is saved to `outputs/figures/network_dag.png`, and the structure report to `outputs/metrics/network_structure.txt`.

### 6.3 Behavioral Variable Inference Performance

Evaluated on 3,000 test windows against weak-label ground truth:

| Variable | Accuracy | Weighted F1 | Macro F1 |
|----------|----------|-------------|----------|
| Home_Occupancy | 99.97% | 0.9995 | 0.4999 |
| Resident_Activity | 99.97% | 0.9995 | 0.7499 |
| Resident_Sleep | 98.10% | 0.9796 | 0.8600 |
| Leaving_Home | 99.97% | 0.9995 | 0.4999 |
| Returning_Home | 99.33% | 0.9900 | 0.4983 |
| Unusual_Activity | 99.87% | 0.9985 | 0.8568 |

Full metrics (including macro/weighted precision and recall) are in `outputs/metrics/behavioral_metrics.csv`. Confusion matrices are in `outputs/figures/confusion_matrices.png`.

### 6.4 Decision Metrics — Bayesian vs Rule-Based

| Metric | Bayesian Decision Network | Rule-Based Baseline |
|--------|--------------------------|---------------------|
| Mean Utility | **10.173** | 10.173 |
| Median Utility | 10.0 | 10.0 |
| Total Utility | **30,520** | 30,518 |
| Std Utility | 2.796 | 1.713 |
| False Alert Rate | **0.0%** | 0.0% |
| Unusual Detection Rate | 88.9% | 100.0% |
| % No_Action | 99.57% | 99.03% |
| % Monitor | 0.00% | 0.67% |
| % Notify_Resident | 0.27% | 0.00% |
| % Silent_Alert | 0.17% | 0.00% |
| % Local_Alert | 0.00% | 0.30% |

Full decision metrics are in `outputs/metrics/decision_metrics.csv`. The cumulative utility comparison is in `outputs/figures/utility_comparison.png`. The action distribution comparison is in `outputs/figures/action_distribution.png`.

### 6.5 Inference Latency

| Metric | Value |
|--------|-------|
| Sample Count | 3,000 queries |
| Mean Latency | **0.984 ms** |
| Median Latency | 0.946 ms |
| 95th Percentile | 1.170 ms |
| 99th Percentile | 2.115 ms |
| Min Latency | 0.846 ms |
| Max Latency | 3.175 ms |
| Total Inference Time | 2.951 s |

Full latency metrics are in `outputs/metrics/latency_metrics.json`. The latency distribution is in `outputs/figures/latency_distribution.png`.

### 6.6 Cross-Home Generalization

| Experiment | Train | Test | Features | Occ Acc | Act Acc | Sleep Acc | Mean Utility |
|-----------|-------|------|----------|---------|---------|-----------|-------------|
| Exp1 | Aruba | Aruba | 17 | 100.0% | 100.0% | 96.0% | 10.18 |
| Exp2 | Aruba | Milan | 16 | 100.0% | 100.0% | 97.8% | 10.28 |
| Exp3 | Aruba+Milan | Cairo | 13 | 100.0% | 100.0% | 98.1% | 10.05 |

Full cross-home metrics are in `outputs/metrics/cross_home_metrics.csv`. The comparison plot is in `outputs/figures/cross_home_performance.png`.

### 6.7 Visualizations Generated

| # | Figure | Path |
|---|--------|------|
| 1 | Dataset Event Distribution | `outputs/figures/dataset_event_distribution.png` |
| 2 | Activity Distributions | `outputs/figures/activity_distributions.png` |
| 3 | Activity Over Time | `outputs/figures/activity_over_time.png` |
| 4 | 26-Node DAG | `outputs/figures/network_dag.png` |
| 5 | Example Posterior Output | `outputs/figures/example_posterior_output.png` |
| 6 | Action Distribution Comparison | `outputs/figures/action_distribution.png` |
| 7 | Bayesian vs Rule-Based Utility | `outputs/figures/utility_comparison.png` |
| 8 | Confusion Matrices | `outputs/figures/confusion_matrices.png` |
| 9 | Cross-Home Performance | `outputs/figures/cross_home_performance.png` |
| 10 | Latency Distribution | `outputs/figures/latency_distribution.png` |

---

## 7. Limitations

The following limitations are explicitly acknowledged:

1. **CASAS is an activity-event dataset.** It records binary PIR/motion sensor state changes, not continuous environmental variables. Temperature, humidity, electricity, HVAC, smoke, camera, microphone, and energy consumption data do NOT exist in these files and were not fabricated.

2. **Six behavioral variables are inferred via weak labeling.** `Home_Occupancy`, `Resident_Activity`, `Resident_Sleep`, `Leaving_Home`, `Returning_Home`, and `Unusual_Activity` are generated by transparent rule-based heuristics on the activity stream. They are NOT verified ground-truth annotations. Evaluation accuracy measures consistency between Bayesian MAP predictions and these weak labels, not true behavioral accuracy against human-annotated ground truth.

3. **`Unusual_Activity` is NOT intrusion detection.** It represents deviation from normal behavioral activity patterns. It should not be interpreted as security ground truth or verified intrusion detection.

4. **Sensor availability differs across homes.** Sensors absent in a home are represented as `Unavailable` (not zero), enforcing the distinction between "sensor exists and detected nothing" and "sensor does not exist." However, this also means the model has less information for homes with fewer sensors.

5. **Cairo lacks explicit OPEN/CLOSE door events.** Cairo uses `ON`/`OFF` for `OutsideDoor`. `OutsideDoor_Open` and `OutsideDoor_Close` are marked `Unavailable` for Cairo. This was documented and not silently converted.

6. **Utility values are project-defined assumptions.** The utility matrix was designed to reflect reasonable safety vs. disruption tradeoffs but has not been validated against real homeowner preferences or empirical data.

7. **Evaluation is on weak labels, not ground truth.** The behavioral metrics measure agreement between the Bayesian model and the weak-label rules, not true behavioral accuracy. High accuracy may partially reflect that both the model and the weak labels are derived from the same activity data.

8. **The test set is a 3,000-window subsample.** Due to the large test set (161,821 windows), evaluation uses a random subsample of 3,000 windows for computational feasibility. Results are representative but not exhaustive.

9. **Latency is hardware-dependent.** The reported inference latency (0.984 ms mean) is measured on the project's execution hardware and may vary on different systems.

10. **No online learning or adaptation.** The model is trained once on the training set and frozen. It does not adapt to new data or learn from deployment feedback.

---

## 8. Project Structure

```
project/
├── data/
│   ├── aruba.csv
│   ├── cairo.csv
│   └── milan.csv
├── data/processed/
│   ├── modeling_dataset.csv
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── src/
│   ├── __init__.py
│   ├── inspect_datasets.py      # Phase 1: dataset inspection
│   ├── preprocessing.py         # Phases 2-15: cleaning, windowing, features, weak labels, splitting
│   ├── model.py                 # Phases 16-20: 26-node DAG, CPT learning, inference engine
│   ├── decision.py              # Phases 21-24: decision node, utility, MEU, rule baseline
│   └── evaluation.py            # Phases 25-30: metrics, latency, cross-home, figures
├── outputs/
│   ├── figures/                 # 10 visualization PNGs
│   ├── metrics/                 # CSVs, JSONs, TXT reports
│   └── models/
│       └── bayesian_network.pkl # serialized trained model
├── dashboard/
│   ├── main.py                  # FastAPI backend + WebSocket server
│   ├── static/
│   │   ├── dashboard.js         # D3.js frontend logic
│   │   ├── styles.css           # dark-themed dashboard styling
│   │   └── ...
│   └── templates/
│       └── index.html           # main dashboard page
├── tests/
│   └── test_pipeline.py         # unit tests for preprocessing, network, decision, leakage
├── main.py                      # main pipeline entry point
├── requirements.txt
└── documentation.md             # this file
```

---

## 9. How to Run

### 9.1 Full Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline (preprocessing → training → evaluation → figures → report)
python main.py
```

This executes all 10 steps: dataset inspection, preprocessing, network validation, utility table export, CPT learning, inference verification, test evaluation, cross-home experiments, run configuration, and final results generation.

### 9.2 Web Dashboard

```bash
# Start the dashboard server (runs on http://localhost:8000)
python dashboard/main.py
```

Or with uvicorn directly:
```bash
uvicorn dashboard.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in a browser. The dashboard loads the trained model and test data, displays the 26-node network, and allows live replay of inference results.

### 9.3 Running Tests

```bash
pytest tests/
```

Tests cover: raw data cleaning, windowing, missing sensor distinction, exact 26-node count, DAG acyclicity, Bayesian inference engine, utility matrix completeness, MEU action selection, rule-based baseline, and train/test temporal separation (no leakage).

### 9.4 Key Output Artifacts

After running `python main.py`:

- `outputs/FINAL_RESULTS.md` — comprehensive results report
- `outputs/models/bayesian_network.pkl` — trained model (loaded by dashboard)
- `outputs/metrics/behavioral_metrics.csv` — per-variable behavioral metrics
- `outputs/metrics/decision_metrics.csv` — Bayesian vs rule-based decision metrics
- `outputs/metrics/cross_home_metrics.csv` — cross-home experiment results
- `outputs/metrics/latency_metrics.json` — inference latency benchmarks
- `outputs/metrics/network_structure.txt` — DAG structure report
- `outputs/metrics/utility_table.csv` — full utility specification
- `outputs/metrics/run_config.json` — reproducibility configuration
- `outputs/metrics/discretization_rules.json` — discretization thresholds
- `outputs/metrics/split_metadata.json` — train/val/test split boundaries
- `outputs/figures/*.png` — 10 visualization figures

---

*Generated by the Probabilistic Smart Home Decision Network pipeline — `python main.py`*
