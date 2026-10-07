# Pipeline Data Flow — Probabilistic Smart Home Decision Network

---

## 1. The 26 Nodes

### Evidence Nodes (18) — Observable / Input

| # | Node Name | Type | States |
|---|-----------|------|--------|
| 1 | `Bedroom_Activity` | Room activity count | Unavailable / None / Low / High |
| 2 | `Bathroom_Activity` | Room activity count | Unavailable / None / Low / High |
| 3 | `DiningRoom_Activity` | Room activity count | Unavailable / None / Low / High |
| 4 | `GuestRoom_Activity` | Room activity count | Unavailable / None / Low / High |
| 5 | `Kitchen_Activity` | Room activity count | Unavailable / None / Low / High |
| 6 | `LivingRoom_Activity` | Room activity count | Unavailable / None / Low / High |
| 7 | `LoungeChair_Activity` | Room activity count | Unavailable / None / Low / High |
| 8 | `OtherRoom_Activity` | Room activity count | Unavailable / None / Low / High |
| 9 | `OutsideDoor_Activity` | Room activity count | Unavailable / None / Low / High |
| 10 | `WorkArea_Activity` | Room activity count | Unavailable / None / Low / High |
| 11 | `Hall_Activity` | Room activity count | Unavailable / None / Low / High |
| 12 | `OutsideDoor_Open` | Door event flag | Unavailable / No / Yes |
| 13 | `OutsideDoor_Close` | Door event flag | Unavailable / No / Yes |
| 14 | `Time_Of_Day` | Temporal | Night / Morning / Afternoon / Evening |
| 15 | `Day_Of_Week` | Temporal | Weekday / Weekend |
| 16 | `Is_Night` | Temporal | No / Yes |
| 17 | `Total_Activity` | Aggregate | None / Low / Medium / High |
| 18 | `Total_Contact_Events` | Aggregate | No / Yes |

### Behavioral Nodes (6) — Inferred / Latent

| # | Node Name | Type | States |
|---|-----------|------|--------|
| 19 | `Home_Occupancy` | Inferred | Empty / Occupied |
| 20 | `Resident_Activity` | Inferred | Inactive / Low / Moderate / High |
| 21 | `Resident_Sleep` | Inferred | Awake / Sleeping |
| 22 | `Leaving_Home` | Inferred | No / Yes |
| 23 | `Returning_Home` | Inferred | No / Yes |
| 24 | `Unusual_Activity` | Inferred | Normal / Unusual |

### Decision Node (1)

| # | Node Name | Type | States |
|---|-----------|------|--------|
| 25 | `Smart_Home_Action` | Decision | No_Action / Monitor / Notify_Resident / Silent_Alert / Local_Alert |

### Utility Node (1)

| # | Node Name | Type | States |
|---|-----------|------|--------|
| 26 | `Homeowner_Utility` | Utility | Continuous scalar value |

**Total: 18 + 6 + 1 + 1 = 26 nodes.**

---

## 2. Data Flow — Step by Step

### STEP 1 — Raw Input (3 CSV files)

```
data/
├── aruba.csv   (1,602,821 rows)
├── cairo.csv    (647,487 rows)
└── milan.csv    (421,392 rows)
```

**Schema (no header):**
```
date, time, location, event
```

**Example row:**
```
2010-11-04, 22:33:01.100, Bedroom, ON
```

**Event types present:**
- Aruba: ON, OFF, OPEN, CLOSE
- Cairo: ON, OFF (no OPEN/CLOSE)
- Milan: ON, OFF, OPEN, CLOSE

---

### STEP 2 — Raw Event Cleaning (`src/preprocessing.py → load_and_clean_raw_data()`)

**Per home, per row:**

1. Read CSV with no header → assign columns `date`, `time`, `location`, `event`
2. Strip whitespace from `location` and `event`
3. Construct `timestamp = pd.to_datetime(date + " " + time, format="mixed", errors="coerce")`
4. Drop rows where `timestamp` is NaT (invalid)
5. Drop exact duplicates (same date, time, location, event)
6. Sort by `timestamp` ascending
7. Add `home_id` column: "Aruba", "Cairo", or "Milan"

**Output per home:** a chronologically sorted, deduplicated DataFrame with columns:
```
date, time, location, event, timestamp, home_id
```

---

### STEP 3 — 5-Second Tumbling Windowing (`src/preprocessing.py → aggregate_5s_windows()`)

For each home's cleaned events:

```
window_start = timestamp.floor("5s")
```

Each window = `[window_start, window_start + 5s)`.

Group all events by `(home_id, window_start)`.

**For each window, compute:**

1. **Room activity counts** — count of ON/OFF events per location within the window:
   ```
   Bedroom_Activity = count(Bedroom ON/OFF events in window)
   Kitchen_Activity = count(Kitchen ON/OFF events in window)
   ... (all 11 locations)
   ```

2. **Door flags** (Aruba and Milan only — have OPEN/CLOSE):
   ```
   OutsideDoor_Open  = 1 if OutsideDoor OPEN in window, else 0
   OutsideDoor_Close = 1 if OutsideDoor CLOSE in window, else 0
   ```
   For Cairo (no OPEN/CLOSE): set both to -1 (sentinel for unavailable)

3. **Total_Contact_Events:**
   - Aruba/Milan: `OutsideDoor_Open + OutsideDoor_Close`
   - Cairo: count of OutsideDoor ON/OFF events (Cairo's door events are ON/OFF, not OPEN/CLOSE)

4. **Temporal features** (from `window_start` timestamp):
   ```
   hour = window_start.hour
   dayofweek = window_start.dayofweek

   Time_Of_Day:
       Night      if hour >= 22 OR hour < 6
       Morning    if 6 <= hour < 12
       Afternoon  if 12 <= hour < 18
       Evening    if 18 <= hour < 22

   Day_Of_Week:
       Weekday  if dayofweek in [0,1,2,3,4]
       Weekend  if dayofweek in [5,6]

   Is_Night:
       Yes  if hour >= 22 OR hour < 6
       No   otherwise
   ```

5. **Total_Activity:**
   ```
   Total_Activity = sum of all room activity counts for sensors that exist in this home
   ```
   (Only native sensors — missing sensors are not counted as zero)

**Output:** one row per active window per home, with all numerical features.

**Missing sensor rule:** If a home doesn't have a sensor (e.g., Aruba has no Hall), the count is `-1` (not zero). This is the sentinel for "sensor does not exist."

**Active windows generated:**
- Aruba: 656,811
- Cairo: 226,259
- Milan: 195,727
- **Total: 1,078,797 windows**

---

### STEP 4 — Discretization (`src/preprocessing.py → discretize_evidence_features()`)

Convert all numerical features to categorical states.

**Room Activity (11 variables):**
| Raw Value | Categorical State |
|-----------|-------------------|
| -1 | `Unavailable` (sensor doesn't exist) |
| 0 | `None` (sensor exists, no events) |
| 1–2 | `Low` |
| >=3 | `High` |

**OutsideDoor_Open / OutsideDoor_Close:**
| Raw Value | Categorical State |
|-----------|-------------------|
| -1 | `Unavailable` (Cairo — no OPEN/CLOSE) |
| 0 | `No` |
| >0 | `Yes` |

**Total_Activity:**
| Raw Value | Categorical State |
|-----------|-------------------|
| 0 | `None` |
| 1–2 | `Low` |
| 3–5 | `Medium` |
| >=6 | `High` |

**Total_Contact_Events:**
| Raw Value | Categorical State |
|-----------|-------------------|
| 0 | `No` |
| >0 | `Yes` |

**Time_Of_Day, Day_Of_Week, Is_Night:** already categorical, no conversion needed.

**Output:** all 18 evidence variables are now categorical strings.

**Discretization rules saved to:** `outputs/metrics/discretization_rules.json`

---

### STEP 5 — Behavioral Weak Label Generation (`src/preprocessing.py → generate_behavioral_weak_labels()`)

For each home's chronologically sorted discretized windows, generate 6 behavioral variables using rule-based heuristics.

**Input to this step:** window-level discretized evidence (already has all 18 evidence variables).

**Computed features used by the rules:**
```
prev_gap  = seconds since previous window's start  (fillna 5s)
next_gap  = seconds until next window's start       (fillna 5s)
is_door_active = (OutsideDoor_Activity in [Low, High]) OR (Total_Contact_Events == "Yes")
is_night       = (Is_Night == "Yes")
bedroom_act    = Bedroom_Activity in [Low, High]
kitchen_act    = Kitchen_Activity in [Low, High]
living_act     = LivingRoom_Activity in [Low, High]
total_act      = Total_Activity
```

#### Rule 1 — Returning_Home
```
IF prev_gap >= 600 seconds (10 min)
   AND (is_door_active OR total_act in [Low, Medium, High])
THEN Returning_Home = "Yes"
ELSE Returning_Home = "No"
```
*Rationale: long absence followed by door activity or new indoor activity = resident returning.*

#### Rule 2 — Leaving_Home
```
IF is_door_active
   AND next_gap >= 600 seconds (10 min)
THEN Leaving_Home = "Yes"
ELSE Leaving_Home = "No"
```
*Rationale: door activity followed by prolonged silence = resident left.*

#### Rule 3 — Home_Occupancy
```
IF Leaving_Home == "Yes"
THEN Home_Occupancy = "Empty"
ELSE Home_Occupancy = "Occupied"
```
*Rationale: occupancy drops when a leaving event has just finalized.*

#### Rule 4 — Resident_Activity
```
IF Home_Occupancy == "Empty" OR total_act == "None"
THEN Resident_Activity = "Inactive"
ELIF total_act == "Low"
THEN Resident_Activity = "Low"
ELIF total_act == "Medium"
THEN Resident_Activity = "Moderate"
ELIF total_act == "High"
THEN Resident_Activity = "High"
```
*Rationale: activity level mirrors total activity, except when empty.*

#### Rule 5 — Resident_Sleep
```
IF is_night
   AND Home_Occupancy == "Occupied"
   AND NOT kitchen_act
   AND NOT living_act
   AND Resident_Activity in [Inactive, Low]
THEN Resident_Sleep = "Sleeping"
ELSE Resident_Sleep = "Awake"
```
*Rationale: night + occupied + no kitchen/living activity + low activity = likely sleeping.*

#### Rule 6 — Unusual_Activity
```
IF (is_night AND is_door_active)
   OR (is_night AND total_act == "High" AND NOT bedroom_act)
   OR (Home_Occupancy == "Empty" AND total_act in [Medium, High])
THEN Unusual_Activity = "Unusual"
ELSE Unusual_Activity = "Normal"
```
*Rationale: three anomaly patterns — nighttime door use, high non-bedroom activity at night, activity when home should be empty.*

**Output:** 6 behavioral variables added to each window row.

**Important:** These are **weak labels**, not ground truth. They are transparent rule-based inferences from the activity stream.

---

### STEP 6 — Assemble Modeling Dataset

**Columns in final modeling dataset (`data/processed/modeling_dataset.csv`):**
```
home_id, window_start,
Bedroom_Activity, Bathroom_Activity, DiningRoom_Activity, GuestRoom_Activity,
Kitchen_Activity, LivingRoom_Activity, LoungeChair_Activity, OtherRoom_Activity,
OutsideDoor_Activity, WorkArea_Activity, Hall_Activity,
OutsideDoor_Open, OutsideDoor_Close,
Time_Of_Day, Day_Of_Week, Is_Night,
Total_Activity, Total_Contact_Events,
Home_Occupancy, Resident_Activity, Resident_Sleep,
Leaving_Home, Returning_Home, Unusual_Activity
```

**24 columns total:** 18 evidence + 6 behavioral, plus `home_id` and `window_start`.

---

### STEP 7 — Chronological Train / Val / Test Split

Per home, sort by `window_start`, then split:

```
Training:   first 70%  of windows
Validation: next 15%   of windows
Test:       final 15%  of windows
```

**Sizes:**
- Train: 755,156 windows
- Validation: 161,820 windows
- Test: 161,821 windows

**Output files:**
```
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

**Split metadata saved to:** `outputs/metrics/split_metadata.json` (start/end timestamps per home per split).

**No temporal leakage:** max(train timestamps) ≤ min(val timestamps) ≤ min(test timestamps) for each home.

**Critical:** All subsequent steps use training data only for fitting. Test data is untouched until final evaluation.

---

### STEP 8 — Define the 26-Node DAG (`src/model.py`)

The DAG is defined as an explicit edge list (`DAG_EDGES`) in `src/model.py`. The graph is built with `networkx.DiGraph()`.

#### 8.1 Full Edge List (39 edges)

**Temporal layer:**
```
Day_Of_Week          → Time_Of_Day
Time_Of_Day          → Is_Night
```

**Door / contact layer:**
```
OutsideDoor_Open           → Total_Contact_Events
OutsideDoor_Close          → Total_Contact_Events
Total_Contact_Events       → OutsideDoor_Activity
```

**Activity aggregation layer:**
```
LivingRoom_Activity   → Total_Activity
Kitchen_Activity      → Total_Activity
Bedroom_Activity      → Total_Activity
```

**Occupancy layer:**
```
Total_Activity        → Home_Occupancy
OutsideDoor_Activity  → Home_Occupancy
```

**Resident activity layer:**
```
Home_Occupancy        → Resident_Activity
Total_Activity        → Resident_Activity
Time_Of_Day           → Resident_Activity
Resident_Activity     → Bathroom_Activity
Resident_Activity     → DiningRoom_Activity
Resident_Activity     → GuestRoom_Activity
Resident_Activity     → LoungeChair_Activity
Resident_Activity     → OtherRoom_Activity
Resident_Activity     → WorkArea_Activity
Resident_Activity     → Hall_Activity
```

**Sleep layer:**
```
Home_Occupancy        → Resident_Sleep
Is_Night              → Resident_Sleep
Bedroom_Activity      → Resident_Sleep
```

**Leaving / Returning layer:**
```
Home_Occupancy           → Leaving_Home
OutsideDoor_Activity     → Leaving_Home
Total_Contact_Events     → Leaving_Home
Home_Occupancy           → Returning_Home
OutsideDoor_Activity     → Returning_Home
Total_Contact_Events     → Returning_Home
```

**Unusual activity layer:**
```
Home_Occupancy           → Unusual_Activity
Resident_Activity        → Unusual_Activity
Is_Night                 → Unusual_Activity
OutsideDoor_Activity     → Unusual_Activity
```

**Decision layer (informational parents of the decision node):**
```
Unusual_Activity     → Smart_Home_Action
Home_Occupancy       → Smart_Home_Action
Resident_Sleep       → Smart_Home_Action
```

**Utility layer (parents of the utility node):**
```
Smart_Home_Action    → Homeowner_Utility
Unusual_Activity     → Homeowner_Utility
Home_Occupancy       → Homeowner_Utility
```

#### 8.2 Graph Properties

- Nodes: 26
- Edges: 39
- Max in-degree (max parents for any node): 4
- Acyclic: **Yes** (verified)

#### 8.3 Validation

`validate_network_structure()` checks:
1. Node count == 26
2. Every edge references valid nodes
3. Graph is acyclic (`nx.is_directed_acyclic_graph`)
4. Max parent set size ≤ 6

**Outputs:**
- `outputs/metrics/network_structure.txt` (full structure report)
- `outputs/figures/network_dag.png` (visualization)

---

### STEP 9 — CPT Learning (`src/model.py → train_bayesian_network()`)

**What is learned:** Conditional Probability Tables for the 24 probabilistic nodes (18 evidence + 6 behavioral). The decision node and utility node do NOT get CPTs — they are handled by the decision layer.

**Training data:** `data/processed/train.csv` (755,156 windows), using only the 24 probabilistic node columns.

**Method:**
```python
model = DiscreteBayesianNetwork(prob_edges)  # edges among the 24 probabilistic nodes only
estimator = DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=5)
model.fit(train_data.astype(str), estimator=estimator)
```

**BDeu prior with equivalent_sample_size=5:** adds a small pseudo-count to every parent→child state configuration, preventing zero-probability entries that would break inference. The pseudo-count is small enough that abundant data dominates, but large enough to keep every CPT entry strictly positive.

**What is saved:** `outputs/models/bayesian_network.pkl` (full `DiscreteBayesianNetwork` object with all 24 CPTs, via `joblib.dump`).

**CPTs learned:** 24 (one per probabilistic node, each specifying P(Node | Parents) for all parent-state combinations).

---

### STEP 10 — Bayesian Inference (`src/model.py → BayesianInferenceEngine`)

**Input:** An evidence dictionary mapping observable variable names to their observed categorical states for a single window.

**Example evidence for one window:**
```python
{
    "Bedroom_Activity": "Low",
    "Kitchen_Activity": "None",
    "LivingRoom_Activity": "None",
    "Is_Night": "Yes",
    "Total_Activity": "Low",
    "Time_Of_Day": "Night"
}
```

**Evidence filtering before inference:**
1. Only variables that exist in the model's node set are kept.
2. Values must be valid states in the node's CPD (otherwise dropped with a warning).
3. Target variables (the 6 behavioral nodes) are excluded from evidence — you cannot observe what you're inferring.
4. Variables with value `Unavailable` are excluded (marginalized out).

**Inference algorithm:** `VariableElimination` (exact inference from pgmpy). Eliminates non-query, non-evidence variables one at a time using the chain rule and the DAG factorization.

**Output:** Posterior probability distributions over the 6 behavioral variables:

```python
{
    "Home_Occupancy":     {"Empty": 0.08,  "Occupied": 0.92},
    "Resident_Activity":  {"Inactive": 0.05, "Low": 0.30, "Moderate": 0.45, "High": 0.20},
    "Resident_Sleep":     {"Awake": 0.35,  "Sleeping": 0.65},
    "Leaving_Home":       {"No": 0.97,     "Yes": 0.03},
    "Returning_Home":     {"No": 0.88,     "Yes": 0.12},
    "Unusual_Activity":   {"Normal": 0.94, "Unusual": 0.06}
}
```

Each distribution sums to 1.0.

**MAP prediction** (used for evaluation): the state with the highest posterior probability for each variable.

---

### STEP 11 — Situational State Probability Computation (`src/decision.py → compute_situational_state_probabilities()`)

**Input:** The 6 behavioral posteriors from Step 10, plus optionally the inference engine and evidence (for exact joint query).

**Output:** Probability distribution over 6 mutually exclusive situational states:

| Situational State | Meaning |
|-------------------|---------|
| `Normal_Awake` | Occupied + Awake + Normal |
| `Normal_Sleeping` | Occupied + Sleeping + Normal |
| `Normal_Empty` | Empty + Normal |
| `Unusual_Occupied_Awake` | Occupied + Awake + Unusual |
| `Unusual_Occupied_Sleeping` | Occupied + Sleeping + Unusual |
| `Unusual_Empty` | Empty + Unusual |

**Computation (exact, when engine provided):**
The joint distribution over `(Home_Occupancy, Resident_Sleep, Unusual_Activity)` is queried from the BN. The 8 joint combinations are mapped to the 6 situational states:

```
Normal_Awake:         P(Occupied, Awake, Normal)
Normal_Sleeping:      P(Occupied, Sleeping, Normal)
Normal_Empty:         P(Empty, Awake, Normal) + P(Empty, Sleeping, Normal)
Unusual_Occupied_Awake:       P(Occupied, Awake, Unusual)
Unusual_Occupied_Sleeping:    P(Occupied, Sleeping, Unusual)
Unusual_Empty:        P(Empty, Awake, Unusual) + P(Empty, Sleeping, Unusual)
```

**Fallback (approximate, when no engine):** independence assumption from marginal posteriors:
```
P(Normal) = 1 - P(Unusual)
P(Occupied) = posterior["Home_Occupancy"]["Occupied"]
P(Sleeping) = posterior["Resident_Sleep"]["Sleeping"]
P(Awake) = 1 - P(Sleeping)

Normal_Awake = P(Normal) * P(Occupied) * P(Awake)
Normal_Sleeping = P(Normal) * P(Occupied) * P(Sleeping)
Normal_Empty = P(Normal) * (1 - P(Occupied))
Unusual_Occupied_Awake = P(Unusual) * P(Occupied) * P(Awake)
Unusual_Occupied_Sleeping = P(Unusual) * P(Occupied) * P(Sleeping)
Unusual_Empty = P(Unusual) * (1 - P(Occupied))
```
Then normalized to sum to 1.

---

### STEP 12 — Maximum Expected Utility Decision (`src/decision.py → select_action_meu()`)

**Step 12a — Utility Matrix**

The utility function `Homeowner_Utility` is a matrix `U(action, situational_state)`:

| State \ Action | No_Action | Monitor | Notify_Resident | Silent_Alert | Local_Alert |
|----------------|-----------|---------|-----------------|--------------|-------------|
| `Normal_Awake` | 10.0 | 8.0 | -5.0 | -10.0 | -50.0 |
| `Normal_Sleeping` | 15.0 | 10.0 | -25.0 | -15.0 | -80.0 |
| `Normal_Empty` | 10.0 | 12.0 | -10.0 | -20.0 | -60.0 |
| `Unusual_Occupied_Awake` | -30.0 | 10.0 | 30.0 | 15.0 | -10.0 |
| `Unusual_Occupied_Sleeping` | -100.0 | -20.0 | 20.0 | 35.0 | 45.0 |
| `Unusual_Empty` | -120.0 | -30.0 | 30.0 | 55.0 | 45.0 |

**Full utility table saved to:** `outputs/metrics/utility_table.csv`

**Step 12b — Expected Utility Calculation**

For each action `a`:
```
EU(a) = Σ_{s in situational_states} P(s | evidence) × U(a, s)
```

**Step 12c — Action Selection**
```
best_action = argmax_{a in {No_Action, Monitor, Notify_Resident, Silent_Alert, Local_Alert}} EU(a)
```

**Output dict:**
```python
{
    "selected_action": "Silent_Alert",          # the best action
    "expected_utilities": {
        "No_Action": -30.0,
        "Monitor": 10.0,
        "Notify_Resident": 30.0,
        "Silent_Alert": 35.0,
        "Local_Alert": 25.0
    },
    "state_probabilities": {
        "Normal_Awake": 0.02,
        "Normal_Sleeping": 0.03,
        "Normal_Empty": 0.01,
        "Unusual_Occupied_Awake": 0.35,
        "Unusual_Occupied_Sleeping": 0.55,
        "Unusual_Empty": 0.04
    }
}
```

---

### STEP 13 — Final Decision Output

For a single 5-second window, the complete decision output is:

```
{
    "window_id": 12345,
    "timestamp": "2010-11-04 22:33:00",
    "home_id": "Aruba",

    "evidence": {
        "Bedroom_Activity": "Low",
        "Kitchen_Activity": "None",
        "LivingRoom_Activity": "None",
        "Is_Night": "Yes",
        "Total_Activity": "Low",
        "Time_Of_Day": "Night",
        ... (all observed evidence variables)
    },

    "posteriors": {
        "Home_Occupancy":     {"Empty": 0.08,  "Occupied": 0.92},
        "Resident_Activity":  {"Inactive": 0.05, "Low": 0.30, "Moderate": 0.45, "High": 0.20},
        "Resident_Sleep":     {"Awake": 0.35,  "Sleeping": 0.65},
        "Leaving_Home":       {"No": 0.97,     "Yes": 0.03},
        "Returning_Home":     {"No": 0.88,     "Yes": 0.12},
        "Unusual_Activity":   {"Normal": 0.94, "Unusual": 0.06}
    },

    "decision": {
        "selected_action": "No_Action",
        "expected_utilities": {
            "No_Action": 10.0,
            "Monitor": 8.0,
            "Notify_Resident": -5.0,
            "Silent_Alert": -10.0,
            "Local_Alert": -50.0
        },
        "state_probabilities": {
            "Normal_Awake": 0.60,
            "Normal_Sleeping": 0.25,
            "Normal_Empty": 0.08,
            "Unusual_Occupied_Awake": 0.04,
            "Unusual_Occupied_Sleeping": 0.02,
            "Unusual_Empty": 0.01
        }
    },

    "true_labels": {
        "Home_Occupancy": "Occupied",
        "Resident_Activity": "Low",
        "Resident_Sleep": "Sleeping",
        "Leaving_Home": "No",
        "Returning_Home": "No",
        "Unusual_Activity": "Normal"
    },

    "predictions": {
        "Home_Occupancy": "Occupied",
        "Resident_Activity": "Moderate",
        "Resident_Sleep": "Sleeping",
        "Leaving_Home": "No",
        "Returning_Home": "No",
        "Unusual_Activity": "Normal"
    }
}
```

**This is the output for one window.** The dashboard streams this object per window over WebSocket; the evaluation script computes metrics by comparing `predictions` vs `true_labels` across 3,000 test windows.

---

## 3. Full Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RAW INPUT                                         │
│  data/aruba.csv (1,602,821 rows)                                       │
│  data/cairo.csv  (647,487 rows)                                        │
│  data/milan.csv  (421,392 rows)                                        │
│  Schema: date, time, location, event (no header)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: RAW EVENT CLEANING                                            │
│  src/preprocessing.py → load_and_clean_raw_data()                      │
│                                                                          │
│  • Assign columns (no header)                                          │
│  • Strip whitespace                                                    │
│  • Parse timestamp (date + time)                                       │
│  • Drop invalid timestamps                                             │
│  • Drop exact duplicates                                               │
│  • Sort chronologically                                                │
│  • Add home_id                                                        │
│                                                                          │
│  Output per home: DataFrame with date, time, location, event,         │
│                    timestamp, home_id                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: 5-SECOND WINDOWING                                            │
│  src/preprocessing.py → aggregate_5s_windows()                        │
│                                                                          │
│  • window_start = timestamp.floor("5s")                                │
│  • Group events by (home_id, window_start)                             │
│  • Count ON/OFF events per room location → 11 room activity counts    │
│  • Count door OPEN/CLOSE events (Aruba/Milan) or ON/OFF (Cairo)      │
│  • Derive Time_Of_Day, Day_Of_Week, Is_Night from window_start       │
│  • Compute Total_Activity (sum of native room counts)                 │
│  • Compute Total_Contact_Events                                       │
│  • Missing sensors → -1 sentinel (NOT zero)                           │
│                                                                          │
│  Output: 1,078,797 active windows with numerical features             │
│  • Aruba: 656,811 windows                                             │
│  • Cairo: 226,259 windows                                             │
│  • Milan: 195,727 windows                                             │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: DISCRETIZATION                                                │
│  src/preprocessing.py → discretize_evidence_features()                │
│                                                                          │
│  Room Activity:  -1→Unavailable, 0→None, 1-2→Low, >=3→High          │
│  Door Open/Close: -1→Unavailable, 0→No, >0→Yes                       │
│  Total_Activity: 0→None, 1-2→Low, 3-5→Medium, >=6→High              │
│  Total_Contact:   0→No, >0→Yes                                       │
│                                                                          │
│  Output: 18 evidence variables are now categorical                    │
│  Rules saved to: outputs/metrics/discretization_rules.json            │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 5: BEHAVIORAL WEAK LABELS                                       │
│  src/preprocessing.py → generate_behavioral_weak_labels()             │
│                                                                          │
│  Input: discretized evidence (18 vars) + chronological order          │
│                                                                          │
│  Rules (per window):                                                   │
│  • Returning_Home: long gap (>=600s) + door/activity → Yes           │
│  • Leaving_Home: door activity + next gap >=600s → Yes               │
│  • Home_Occupancy: Empty if Leaving_Home==Yes, else Occupied         │
│  • Resident_Activity: based on occupancy + Total_Activity level       │
│  • Resident_Sleep: Night + Occupied + no kitchen/living act           │
│                      + low activity → Sleeping                        │
│  • Unusual_Activity: night door use OR high non-bedroom night act     │
│                       OR activity when Empty → Unusual                │
│                                                                          │
│  Output: 6 behavioral variables added (weak labels, NOT ground truth) │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 6: ASSEMBLE MODELING DATASET                                    │
│                                                                          │
│  Columns: home_id, window_start, [18 evidence], [6 behavioral]       │
│  Saved to: data/processed/modeling_dataset.csv                        │
│  Shape: (1,078,797 rows, 26 columns)                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 7: CHRONOLOGICAL SPLIT (per home)                               │
│                                                                          │
│  Training:   first 70%  → 755,156 windows → data/processed/train.csv │
│  Validation: next 15%   → 161,820 windows → data/processed/val.csv   │
│  Test:       final 15%  → 161,821 windows → data/processed/test.csv  │
│                                                                          │
│  No temporal leakage (verified by unit test).                         │
│  Split metadata → outputs/metrics/split_metadata.json                 │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 8: DEFINE 26-NODE DAG                                           │
│  src/model.py → build_full_dag()                                      │
│                                                                          │
│  26 nodes, 39 directed edges, max parent set = 4, acyclic            │
│  Nodes 1-18: Evidence (observable)                                    │
│  Nodes 19-24: Behavioral (inferred)                                   │
│  Node 25: Smart_Home_Action (decision)                                │
│  Node 26: Homeowner_Utility (utility)                                 │
│                                                                          │
│  Validated by: validate_network_structure()                           │
│  Output: outputs/metrics/network_structure.txt                        │
│          outputs/figures/network_dag.png                               │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 9: CPT LEARNING (on TRAINING data only)                         │
│  src/model.py → train_bayesian_network()                              │
│                                                                          │
│  • Model: pgmpy DiscreteBayesianNetwork                               │
│  • Estimator: DiscreteBayesianEstimator, prior_type=BDeu,             │
│               equivalent_sample_size=5                                │
│  • Learns: 24 CPTs (18 evidence + 6 behavioral nodes)                │
│  • Decision node and utility node do NOT get CPTs                    │
│                                                                          │
│  Output: outputs/models/bayesian_network.pkl (joblib serialization)   │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 10: BAYESIAN INFERENCE (per window)                             │
│  src/model.py → BayesianInferenceEngine.query()                       │
│                                                                          │
│  Input: evidence dict (observed evidence vars → state values)         │
│  • Filter: drop Unavailable, drop unknown vars, drop target vars      │
│  • Algorithm: VariableElimination (exact inference)                   │
│                                                                          │
│  Output: posterior distributions over 6 behavioral variables          │
│  {Home_Occupancy: {Empty: p, Occupied: 1-p},                        │
│   Resident_Activity: {Inactive: p1, Low: p2, ...},                   │
│   Resident_Sleep: {Awake: p, Sleeping: 1-p},                         │
│   Leaving_Home: {No: p, Yes: 1-p},                                   │
│   Returning_Home: {No: p, Yes: 1-p},                                 │
│   Unusual_Activity: {Normal: p, Unusual: 1-p}}                       │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 11: SITUATIONAL STATE PROBABILITIES                             │
│  src/decision.py → compute_situational_state_probabilities()          │
│                                                                          │
│  Collapses 6 behavioral posteriors into 6 mutually exclusive states:  │
│  • Normal_Awake, Normal_Sleeping, Normal_Empty                       │
│  • Unusual_Occupied_Awake, Unusual_Occupied_Sleeping, Unusual_Empty │
│                                                                          │
│  Either via exact joint query from BN, or approximate independence    │
│  fallback. Normalized to sum to 1.0.                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 12: MAXIMUM EXPECTED UTILITY (MEU) DECISION                     │
│  src/decision.py → select_action_meu()                                │
│                                                                          │
│  • Utility matrix U(action, situational_state): 5 actions × 6 states │
│  • EU(a) = Σ P(state | evidence) × U(a, state)  for each action a    │
│  • best_action = argmax EU(a)                                         │
│  • Returns: selected_action, expected_utilities, state_probabilities  │
│                                                                          │
│  Output: decision dict with selected action + EU for all 5 actions    │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 13: FINAL PER-WINDOW DECISION OUTPUT                            │
│                                                                          │
│  Complete output for one 5-second window:                             │
│  • window_id, timestamp, home_id                                     │
│  • evidence: observed evidence variables and their states            │
│  • posteriors: P(behavioral_var = state | evidence) for 6 vars       │
│  • decision: selected_action, EU for all 5 actions,                  │
│              situational state probabilities                          │
│  • true_labels: weak-label ground truth for 6 behavioral vars        │
│  • predictions: MAP prediction for 6 behavioral vars                 │
│                                                                          │
│  This is what the dashboard streams per window over WebSocket,        │
│  and what the evaluation script uses to compute all metrics.          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Decision Output Schema (Single Window)

```json
{
  "window_id": 12345,
  "timestamp": "2010-11-04 22:33:00",
  "home_id": "Aruba",
  "evidence": {
    "Bedroom_Activity": "Low",
    "Kitchen_Activity": "None",
    "LivingRoom_Activity": "None",
    "Is_Night": "Yes",
    "Total_Activity": "Low",
    "Time_Of_Day": "Night",
    "OutsideDoor_Activity": "None",
    "OutsideDoor_Open": "No",
    "OutsideDoor_Close": "No",
    "Total_Contact_Events": "No",
    "Day_Of_Week": "Weekday",
    "Bathroom_Activity": "Unavailable",
    "DiningRoom_Activity": "None",
    "GuestRoom_Activity": "None",
    "LoungeChair_Activity": "None",
    "OtherRoom_Activity": "None",
    "WorkArea_Activity": "None",
    "Hall_Activity": "Unavailable"
  },
  "posteriors": {
    "Home_Occupancy": {"Empty": 0.08, "Occupied": 0.92},
    "Resident_Activity": {"Inactive": 0.05, "Low": 0.30, "Moderate": 0.45, "High": 0.20},
    "Resident_Sleep": {"Awake": 0.35, "Sleeping": 0.65},
    "Leaving_Home": {"No": 0.97, "Yes": 0.03},
    "Returning_Home": {"No": 0.88, "Yes": 0.12},
    "Unusual_Activity": {"Normal": 0.94, "Unusual": 0.06}
  },
  "decision": {
    "selected_action": "No_Action",
    "expected_utilities": {
      "No_Action": 10.0,
      "Monitor": 8.0,
      "Notify_Resident": -5.0,
      "Silent_Alert": -10.0,
      "Local_Alert": -50.0
    },
    "state_probabilities": {
      "Normal_Awake": 0.60,
      "Normal_Sleeping": 0.25,
      "Normal_Empty": 0.08,
      "Unusual_Occupied_Awake": 0.04,
      "Unusual_Occupied_Sleeping": 0.02,
      "Unusual_Empty": 0.01
    }
  },
  "true_labels": {
    "Home_Occupancy": "Occupied",
    "Resident_Activity": "Low",
    "Resident_Sleep": "Sleeping",
    "Leaving_Home": "No",
    "Returning_Home": "No",
    "Unusual_Activity": "Normal"
  },
  "predictions": {
    "Home_Occupancy": "Occupied",
    "Resident_Activity": "Moderate",
    "Resident_Sleep": "Sleeping",
    "Leaving_Home": "No",
    "Returning_Home": "No",
    "Unusual_Activity": "Normal"
  }
}
```

---

## 5. Batch Evaluation Flow (Test Set)

```
For each of 3,000 test windows:
    1. Build evidence dict from window (drop Unavailable)
    2. Time engine.query(evidence) → posteriors  (latency measurement)
    3. MAP prediction = argmax of each posterior
    4. select_action_meu(posteriors, engine, evidence) → decision
    5. select_rule_based_action(observation) → rule_action
    6. evaluate_action_realized_utility(action, true_labels) → utility
    7. Accumulate: y_true, y_pred, bn_actions, rule_actions, bn_utilities, rule_utilities, latencies

After all 3,000 windows:
    → Behavioral metrics (accuracy, precision, recall, F1 per variable)
    → Decision metrics (mean/median/total/utility std, false alert rate,
                        unusual detection rate, action distributions)
    → Latency metrics (mean, median, p95, p99, min, max, total time)
    → Generate 10 figures
    → Save CSVs and JSON to outputs/metrics/
```

---

## 6. Cross-Home Experiment Flow

```
Experiment 1: Train Aruba → Test Aruba (17 features available)
Experiment 2: Train Aruba → Test Milan (16 features, no Hall)
Experiment 3: Train Aruba+Milan → Test Cairo (13 features, no Bathroom/
              DiningRoom/LoungeChair, no explicit OPEN/CLOSE)

For each experiment:
    1. Filter train/test to relevant homes
    2. Determine target home's available sensors
    3. Train new model on train subset (only available features as evidence)
    4. For each test window:
       - Build evidence using ONLY target-home-available features
       - Run inference, MEU decision, utility evaluation
    5. Report: occupancy accuracy, activity accuracy, sleep accuracy, mean utility
```

---

*Pipeline data flow — Probabilistic Smart Home Decision Network*
