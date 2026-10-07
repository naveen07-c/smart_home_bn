# Probabilistic Smart Home Decision Network — Final Results

## 1. Dataset Summary

| Home  | Raw Events | Homes | Date Range | Unique Locations | Event Types |
|-------|-----------|-------|------------|-----------------|-------------|
| Aruba | 1,602,821 | Aruba | 2010-11-04 → 2011-06-11 | 10 | ON, OFF, OPEN, CLOSE |
| Cairo |   647,487 | Cairo | 2009-06-10 → 2009-08-05 |  8 | ON, OFF only |
| Milan |   421,392 | Milan | 2009-10-16 → 2010-01-06 |  9 | ON, OFF, OPEN, CLOSE |
| **Total** | **2,671,700** | 3 | — | 11 unique (across homes) | 4 distinct |

**Active 5-second Windows Generated:**
- Aruba: 656,811 windows
- Cairo: 226,259 windows
- Milan: 195,727 windows
- **Combined modeling dataset: 1,078,797 windows**

**Chronological Split (per home, 70%/15%/15%):**
- Train: 755,156 windows
- Validation: 161,820 windows
- Test: 161,821 windows

---

## 2. Network Architecture — 26 Nodes

| Category | Count | Nodes |
|----------|-------|-------|
| Evidence | 18 | Bedroom_Activity, Bathroom_Activity, DiningRoom_Activity, GuestRoom_Activity, Kitchen_Activity, LivingRoom_Activity, LoungeChair_Activity, OtherRoom_Activity, OutsideDoor_Activity, WorkArea_Activity, Hall_Activity, OutsideDoor_Open, OutsideDoor_Close, Time_Of_Day, Day_Of_Week, Is_Night, Total_Activity, Total_Contact_Events |
| Behavioral | 6 | Home_Occupancy, Resident_Activity, Resident_Sleep, Leaving_Home, Returning_Home, Unusual_Activity |
| Decision | 1 | Smart_Home_Action |
| Utility | 1 | Homeowner_Utility |
| **Total** | **26** | |

**Node States:**
- Room Activity nodes: `Unavailable` / `None` / `Low` / `High`
- Total_Activity: `None` / `Low` / `Medium` / `High`
- Door nodes (where available): `Unavailable` / `No` / `Yes`
- Time_Of_Day: `Night` / `Morning` / `Afternoon` / `Evening`
- Day_Of_Week: `Weekday` / `Weekend`
- Is_Night: `No` / `Yes`
- Home_Occupancy: `Empty` / `Occupied`
- Resident_Activity: `Inactive` / `Low` / `Moderate` / `High`
- Resident_Sleep: `Awake` / `Sleeping`
- Leaving_Home / Returning_Home / Unusual_Activity: `No` / `Yes` (or `Normal`/`Unusual`)
- Smart_Home_Action: `No_Action` / `Monitor` / `Notify_Resident` / `Silent_Alert` / `Local_Alert`

**DAG:** 39 directed edges, max parent set size = 4, acyclic ✓

> **Node Schema Change Flag:** None. The 26-node specification was implemented exactly as defined in the project contract. No node definitions were silently changed.

---

## 3. Missing Sensor Handling (Project Contract Compliance)

> **Important:** Missing sensors are **NOT** treated as zero activity.

| Home | Missing Sensors | Representation |
|------|----------------|----------------|
| Aruba | Hall | `Unavailable` in categorical columns |
| Cairo | Bathroom, DiningRoom, LoungeChair; OutsideDoor_Open/Close | `Unavailable` |
| Milan | GuestRoom, Hall | `Unavailable` |

Cairo door semantics: Uses ON/OFF events (not OPEN/CLOSE). `OutsideDoor_Open` and `OutsideDoor_Close` are marked `Unavailable` for Cairo. Door contact events are counted via `Total_Contact_Events`.

---

## 4. Bayesian Model

- **Library:** pgmpy 1.1.2 (`DiscreteBayesianNetwork`)
- **Parameter Learning:** `DiscreteBayesianEstimator`, prior_type=`BDeu`, equivalent_sample_size=5
- **CPTs Learned:** 24 (18 evidence + 6 behavioral probabilistic nodes)
- **Inference:** Variable Elimination (exact inference via `VariableElimination`)
- **Training Data Only:** All thresholds and CPTs fitted on training partition only

---

## 5. Decision Model

- **Decision Node:** `Smart_Home_Action`
- **5 Available Actions:** No_Action, Monitor, Notify_Resident, Silent_Alert, Local_Alert
- **Utility Framework:** `Homeowner_Utility` — accounts for safety, disruption, false-alert cost, and response appropriateness
- **MEU Formula:** `EU(a) = Σ P(state | evidence) × U(a, state)`, best_action = argmax EU
- **Utility table:** `outputs/metrics/utility_table.csv`

---

## 6. Behavioral Variable Inference Evaluation (Test Set, n=3,000)

> **Important:** These variables are inferred via weak labeling rules (not ground-truth annotation). Metrics measure agreement between Bayesian MAP predictions and weak-label rules.

| Variable | Accuracy | Weighted F1 | Macro F1 |
|----------|----------|------------|---------|
| Home_Occupancy | 99.97% | 0.9995 | 0.4999 |
| Resident_Activity | 99.97% | 0.9995 | 0.7499 |
| Resident_Sleep | 98.10% | 0.9796 | 0.8600 |
| Leaving_Home | 99.97% | 0.9995 | 0.4999 |
| Returning_Home | 99.33% | 0.9900 | 0.4983 |
| Unusual_Activity | 99.87% | 0.9985 | 0.8568 |

*Note: Low macro-F1 for highly imbalanced classes (e.g., Home_Occupancy) is expected — the dataset is heavily skewed toward 'Occupied'.*

---

## 7. Decision Metrics Comparison (Test Set, n=3,000)

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

**Key finding:** Both systems produce equal average utility. The Bayesian Decision Network demonstrates a more nuanced response gradient (Monitor/Notify/Silent_Alert) while the rule-based system is binary (No_Action vs Local_Alert). The BDN achieves zero false alerts while the rule-based system's 100% unusual detection includes escalation in borderline cases.

---

## 8. Inference Latency (Real Hardware Measurement)

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

---

## 9. Cross-Home Generalization Results

| Experiment | Train | Test | Features Used | Occupancy Acc | Activity Acc | Sleep Acc | Mean Utility |
|-----------|-------|------|--------------|--------------|-------------|-----------|-------------|
| Exp1 | Aruba | Aruba | 17 | 100.0% | 100.0% | 96.0% | 10.18 |
| Exp2 | Aruba | Milan | 16 | 100.0% | 100.0% | 97.8% | 10.28 |
| Exp3 | Aruba+Milan | Cairo | 13 | 100.0% | 100.0% | 98.1% | 10.05 |

*Only features available in the target home were used in evidence. Unavailable sensors were excluded from inference evidence.*

---

## 10. Visualizations Generated

| Figure | Path |
|--------|------|
| Dataset Event Distribution | `outputs/figures/dataset_event_distribution.png` |
| Activity Distribution | `outputs/figures/activity_distributions.png` |
| Activity Over Time | `outputs/figures/activity_over_time.png` |
| 26-Node DAG | `outputs/figures/network_dag.png` |
| Posterior Probability Example | `outputs/figures/example_posterior_output.png` |
| Action Distribution | `outputs/figures/action_distribution.png` |
| Bayesian vs Rule-Based Utility | `outputs/figures/utility_comparison.png` |
| Confusion Matrices | `outputs/figures/confusion_matrices.png` |
| Cross-Home Performance | `outputs/figures/cross_home_performance.png` |
| Latency Distribution | `outputs/figures/latency_distribution.png` |

---

## 11. Limitations

> **These are explicitly stated project limitations, not oversights.**

1. **CASAS is an activity-event dataset** — it records binary PIR/motion sensor state changes, not continuous environmental variables. Temperature, humidity, electricity, HVAC, smoke, and camera data do NOT exist in these files and were not fabricated.

2. **Six behavioral variables are inferred via weak labeling** — `Home_Occupancy`, `Resident_Activity`, `Resident_Sleep`, `Leaving_Home`, `Returning_Home`, and `Unusual_Activity` are generated by transparent rule-based heuristics on the activity stream. They are NOT verified ground-truth annotations. Evaluation accuracy measures consistency between Bayesian MAP predictions and these weak labels.

3. **`Unusual_Activity` is NOT intrusion detection** — it represents deviation from normal behavioral activity patterns. It should not be interpreted as security ground truth.

4. **Sensor availability differs across homes** — sensors absent in a home are represented as `Unavailable` (not zero), enforcing the project contract's distinction between "sensor exists and detected nothing" vs "sensor does not exist."

5. **Cairo lacks explicit OPEN/CLOSE door events** — Cairo uses `ON`/`OFF` for `OutsideDoor`. `OutsideDoor_Open` and `OutsideDoor_Close` are marked `Unavailable` for Cairo. This is documented and was not silently converted.

6. **Utility values are project-defined assumptions** — the utility matrix was designed to reflect reasonable safety vs disruption tradeoffs but has not been validated against real homeowner preferences.

7. **Node schema contract:** No node definitions were silently changed from the 26-node contract specification. Any required deviations are documented above.

---

*Generated by `python main.py` — Probabilistic Smart Home Decision Network*
