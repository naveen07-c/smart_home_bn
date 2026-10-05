# Probabilistic Smart Home Decision Network
# Complete Agentic Implementation Plan

## 0. PROJECT OBJECTIVE

Build a complete semester-level Probabilistic Smart Home Decision Network using the CASAS Aruba, Cairo, and Milan datasets.

The system must:

1. Load and validate the three CASAS CSV datasets.
2. Preprocess the event streams.
3. Convert events into 5-second windows.
4. Construct 18 observable/evidence variables.
5. Construct 6 behavioral/inferred variables.
6. Build a 26-node Bayesian/Decision Network.
7. Learn Bayesian CPT parameters from training data.
8. Perform probabilistic inference.
9. Implement a decision node and utility function.
10. Select actions using Maximum Expected Utility (MEU).
11. Compare against a rule-based baseline.
12. Evaluate on unseen test data.
13. Perform cross-home generalization experiments.
14. Produce reproducible metrics, plots, tables, and documentation.

The final system must be executable from a single entry point.

---

# 1. IMPORTANT PROJECT RULES

## 1.1 Do not invent dataset variables

The raw datasets contain event-level smart-home data.

Expected raw schema:

    date
    time
    location
    event

However, the coding agent MUST inspect the actual files before assuming anything.

Do not claim that CASAS contains:

- temperature
- humidity
- electricity
- HVAC
- smoke
- camera
- microphone
- energy consumption
- intrusion labels

unless those variables actually exist in the files.

They do not currently appear in the inspected CASAS files.

---

## 1.2 Preserve the three homes

Every processed record must contain:

    home_id

with:

    Aruba
    Cairo
    Milan

Do not merge the datasets in a way that loses home identity.

---

## 1.3 Handle missing room types correctly

The three homes do not have exactly the same room sensors.

For example:

- Aruba contains GuestRoom.
- Cairo contains Hall.
- Milan does not contain GuestRoom.
- Cairo does not have the same explicit OPEN/CLOSE door events as Aruba/Milan.

A missing sensor must NOT automatically become numerical zero.

Distinguish:

    sensor exists and detected nothing

from:

    sensor does not exist in this home.

Document the chosen representation.

---

## 1.4 Do not fabricate Cairo OPEN/CLOSE events

Aruba and Milan contain:

    OutsideDoor OPEN
    OutsideDoor CLOSE

Cairo uses:

    OutsideDoor ON
    OutsideDoor OFF

Do NOT silently convert Cairo ON/OFF into OPEN/CLOSE.

The common schema must explicitly account for this difference.

---

# 2. INITIAL PROJECT STRUCTURE

The project currently contains:

    data/
        aruba.csv
        cairo.csv
        milan.csv

Create the following structure:

    project/
    │
    ├── data/
    │   ├── aruba.csv
    │   ├── cairo.csv
    │   └── milan.csv
    │
    ├── data/processed/
    │
    ├── src/
    │   ├── preprocessing.py
    │   ├── model.py
    │   ├── decision.py
    │   └── evaluation.py
    │
    ├── outputs/
    │   ├── figures/
    │   ├── metrics/
    │   └── models/
    │
    ├── tests/
    │
    ├── main.py
    ├── requirements.txt
    └── README.md

Do not create unnecessary notebooks or dozens of files.

The architecture should remain minimal.

---

# 3. PHASE 1 — DATASET INSPECTION

Before implementing preprocessing, inspect:

    data/aruba.csv
    data/cairo.csv
    data/milan.csv

For each dataset calculate and report:

- number of rows
- number of columns
- column names
- data types
- missing values
- duplicate rows
- unique locations
- unique events
- timestamp range
- events per location
- events per event type

Create:

    outputs/metrics/dataset_summary.csv

Also create:

    outputs/metrics/location_summary.csv

and:

    outputs/metrics/event_summary.csv

The agent must verify that the three datasets actually follow the expected schema.

If a schema differs, adapt preprocessing instead of crashing or silently assuming equivalence.

---

# 4. PHASE 2 — GENERIC PREPROCESSOR

Implement:

    src/preprocessing.py

Create a generic CASAS preprocessing pipeline.

Do NOT create three unrelated preprocessors.

Use one common implementation with:

    home_id

as metadata.

Recommended pipeline:

    load_data()
        ↓
    parse_time()
        ↓
    clean_data()
        ↓
    classify_events()
        ↓
    classify_locations()
        ↓
    validate_data()
        ↓
    create_windows()
        ↓
    create_features()
        ↓
    create_behavioral_labels()

---

# 5. PHASE 3 — RAW EVENT CLEANING

For every file:

1. Read without assuming a header if the dataset has no header.
2. Assign:

       date
       time
       location
       event

3. Construct:

       timestamp

   from date + time.

4. Remove rows with invalid timestamps.
5. Strip whitespace from location/event values.
6. Sort chronologically.
7. Remove exact duplicate records.

Do NOT remove multiple legitimate events merely because they occur in the same 5-second window.

Example:

    22:33:01 Bedroom ON
    22:33:02 Bedroom OFF
    22:33:03 Bedroom OFF

These are separate events.

---

# 6. PHASE 4 — 5-SECOND WINDOWING

Use non-overlapping/tumbling windows:

    window_start = timestamp.floor("5s")

Interpret each window as:

    [window_start, window_start + 5 seconds)

Do not use sliding windows unless explicitly required later.

Create:

    window_start
    home_id

for every feature record.

Important:

A window containing multiple events is still one modeling instance.

---

# 7. PHASE 5 — EVIDENCE NODE SCHEMA

The final network must contain these 18 evidence variables.

## Activity variables

1. Bedroom_Activity
2. Bathroom_Activity
3. DiningRoom_Activity
4. GuestRoom_Activity
5. Kitchen_Activity
6. LivingRoom_Activity
7. LoungeChair_Activity
8. OtherRoom_Activity
9. OutsideDoor_Activity
10. WorkArea_Activity
11. Hall_Activity

## Door variables

12. OutsideDoor_Open
13. OutsideDoor_Close

## Temporal variables

14. Time_Of_Day
15. Day_Of_Week
16. Is_Night

## Aggregate variables

17. Total_Activity
18. Total_Contact_Events

These are the observable/evidence layer.

---

# 8. PHASE 6 — ACTIVITY FEATURE CONSTRUCTION

For each location, count activity events within the 5-second window.

For ON/OFF activity:

    activity_count = number of ON/OFF events

Do not interpret an ON/OFF event as occupancy by itself.

The resulting feature is an activity intensity measure.

For each location create:

    <Location>_Activity

For example:

    Bedroom_Activity
    Kitchen_Activity
    LivingRoom_Activity

etc.

---

# 9. PHASE 7 — DOOR FEATURES

For homes with explicit OPEN/CLOSE events:

    OutsideDoor_Open
    OutsideDoor_Close

should represent whether those events occurred in the window.

For Cairo:

    OutsideDoor ON/OFF

must remain a separate dataset-specific representation.

Do not fabricate OPEN/CLOSE semantics.

For the common model, decide and document a principled handling strategy.

Possible strategy:

- retain common `OutsideDoor_Activity`
- retain OPEN/CLOSE variables where semantically available
- mark unsupported variables as unavailable rather than zero

The final modeling pipeline must document this choice.

---

# 10. PHASE 8 — TEMPORAL FEATURES

Derive:

    Time_Of_Day
    Day_Of_Week
    Is_Night

Recommended Time_Of_Day states:

    Night
    Morning
    Afternoon
    Evening

Recommended Day_Of_Week states:

    Weekday
    Weekend

Recommended Is_Night states:

    No
    Yes

Use a consistent definition throughout the entire project.

For example:

    Night = hour >= 22 OR hour < 6

Do not change these thresholds after model training.

---

# 11. PHASE 9 — AGGREGATE FEATURES

Create:

    Total_Activity

as the sum of valid room activity counts within the window.

Create:

    Total_Contact_Events

from contact/door events.

Do not count unavailable sensors as observed zero activity.

Document exactly how aggregates are calculated when a home has a different sensor configuration.

---

# 12. PHASE 10 — FEATURE DISTRIBUTION ANALYSIS

Before choosing categorical states, inspect distributions.

For every numerical feature calculate:

    min
    max
    mean
    median
    25th percentile
    50th percentile
    75th percentile
    90th percentile
    95th percentile
    99th percentile

Generate histograms for major activity variables.

Do not arbitrarily select thresholds before inspecting the actual distributions.

Save:

    outputs/figures/activity_distributions.png

and related plots.

---

# 13. PHASE 11 — DISCRETIZATION

Bayesian-network variables should use manageable categorical states.

Recommended initial state scheme:

## Room activity

    None
    Low
    High

## Total activity

    None
    Low
    Medium
    High

## Door events

    No
    Yes

## Time

    Night
    Morning
    Afternoon
    Evening

## Day

    Weekday
    Weekend

The agent may modify thresholds after distribution analysis.

Every threshold must be documented.

Save the final thresholds in:

    outputs/metrics/discretization_rules.json

---

# 14. PHASE 12 — SIX BEHAVIORAL VARIABLES

Create these six inferred/behavioral variables:

19. Home_Occupancy
20. Resident_Activity
21. Resident_Sleep
22. Leaving_Home
23. Returning_Home
24. Unusual_Activity

These are NOT raw CASAS columns.

The implementation must clearly distinguish:

    observed variables

from:

    derived/weakly labeled variables

---

# 15. PHASE 13 — WEAK LABEL GENERATION

Because CASAS does not provide direct ground-truth labels for all six behavioral states, implement transparent weak-label rules.

Do not claim these are ground truth.

## Home_Occupancy

Use recent activity evidence.

Example principle:

    sufficient recent activity → Occupied
    prolonged inactivity → Empty

The inactivity threshold must be determined from the data and documented.

Possible states:

    Empty
    Occupied

---

## Resident_Activity

Use:

    Total_Activity
    room activity
    Home_Occupancy
    Time_Of_Day

States:

    Inactive
    Low
    Moderate
    High

---

## Resident_Sleep

Use:

    Is_Night
    Bedroom_Activity
    Total_Activity
    Home_Occupancy

States:

    Awake
    Sleeping

This is an inferred behavioral label, not a medical/sleep ground truth.

---

## Leaving_Home

Use temporal transition patterns involving:

    OutsideDoor activity
    OutsideDoor OPEN/CLOSE where available
    activity before event
    activity after event
    occupancy state

States:

    No
    Yes

---

## Returning_Home

Use:

    door activity
    activity after door event
    occupancy transition

States:

    No
    Yes

---

## Unusual_Activity

Do NOT call this ground-truth intrusion.

It represents behavioral deviation/anomaly.

Possible states:

    Normal
    Unusual

Use a transparent method based on deviation from normal activity patterns.

The method must be documented.

---

# 16. PHASE 14 — MODELING DATASET

Create one processed modeling dataset.

Recommended:

    data/processed/modeling_dataset.csv

It should contain:

    home_id
    window_start

followed by all 24 observable + behavioral variables.

The decision and utility variables are not ordinary evidence features and should be generated by the decision layer.

---

# 17. PHASE 15 — TRAIN / VALIDATION / TEST SPLIT

Do NOT randomly split individual windows.

Adjacent 5-second windows are temporally correlated.

Use chronological splitting.

For each home:

    first 70%  → train
    next 15%   → validation
    final 15%  → test

Preserve chronological order.

Save split metadata.

Recommended files:

    train.csv
    validation.csv
    test.csv

inside:

    data/processed/

Ensure preprocessing decisions such as thresholds are fitted using training data only where appropriate.

---

# 18. PHASE 16 — FINAL 26-NODE NETWORK

The complete network contains:

## Evidence — 18

    Bedroom_Activity
    Bathroom_Activity
    DiningRoom_Activity
    GuestRoom_Activity
    Kitchen_Activity
    LivingRoom_Activity
    LoungeChair_Activity
    OtherRoom_Activity
    OutsideDoor_Activity
    WorkArea_Activity
    Hall_Activity
    OutsideDoor_Open
    OutsideDoor_Close
    Time_Of_Day
    Day_Of_Week
    Is_Night
    Total_Activity
    Total_Contact_Events

## Behavioral — 6

    Home_Occupancy
    Resident_Activity
    Resident_Sleep
    Leaving_Home
    Returning_Home
    Unusual_Activity

## Decision — 1

    Smart_Home_Action

## Utility — 1

    Homeowner_Utility

TOTAL:

    18 + 6 + 1 + 1 = 26

---

# 19. PHASE 17 — DAG DESIGN

Implement the Bayesian network in:

    src/model.py

Do not connect every node to every other node.

Use a sparse, semantically meaningful DAG.

Recommended conceptual dependencies:

## Occupancy

    room activity variables
            ↓
    Total_Activity
            ↓
    Home_Occupancy

Relevant room activity variables may also directly influence occupancy.

---

## Resident Activity

    Home_Occupancy
    Total_Activity
    Time_Of_Day
    major room activities
            ↓
    Resident_Activity

---

## Sleep

    Home_Occupancy
    Is_Night
    Time_Of_Day
    Bedroom_Activity
    Total_Activity
            ↓
    Resident_Sleep

---

## Leaving

    Home_Occupancy
    OutsideDoor_Activity
    OutsideDoor_Open
    OutsideDoor_Close
    Total_Activity
            ↓
    Leaving_Home

---

## Returning

    OutsideDoor_Activity
    OutsideDoor_Open
    OutsideDoor_Close
    Total_Activity
    Home_Occupancy
            ↓
    Returning_Home

---

## Unusual Activity

    Time_Of_Day
    Home_Occupancy
    Resident_Activity
    Leaving_Home
    Returning_Home
    relevant activity variables
            ↓
    Unusual_Activity

---

# 20. PHASE 18 — GRAPH VALIDATION

Before learning parameters:

1. Check every node exists.
2. Check every edge references valid nodes.
3. Check the graph is acyclic.
4. Print all nodes.
5. Print all edges.
6. Check that every non-root node has intended parents.
7. Check for accidental enormous parent sets.

Generate:

    outputs/metrics/network_structure.txt

Also generate a graph visualization:

    outputs/figures/network_dag.png

---

# 21. PHASE 19 — CPT LEARNING

Use training data only.

For every Bayesian node:

    P(Node | Parents)

must be learned from the training set.

Use a robust Bayesian parameter estimator with smoothing, such as a Dirichlet/Bayesian estimator where supported.

Avoid raw maximum-frequency estimates that create zero-probability combinations.

Save the trained model under:

    outputs/models/

The exact serialization format should be chosen based on the selected BN library.

---

# 22. PHASE 20 — BAYESIAN INFERENCE

Implement inference for evidence windows.

Given evidence such as:

    Bedroom_Activity = Low
    Kitchen_Activity = High
    LivingRoom_Activity = Medium
    Is_Night = Yes
    Total_Activity = Medium

infer:

    P(Home_Occupancy)
    P(Resident_Activity)
    P(Resident_Sleep)
    P(Leaving_Home)
    P(Returning_Home)
    P(Unusual_Activity)

The implementation must expose posterior probabilities.

Example output:

    Home_Occupancy:
        Empty      0.08
        Occupied  0.92

    Resident_Sleep:
        Awake      0.35
        Sleeping   0.65

---

# 23. PHASE 21 — DECISION NODE

Implement:

    Smart_Home_Action

Possible states:

    No_Action
    Monitor
    Notify_Resident
    Silent_Alert
    Local_Alert

The action must be chosen from expected utility rather than a simple maximum-probability state.

---

# 24. PHASE 22 — UTILITY FUNCTION

Implement:

    Homeowner_Utility

Utility should account for:

- safety
- unnecessary alerts
- resident disruption
- intervention cost
- response appropriateness

Use a transparent utility table.

Do not hide utility values inside arbitrary code.

Store the utility specification in:

    outputs/metrics/utility_table.csv

Example:

    state,action,utility

The actual numerical values should be justified and documented.

---

# 25. PHASE 23 — MAXIMUM EXPECTED UTILITY

For each possible action:

    EU(action) = Σ P(state | evidence) × U(action,state)

Select:

    best_action = argmax(EU(action))

The implementation must return:

    selected action
    expected utility of each action
    posterior state probabilities

Example:

    No_Action      = 5.1
    Monitor        = 8.7
    Notify         = 11.3
    Silent_Alert   = 13.2
    Local_Alert    = 9.8

Selected:

    Silent_Alert

---

# 26. PHASE 24 — RULE-BASED BASELINE

Create a simple rule-based baseline.

Example:

    IF Unusual_Activity = Unusual
        → Local_Alert

    ELSE IF Returning_Home = Yes
        → Monitor

    ELSE
        → No_Action

The baseline must use the same test observations.

This provides a meaningful comparison:

    Rule-based system
            VS
    Bayesian Decision Network

---

# 27. PHASE 25 — EVALUATION METRICS

Evaluate the behavioral inference layer.

Calculate:

    Accuracy
    Precision
    Recall
    F1-score
    Confusion Matrix

For:

    Home_Occupancy
    Resident_Activity
    Resident_Sleep
    Leaving_Home
    Returning_Home
    Unusual_Activity

Remember these are evaluated against weak labels unless genuine labels are available.

Clearly state this limitation.

---

# 28. PHASE 26 — DECISION METRICS

Calculate:

    Average Utility
    Median Utility
    Total Utility
    Action frequency
    False-alert rate
    Unnecessary intervention rate
    Unusual-activity detection rate

Compare:

    Bayesian Decision Network

against:

    Rule-based baseline

The key result should be whether the probabilistic decision network produces better expected utility and/or fewer inappropriate actions.

---

# 29. PHASE 27 — INFERENCE PERFORMANCE

Measure:

    average inference latency
    median latency
    95th percentile latency
    total test-set inference time

Do not make an unrealistic latency claim.

Report actual measurements from the user's hardware.

---

# 30. PHASE 28 — CROSS-HOME GENERALIZATION

Perform multiple experiments.

## Experiment 1

    Train: Aruba
    Test: Aruba

## Experiment 2

    Train: Aruba
    Test: Milan

## Experiment 3

    Train: Aruba + Milan
    Test: Cairo

Because sensor configurations differ, evaluation must use only variables supported by the target home.

Do not fabricate unsupported variables.

Report the exact feature availability for each experiment.

---

# 31. PHASE 29 — FINAL TEST

The test set must remain untouched until all modeling decisions are finalized.

Final pipeline:

    Training
        ↓
    CPT learning
        ↓
    Validation
        ↓
    Architecture/threshold tuning
        ↓
    Freeze model
        ↓
    Test once
        ↓
    Final metrics

Do not repeatedly tune using test results.

---

# 32. PHASE 30 — VISUALIZATIONS

Generate at least:

1. Dataset event distribution
2. Activity distribution
3. Activity over time
4. 26-node DAG
5. Example posterior probability output
6. Action distribution
7. Bayesian vs rule-based utility comparison
8. Confusion matrix for key behavioral variables
9. Cross-home performance comparison
10. Inference latency distribution

Save all figures in:

    outputs/figures/

---

# 33. PHASE 31 — FINAL RESULTS

Create:

    outputs/FINAL_RESULTS.md

Include:

## Dataset

- number of events
- number of homes
- date ranges
- locations
- event types

## Network

- 26 nodes
- node categories
- states
- number of edges

## Model

- BN library
- parameter-learning method
- inference algorithm

## Decision model

- available actions
- utility definition
- MEU method

## Evaluation

- behavioral metrics
- decision metrics
- utility comparison
- baseline comparison
- cross-home results
- latency

## Limitations

Explicitly state:

- CASAS is an activity-event dataset.
- Six behavioral variables are inferred/weakly labeled.
- Intrusion is NOT ground-truth intrusion detection.
- Missing sensor types differ between homes.
- Utility values are project-defined assumptions.

---

# 34. PHASE 32 — TESTING

Create unit tests for:

## Preprocessing

- timestamp parsing
- invalid rows
- duplicate handling
- window assignment
- event classification
- location handling

## Feature extraction

- room activity counts
- door features
- temporal features
- aggregate features

## Network

- node count = 26
- expected node names
- DAG validity
- no cycles
- expected edges

## Decision

- expected utility calculation
- MEU action selection

## Evaluation

- metrics run without errors
- train/test separation
- no test leakage

Run:

    pytest

before final execution.

---

# 35. PHASE 33 — REPRODUCIBILITY

Use deterministic random seeds wherever randomness is used.

Record:

    Python version
    package versions
    dataset statistics
    preprocessing parameters
    discretization thresholds
    train/validation/test boundaries
    model configuration
    utility table

Generate:

    outputs/metrics/run_config.json

---

# 36. PHASE 34 — MAIN EXECUTION

Create:

    main.py

The preferred execution should be:

    python main.py

It should execute the complete pipeline:

    dataset inspection
        ↓
    preprocessing
        ↓
    feature generation
        ↓
    weak-label generation
        ↓
    train/validation/test split
        ↓
    network construction
        ↓
    CPT learning
        ↓
    validation
        ↓
    decision model
        ↓
    test evaluation
        ↓
    plots
        ↓
    final results

Do not require manual notebook execution.

---

# 37. PHASE 35 — FAILURE HANDLING

The agent must stop and report clearly if:

- CSV schema changes unexpectedly
- timestamps cannot be parsed
- required columns are missing
- feature distributions are pathological
- the DAG contains a cycle
- CPT learning fails
- unsupported variables are accidentally fabricated
- train/test leakage is detected

Do not silently continue with incorrect data.

---

# 38. PHASE 36 — DOCUMENTATION

Update:

    README.md

with:

1. Project objective
2. Dataset description
3. Installation
4. Project structure
5. Preprocessing
6. 26-node architecture
7. Bayesian model
8. Decision model
9. Utility/MEU
10. Evaluation
11. How to run
12. Results
13. Limitations

---

# 39. REQUIRED FINAL DELIVERABLES

At completion the project must contain:

    data/
        raw CSV files

    data/processed/
        processed features
        train.csv
        validation.csv
        test.csv

    src/
        preprocessing.py
        model.py
        decision.py
        evaluation.py

    outputs/
        figures/
        metrics/
        models/

    tests/

    main.py
    requirements.txt
    README.md

    outputs/FINAL_RESULTS.md

---

# 40. FINAL ACCEPTANCE CRITERIA

The project is considered complete only when ALL are true:

[ ] All three CSV files are successfully processed.

[ ] Dataset statistics are generated.

[ ] 5-second windowing works.

[ ] 18 evidence variables are generated.

[ ] Six behavioral variables are generated with documented rules.

[ ] Final network contains exactly 26 nodes.

[ ] DAG is acyclic.

[ ] CPTs are learned from training data.

[ ] Bayesian inference works on unseen windows.

[ ] Decision node contains five actions.

[ ] Utility function is implemented.

[ ] MEU selects the best action.

[ ] Rule-based baseline works.

[ ] Test set is never used for tuning.

[ ] Behavioral metrics are calculated.

[ ] Decision metrics are calculated.

[ ] Utility comparison is calculated.

[ ] Cross-home testing is performed.

[ ] Inference latency is measured.

[ ] Figures are generated.

[ ] Final results are documented.

[ ] Unit tests pass.

[ ] `python main.py` runs the complete pipeline.

---

# 41. AGENT EXECUTION RULE

Do NOT implement the entire project blindly in one pass.

Execute phase-by-phase.

After each major phase:

1. Run the relevant code.
2. Inspect the output.
3. Validate assumptions.
4. Fix errors.
5. Save intermediate results.
6. Only then proceed.

Priority order:

    Data correctness
        >
    Feature correctness
        >
    Network correctness
        >
    CPT learning
        >
    Decision model
        >
    Evaluation
        >
    Visualization

Never sacrifice data correctness merely to make the pipeline execute.

---

# 42. FINAL PROJECT PIPELINE

The final architecture must be:

                    Aruba.csv
                        │
                    Cairo.csv
                        │
                    Milan.csv
                        │
                        ▼
                 DATA VALIDATION
                        │
                        ▼
                  PREPROCESSING
                        │
                        ▼
                  5-SEC WINDOWS
                        │
                        ▼
              18 EVIDENCE VARIABLES
                        │
                        ▼
                 DISCRETIZATION
                        │
                        ▼
              6 BEHAVIORAL STATES
                        │
                        ▼
                 26-NODE DAG
                        │
                        ▼
                  CPT LEARNING
                        │
                        ▼
              BAYESIAN INFERENCE
                        │
                        ▼
                DECISION NODE
                        │
                        ▼
                UTILITY FUNCTION
                        │
                        ▼
                       MEU
                        │
                        ▼
                  BEST ACTION
                        │
                        ▼
               TEST / BASELINE
                        │
                        ▼
                FINAL RESULTS
