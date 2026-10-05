"""
Comprehensive unit test suite for the Probabilistic Smart Home Decision Network.
Covers Phase 32 requirements:
- Preprocessing (timestamp parsing, invalid rows, duplicate handling, window assignment)
- Feature Extraction (room counts, door features, temporal features, aggregates)
- Network Architecture (26 nodes, DAG validity, acyclicity, expected node names)
- Decision Model (EU calculation, MEU action selection, utility table)
- Evaluation & Data Leakage (train/validation/test chronological separation)
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
import networkx as nx

from src.preprocessing import (
    load_and_clean_raw_data, aggregate_5s_windows,
    discretize_evidence_features, generate_behavioral_weak_labels,
    ALL_ROOM_LOCATIONS, HOME_SENSORS
)
from src.model import (
    EVIDENCE_NODES, BEHAVIORAL_NODES, DECISION_NODES, UTILITY_NODES,
    ALL_NODES, DAG_EDGES, build_full_dag, BayesianInferenceEngine
)
from src.decision import (
    ACTIONS, UTILITY_MATRIX, SITUATIONAL_STATES,
    select_action_meu, select_rule_based_action,
    evaluate_action_realized_utility, compute_situational_state_probabilities
)


# --- 1. Preprocessing Tests ---

def test_raw_data_cleaning():
    """Validates raw event cleaning, whitespace trimming, and duplicate dropping."""
    df_clean = load_and_clean_raw_data("milan")
    assert not df_clean.empty
    assert "timestamp" in df_clean.columns
    assert df_clean["timestamp"].isnull().sum() == 0
    # Chronological ordering check
    assert df_clean["timestamp"].is_monotonic_increasing
    # Exact duplicate check
    assert df_clean.duplicated(subset=["date", "time", "location", "event"]).sum() == 0


def test_windowing_and_feature_counts():
    """Validates tumbling 5s window assignment and numerical room activity features."""
    sample_data = pd.DataFrame({
        "date": ["2010-11-04", "2010-11-04", "2010-11-04"],
        "time": ["00:00:01.100", "00:00:03.200", "00:00:07.500"],
        "location": ["Bedroom", "Bedroom", "Kitchen"],
        "event": ["ON", "OFF", "ON"]
    })
    sample_data["timestamp"] = pd.to_datetime(sample_data["date"] + " " + sample_data["time"])
    
    feats = aggregate_5s_windows(sample_data, "Aruba")
    # Windows should be 00:00:00 and 00:00:05
    assert len(feats) == 2
    assert "Bedroom_Activity" in feats.columns
    assert "Kitchen_Activity" in feats.columns
    assert "Total_Activity" in feats.columns
    
    row0 = feats.iloc[0]
    assert row0["Bedroom_Activity"] == 2
    assert row0["Kitchen_Activity"] == 0
    assert row0["Total_Activity"] == 2


def test_missing_sensor_distinction():
    """Verifies that sensors absent in a home are marked as unavailable (-1/Unavailable)."""
    sample_data = pd.DataFrame({
        "date": ["2009-06-10"],
        "time": ["01:00:01.000"],
        "location": ["Bedroom"],
        "event": ["ON"]
    })
    sample_data["timestamp"] = pd.to_datetime(sample_data["date"] + " " + sample_data["time"])
    
    # Cairo does not have Bathroom or DiningRoom
    feats_cairo = aggregate_5s_windows(sample_data, "Cairo")
    assert feats_cairo["Bathroom_Activity"].iloc[0] == -1
    assert feats_cairo["OutsideDoor_Open"].iloc[0] == -1
    
    disc = discretize_evidence_features(feats_cairo)
    assert disc["Bathroom_Activity"].iloc[0] == "Unavailable"
    assert disc["OutsideDoor_Open"].iloc[0] == "Unavailable"


# --- 2. Network Architecture Tests ---

def test_exact_26_nodes():
    """Validates that network contains exactly 26 nodes as specified in the contract."""
    assert len(EVIDENCE_NODES) == 18, f"Expected 18 evidence nodes, got {len(EVIDENCE_NODES)}"
    assert len(BEHAVIORAL_NODES) == 6, f"Expected 6 behavioral nodes, got {len(BEHAVIORAL_NODES)}"
    assert len(DECISION_NODES) == 1, f"Expected 1 decision node, got {len(DECISION_NODES)}"
    assert len(UTILITY_NODES) == 1, f"Expected 1 utility node, got {len(UTILITY_NODES)}"
    assert len(ALL_NODES) == 26, f"Expected 26 total nodes, got {len(ALL_NODES)}"
    assert len(set(ALL_NODES)) == 26, "Duplicate node names detected in node schema!"


def test_dag_acyclicity_and_validity():
    """Validates that graph is a valid Directed Acyclic Graph (DAG) with valid node references."""
    G = build_full_dag()
    assert len(G.nodes) == 26
    assert nx.is_directed_acyclic_graph(G), "The 26-node graph must be acyclic!"
    for u, v in G.edges:
        assert u in ALL_NODES, f"Edge source '{u}' not in declared nodes"
        assert v in ALL_NODES, f"Edge target '{v}' not in declared nodes"


def test_bayesian_inference_engine():
    """Validates that BayesianInferenceEngine produces valid probability distributions."""
    engine = BayesianInferenceEngine()
    test_ev = {
        "Bedroom_Activity": "Low",
        "Kitchen_Activity": "None",
        "Total_Activity": "Low",
        "Is_Night": "Yes"
    }
    res = engine.query(test_ev)
    assert set(res.keys()) == set(BEHAVIORAL_NODES)
    for var, dist in res.items():
        assert isinstance(dist, dict)
        prob_sum = sum(dist.values())
        assert pytest.approx(prob_sum, 0.01) == 1.0


# --- 3. Decision Model Tests ---

def test_utility_matrix_completeness():
    """Validates that all situational states and action alternatives exist in utility table."""
    assert len(ACTIONS) == 5
    for state in SITUATIONAL_STATES:
        assert state in UTILITY_MATRIX
        for action in ACTIONS:
            assert action in UTILITY_MATRIX[state]
            assert isinstance(UTILITY_MATRIX[state][action], (int, float))


def test_meu_action_selection():
    """Validates MEU action selection and expected utility calculation."""
    # Peaceful night scenario: high sleep prob, zero unusual prob
    peaceful_posteriors = {
        "Home_Occupancy": {"Occupied": 0.99, "Empty": 0.01},
        "Resident_Sleep": {"Sleeping": 0.95, "Awake": 0.05},
        "Unusual_Activity": {"Normal": 0.999, "Unusual": 0.001}
    }
    decision = select_action_meu(peaceful_posteriors)
    assert decision["selected_action"] == "No_Action"
    assert decision["expected_utilities"]["No_Action"] > decision["expected_utilities"]["Local_Alert"]
    
    # Intrusion / Anomaly scenario in empty home:
    danger_posteriors = {
        "Home_Occupancy": {"Occupied": 0.05, "Empty": 0.95},
        "Resident_Sleep": {"Awake": 0.95, "Sleeping": 0.05},
        "Unusual_Activity": {"Normal": 0.10, "Unusual": 0.90}
    }
    danger_decision = select_action_meu(danger_posteriors)
    assert danger_decision["selected_action"] in ["Silent_Alert", "Local_Alert"]


def test_rule_based_baseline():
    """Validates rule-based baseline logic on test inputs."""
    obs_unusual = {"Unusual_Activity": "Unusual", "Returning_Home": "No", "Leaving_Home": "No"}
    assert select_rule_based_action(obs_unusual) == "Local_Alert"
    
    obs_returning = {"Unusual_Activity": "Normal", "Returning_Home": "Yes", "Leaving_Home": "No"}
    assert select_rule_based_action(obs_returning) == "Monitor"
    
    obs_normal = {"Unusual_Activity": "Normal", "Returning_Home": "No", "Leaving_Home": "No"}
    assert select_rule_based_action(obs_normal) == "No_Action"


# --- 4. Train / Test Separation & Leakage Tests ---

def test_chronological_split_no_leakage():
    """Validates that train, validation, and test datasets do not overlap in timestamps."""
    train_df = pd.read_csv("data/processed/train.csv", keep_default_na=False)
    val_df = pd.read_csv("data/processed/validation.csv", keep_default_na=False)
    test_df = pd.read_csv("data/processed/test.csv", keep_default_na=False)
    
    for home in ["Aruba", "Cairo", "Milan"]:
        tr_h = train_df[train_df["home_id"] == home]
        va_h = val_df[val_df["home_id"] == home]
        te_h = test_df[test_df["home_id"] == home]
        
        max_tr = pd.to_datetime(tr_h["window_start"]).max()
        min_va = pd.to_datetime(va_h["window_start"]).min()
        max_va = pd.to_datetime(va_h["window_start"]).max()
        min_te = pd.to_datetime(te_h["window_start"]).min()
        
        # Strict chronological precedence
        assert max_tr <= min_va, f"Train-Val temporal leakage detected for {home}!"
        assert max_va <= min_te, f"Val-Test temporal leakage detected for {home}!"
