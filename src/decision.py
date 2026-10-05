"""
Decision Engine and Maximum Expected Utility (MEU) Framework.
Covers Phases 21 - 24:
- Decision node: Smart_Home_Action (5 actions)
- Utility function: Homeowner_Utility accounting for safety, disruption, and false alarms
- Utility table export to outputs/metrics/utility_table.csv
- Maximum Expected Utility (MEU) action selection
- Rule-based comparative baseline
"""

import os
import pandas as pd
import numpy as np

METRICS_DIR = "outputs/metrics"

# Phase 21: Five Action Alternatives
ACTIONS = [
    "No_Action",
    "Monitor",
    "Notify_Resident",
    "Silent_Alert",
    "Local_Alert"
]

# Primary situational states characterizing the home environment
SITUATIONAL_STATES = [
    "Normal_Awake",
    "Normal_Sleeping",
    "Normal_Empty",
    "Unusual_Occupied_Awake",
    "Unusual_Occupied_Sleeping",
    "Unusual_Empty"
]

# Phase 22: Utility Specification Matrix U(action, state)
# Safety vs Disruption vs Cost vs False-Alert tradeoffs
UTILITY_MATRIX = {
    "Normal_Awake": {
        "No_Action": 10.0,         # Optimal: zero disruption, peaceful
        "Monitor": 8.0,            # Passive check: low overhead
        "Notify_Resident": -5.0,    # Unnecessary phone notification
        "Silent_Alert": -10.0,      # False alert to emergency contacts
        "Local_Alert": -50.0       # Terrible: nuisance alarm when everything is normal
    },
    "Normal_Sleeping": {
        "No_Action": 15.0,         # Maximum value: undisturbed rest
        "Monitor": 10.0,           # Silent background logging
        "Notify_Resident": -25.0,  # Wakes up resident unnecessarily
        "Silent_Alert": -15.0,     # False alert
        "Local_Alert": -80.0      # Extreme disruption: sounding siren during peaceful sleep
    },
    "Normal_Empty": {
        "No_Action": 10.0,         # Quiet empty house
        "Monitor": 12.0,           # Periodic health/sensor monitoring is desirable
        "Notify_Resident": -10.0,  # Spurious alert to owner away from home
        "Silent_Alert": -20.0,     # False alert
        "Local_Alert": -60.0      # False siren blaring with no occupants
    },
    "Unusual_Occupied_Awake": {
        "No_Action": -30.0,        # Negligent: failing to inform occupant
        "Monitor": 10.0,           # Helpful to track developments
        "Notify_Resident": 30.0,   # Ideal: occupant is awake, can verify notification
        "Silent_Alert": 15.0,      # Acceptable backup
        "Local_Alert": -10.0       # Overreaction while occupant is awake and in control
    },
    "Unusual_Occupied_Sleeping": {
        "No_Action": -100.0,       # Catastrophic: ignoring threat while resident sleeps!
        "Monitor": -20.0,          # Under-reacting to active risk
        "Notify_Resident": 20.0,   # Wakes them up to respond
        "Silent_Alert": 35.0,      # Prepares external / security alert
        "Local_Alert": 45.0       # Highest safety priority: alarm warns sleeping family
    },
    "Unusual_Empty": {
        "No_Action": -120.0,       # Disaster: intrusion or hazard unaddressed in empty home
        "Monitor": -30.0,          # Way too passive for anomaly in empty house
        "Notify_Resident": 30.0,   # Pings owner remotely
        "Silent_Alert": 55.0,      # Best response: silent dispatch without alerting intruder
        "Local_Alert": 45.0       # Deters potential intruder
    }
}


def export_utility_table(output_dir: str = METRICS_DIR) -> str:
    """Exports the documented utility specification to utility_table.csv."""
    os.makedirs(output_dir, exist_ok=True)
    records = []
    for state, action_map in UTILITY_MATRIX.items():
        for action, util in action_map.items():
            records.append({
                "state": state,
                "action": action,
                "utility": util
            })
    df_util = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "utility_table.csv")
    df_util.to_csv(csv_path, index=False)
    print(f"Exported utility specification to {csv_path}")
    return csv_path


def compute_situational_state_probabilities(posteriors: dict) -> dict:
    """
    Computes probability distribution over the 6 mutually exclusive situational states
    from the marginal posteriors of Home_Occupancy, Resident_Sleep, and Unusual_Activity.
    """
    p_unusual = posteriors.get("Unusual_Activity", {}).get("Unusual", 0.01)
    p_normal = 1.0 - p_unusual
    
    p_occupied = posteriors.get("Home_Occupancy", {}).get("Occupied", 0.95)
    p_empty = 1.0 - p_occupied
    
    p_sleeping = posteriors.get("Resident_Sleep", {}).get("Sleeping", 0.05)
    p_awake = 1.0 - p_sleeping
    
    state_probs = {
        "Normal_Awake": p_normal * p_occupied * p_awake,
        "Normal_Sleeping": p_normal * p_occupied * p_sleeping,
        "Normal_Empty": p_normal * p_empty,
        "Unusual_Occupied_Awake": p_unusual * p_occupied * p_awake,
        "Unusual_Occupied_Sleeping": p_unusual * p_occupied * p_sleeping,
        "Unusual_Empty": p_unusual * p_empty
    }
    
    # Normalize probabilities to sum to 1.0
    total_prob = sum(state_probs.values())
    if total_prob > 0:
        state_probs = {k: v / total_prob for k, v in state_probs.items()}
        
    return state_probs


def select_action_meu(posteriors: dict) -> dict:
    """
    Phase 23: Maximum Expected Utility (MEU) Action Selection.
    EU(action) = sum_state P(state | evidence) * U(action, state)
    best_action = argmax EU(action)
    """
    state_probs = compute_situational_state_probabilities(posteriors)
    
    expected_utilities = {}
    for action in ACTIONS:
        eu = sum(state_probs[state] * UTILITY_MATRIX[state][action] for state in SITUATIONAL_STATES)
        expected_utilities[action] = float(eu)
        
    best_action = max(expected_utilities, key=expected_utilities.get)
    
    return {
        "selected_action": best_action,
        "expected_utilities": expected_utilities,
        "state_probabilities": state_probs
    }


def select_rule_based_action(observation: dict) -> str:
    """
    Phase 24: Rule-Based Baseline System.
    Simple deterministic heuristic:
    - If Unusual_Activity is Unusual -> Local_Alert
    - Else If Returning_Home is Yes -> Monitor
    - Else If Leaving_Home is Yes -> Monitor
    - Else -> No_Action
    """
    unusual = observation.get("Unusual_Activity", "Normal")
    returning = observation.get("Returning_Home", "No")
    leaving = observation.get("Leaving_Home", "No")
    
    if unusual == "Unusual":
        return "Local_Alert"
    elif returning == "Yes":
        return "Monitor"
    elif leaving == "Yes":
        return "Monitor"
    else:
        return "No_Action"


def evaluate_action_realized_utility(action: str, ground_truth: dict) -> float:
    """
    Computes realized utility for an action given ground truth / weakly labeled state.
    """
    is_unusual = ground_truth.get("Unusual_Activity") == "Unusual"
    is_occupied = ground_truth.get("Home_Occupancy") != "Empty"
    is_sleeping = ground_truth.get("Resident_Sleep") == "Sleeping"
    
    if is_unusual:
        if not is_occupied:
            true_state = "Unusual_Empty"
        elif is_sleeping:
            true_state = "Unusual_Occupied_Sleeping"
        else:
            true_state = "Unusual_Occupied_Awake"
    else:
        if not is_occupied:
            true_state = "Normal_Empty"
        elif is_sleeping:
            true_state = "Normal_Sleeping"
        else:
            true_state = "Normal_Awake"
            
    return UTILITY_MATRIX[true_state].get(action, 0.0)


if __name__ == "__main__":
    export_utility_table()
    
    # Test MEU calculation on high unusual probability scenario
    mock_posteriors = {
        "Home_Occupancy": {"Occupied": 0.99, "Empty": 0.01},
        "Resident_Sleep": {"Sleeping": 0.85, "Awake": 0.15},
        "Unusual_Activity": {"Unusual": 0.70, "Normal": 0.30}
    }
    decision = select_action_meu(mock_posteriors)
    print("\nDecision Output for Unusual Sleeping scenario:")
    print(f"Selected Action: {decision['selected_action']}")
    print("Expected Utilities:", decision["expected_utilities"])
