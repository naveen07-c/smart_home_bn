"""
Comprehensive Evaluation Suite for Probabilistic Smart Home Decision Network.
Covers Phases 25 - 30:
- Behavioral inference evaluation (Accuracy, Precision, Recall, F1, Confusion Matrices)
- Decision metrics & comparative analysis (Bayesian MEU vs Rule-Based Baseline)
- Real hardware latency measurement
- Cross-home generalization experiments (Aruba->Aruba, Aruba->Milan, Aruba+Milan->Cairo)
- Visualization figure generation (10 figures)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.model import (
    EVIDENCE_NODES, BEHAVIORAL_NODES, DAG_EDGES,
    BayesianInferenceEngine, train_bayesian_network, build_full_dag
)
from src.decision import (
    ACTIONS, select_action_meu, select_rule_based_action,
    evaluate_action_realized_utility, UTILITY_MATRIX
)
from src.preprocessing import HOME_SENSORS, HOME_DOOR_SEMANTICS

METRICS_DIR = "outputs/metrics"
FIGURES_DIR = "outputs/figures"
PROCESSED_DIR = "data/processed"


def evaluate_test_set(engine: BayesianInferenceEngine, df_test: pd.DataFrame, max_samples: int = 3000):
    """
    Evaluates Bayesian inference and decision making on unseen test data.
    Measures accuracy, precision, recall, F1, utilities, and latency.
    """
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Subsample if test set is very large to balance thoroughness with execution time
    if len(df_test) > max_samples:
        test_sub = df_test.sample(n=max_samples, random_state=42).reset_index(drop=True)
    else:
        test_sub = df_test.reset_index(drop=True)
        
    latencies = []
    y_true = {var: [] for var in BEHAVIORAL_NODES}
    y_pred_bn = {var: [] for var in BEHAVIORAL_NODES}
    
    bn_actions = []
    rule_actions = []
    bn_utilities = []
    rule_utilities = []
    
    sample_posteriors = None
    sample_decision = None
    
    print(f"Evaluating {len(test_sub)} test windows...")
    for idx, row in test_sub.iterrows():
        # Formulate available evidence
        evidence = {}
        for col in EVIDENCE_NODES:
            val = str(row[col])
            if val != "Unavailable":
                evidence[col] = val
                
        # Time the inference call
        t_start = time.perf_counter()
        posteriors = engine.query(evidence)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0  # in ms
        latencies.append(t_elapsed)
        
        # Predicted states via MAP (Argmax probability)
        for var in BEHAVIORAL_NODES:
            dist = posteriors[var]
            pred_state = max(dist, key=dist.get)
            y_pred_bn[var].append(pred_state)
            y_true[var].append(str(row[var]))
            
        # Decision Layer
        decision_bn = select_action_meu(posteriors)
        bn_action = decision_bn["selected_action"]
        bn_actions.append(bn_action)
        
        # Rule baseline
        obs_dict = {k: str(row[k]) for k in row.index}
        rule_action = select_rule_based_action(obs_dict)
        rule_actions.append(rule_action)
        
        # Realized utility calculation
        true_dict = {var: str(row[var]) for var in BEHAVIORAL_NODES}
        u_bn = evaluate_action_realized_utility(bn_action, true_dict)
        u_rule = evaluate_action_realized_utility(rule_action, true_dict)
        bn_utilities.append(u_bn)
        rule_utilities.append(u_rule)
        
        if sample_posteriors is None and row.get("Unusual_Activity") == "Unusual":
            sample_posteriors = posteriors
            sample_decision = decision_bn
            
    if sample_posteriors is None:
        sample_posteriors = posteriors
        sample_decision = decision_bn
        
    # Phase 25: Behavioral Metrics Calculation
    behavioral_records = []
    confusion_matrices = {}
    for var in BEHAVIORAL_NODES:
        yt = y_true[var]
        yp = y_pred_bn[var]
        labels = sorted(list(set(yt) | set(yp)))
        
        acc = accuracy_score(yt, yp)
        prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
        cm = confusion_matrix(yt, yp, labels=labels)
        confusion_matrices[var] = (cm, labels)
        
        behavioral_records.append({
            "variable": var,
            "accuracy": float(acc),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
            "macro_f1": float(f1),
            "weighted_precision": float(w_prec),
            "weighted_recall": float(w_rec),
            "weighted_f1": float(w_f1)
        })
        
    df_behavioral = pd.DataFrame(behavioral_records)
    beh_path = os.path.join(METRICS_DIR, "behavioral_metrics.csv")
    df_behavioral.to_csv(beh_path, index=False)
    
    # Phase 26: Decision Metrics Calculation
    # False alert: alert fired (Notify, Silent, Local) when situation was Normal
    # Unnecessary intervention: intervention fired when No_Action had highest true utility
    n_samples = len(test_sub)
    true_states = [test_sub.iloc[i].to_dict() for i in range(n_samples)]
    
    false_alert_bn = sum(1 for a, s in zip(bn_actions, true_states) if a in ["Silent_Alert", "Local_Alert"] and s["Unusual_Activity"] == "Normal")
    false_alert_rule = sum(1 for a, s in zip(rule_actions, true_states) if a in ["Silent_Alert", "Local_Alert"] and s["Unusual_Activity"] == "Normal")
    
    unusual_total = sum(1 for s in true_states if s["Unusual_Activity"] == "Unusual")
    unusual_detected_bn = sum(1 for a, s in zip(bn_actions, true_states) if a in ["Notify_Resident", "Silent_Alert", "Local_Alert"] and s["Unusual_Activity"] == "Unusual")
    unusual_detected_rule = sum(1 for a, s in zip(rule_actions, true_states) if a in ["Notify_Resident", "Silent_Alert", "Local_Alert"] and s["Unusual_Activity"] == "Unusual")
    
    decision_records = [
        {
            "model": "Bayesian_Decision_Network",
            "mean_utility": float(np.mean(bn_utilities)),
            "median_utility": float(np.median(bn_utilities)),
            "total_utility": float(np.sum(bn_utilities)),
            "std_utility": float(np.std(bn_utilities)),
            "false_alert_rate": float(false_alert_bn / n_samples),
            "unusual_detection_rate": float(unusual_detected_bn / max(1, unusual_total)),
            "pct_No_Action": float(bn_actions.count("No_Action") / n_samples * 100),
            "pct_Monitor": float(bn_actions.count("Monitor") / n_samples * 100),
            "pct_Notify_Resident": float(bn_actions.count("Notify_Resident") / n_samples * 100),
            "pct_Silent_Alert": float(bn_actions.count("Silent_Alert") / n_samples * 100),
            "pct_Local_Alert": float(bn_actions.count("Local_Alert") / n_samples * 100)
        },
        {
            "model": "Rule_Based_Baseline",
            "mean_utility": float(np.mean(rule_utilities)),
            "median_utility": float(np.median(rule_utilities)),
            "total_utility": float(np.sum(rule_utilities)),
            "std_utility": float(np.std(rule_utilities)),
            "false_alert_rate": float(false_alert_rule / n_samples),
            "unusual_detection_rate": float(unusual_detected_rule / max(1, unusual_total)),
            "pct_No_Action": float(rule_actions.count("No_Action") / n_samples * 100),
            "pct_Monitor": float(rule_actions.count("Monitor") / n_samples * 100),
            "pct_Notify_Resident": float(rule_actions.count("Notify_Resident") / n_samples * 100),
            "pct_Silent_Alert": float(rule_actions.count("Silent_Alert") / n_samples * 100),
            "pct_Local_Alert": float(rule_actions.count("Local_Alert") / n_samples * 100)
        }
    ]
    df_decision = pd.DataFrame(decision_records)
    dec_path = os.path.join(METRICS_DIR, "decision_metrics.csv")
    df_decision.to_csv(dec_path, index=False)
    
    # Phase 27: Inference Latency Performance
    latency_summary = {
        "sample_count": len(latencies),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "total_inference_time_sec": float(np.sum(latencies) / 1000.0)
    }
    lat_path = os.path.join(METRICS_DIR, "latency_metrics.json")
    with open(lat_path, "w") as f:
        json.dump(latency_summary, f, indent=2)
        
    # Generate Plots
    generate_evaluation_figures(
        test_sub, y_true, y_pred_bn, confusion_matrices,
        bn_actions, rule_actions, bn_utilities, rule_utilities,
        latencies, sample_posteriors, sample_decision
    )
    
    return df_behavioral, df_decision, latency_summary


def run_cross_home_experiments(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Phase 28: Cross-Home Generalization Experiments.
    Experiment 1: Train: Aruba, Test: Aruba
    Experiment 2: Train: Aruba, Test: Milan
    Experiment 3: Train: Aruba + Milan, Test: Cairo
    Uses only target-home supported sensors; unsupported variables are marginalized out.
    """
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    experiments = [
        {"name": "Exp1_Aruba_to_Aruba", "train_homes": ["Aruba"], "test_home": "Aruba"},
        {"name": "Exp2_Aruba_to_Milan", "train_homes": ["Aruba"], "test_home": "Milan"},
        {"name": "Exp3_ArubaMilan_to_Cairo", "train_homes": ["Aruba", "Milan"], "test_home": "Cairo"}
    ]
    
    results = []
    
    for exp in experiments:
        exp_name = exp["name"]
        print(f"\n--- Running Cross-Home Experiment: {exp_name} ---")
        train_h = df_train[df_train["home_id"].isin(exp["train_homes"])].reset_index(drop=True)
        test_h = df_test[df_test["home_id"] == exp["test_home"]].reset_index(drop=True)
        
        # Determine target home sensor availability
        target_sensors = HOME_SENSORS[exp["test_home"]]
        target_evidence = [f"{s}_Activity" for s in target_sensors]
        if HOME_DOOR_SEMANTICS[exp["test_home"]]:
            target_evidence += ["OutsideDoor_Open", "OutsideDoor_Close"]
        target_evidence += ["Time_Of_Day", "Day_Of_Week", "Is_Night", "Total_Activity", "Total_Contact_Events"]
        
        print(f"Target Home: {exp['test_home']} | Available Evidence Variables: {len(target_evidence)}")
        
        # Train specialized model for experiment
        model_exp = train_bayesian_network(train_h, os.path.join(METRICS_DIR, f"model_{exp_name}.pkl"))
        engine_exp = BayesianInferenceEngine(model_exp)
        
        # Subsample test if needed
        sub_test = test_h.sample(n=min(1500, len(test_h)), random_state=42).reset_index(drop=True)
        
        y_true_occ, y_pred_occ = [], []
        y_true_act, y_pred_act = [], []
        y_true_sleep, y_pred_sleep = [], []
        utilities = []
        
        for _, row in sub_test.iterrows():
            ev = {k: str(row[k]) for k in target_evidence if str(row[k]) != "Unavailable"}
            posteriors = engine_exp.query(ev)
            
            p_occ = max(posteriors["Home_Occupancy"], key=posteriors["Home_Occupancy"].get)
            p_act = max(posteriors["Resident_Activity"], key=posteriors["Resident_Activity"].get)
            p_slp = max(posteriors["Resident_Sleep"], key=posteriors["Resident_Sleep"].get)
            
            y_true_occ.append(str(row["Home_Occupancy"]))
            y_pred_occ.append(p_occ)
            y_true_act.append(str(row["Resident_Activity"]))
            y_pred_act.append(p_act)
            y_true_sleep.append(str(row["Resident_Sleep"]))
            y_pred_sleep.append(p_slp)
            
            dec = select_action_meu(posteriors)
            act = dec["selected_action"]
            u = evaluate_action_realized_utility(act, {var: str(row[var]) for var in BEHAVIORAL_NODES})
            utilities.append(u)
            
        results.append({
            "experiment": exp_name,
            "train_homes": "+".join(exp["train_homes"]),
            "test_home": exp["test_home"],
            "target_features_count": len(target_evidence),
            "occupancy_accuracy": float(accuracy_score(y_true_occ, y_pred_occ)),
            "activity_accuracy": float(accuracy_score(y_true_act, y_pred_act)),
            "sleep_accuracy": float(accuracy_score(y_true_sleep, y_pred_sleep)),
            "mean_utility": float(np.mean(utilities))
        })
        
    df_exp = pd.DataFrame(results)
    exp_path = os.path.join(METRICS_DIR, "cross_home_metrics.csv")
    df_exp.to_csv(exp_path, index=False)
    print(f"\nCross-home experiment results saved to {exp_path}")
    print(df_exp.to_string(index=False))
    
    # Plot cross home comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df_exp))
    width = 0.22
    
    plt.bar(x - width, df_exp["occupancy_accuracy"], width, label="Occupancy Acc", color="#3182bd")
    plt.bar(x, df_exp["activity_accuracy"], width, label="Activity Acc", color="#31a354")
    plt.bar(x + width, df_exp["sleep_accuracy"], width, label="Sleep Acc", color="#fd8d3c")
    
    plt.xticks(x, [r["experiment"] for r in results], rotation=15, ha="right", fontsize=10)
    plt.ylabel("Accuracy Score", fontsize=12)
    plt.title("Cross-Home Generalization Performance", fontsize=14, fontweight="bold")
    plt.ylim(0.0, 1.05)
    plt.legend(fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cross_home_performance.png"), dpi=300)
    plt.close()
    
    return df_exp


def generate_evaluation_figures(test_sub, y_true, y_pred_bn, confusion_matrices,
                                bn_actions, rule_actions, bn_utilities, rule_utilities,
                                latencies, sample_posteriors, sample_decision):
    """
    Phase 30: Generates all required evaluation visualizations.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 1. Dataset event distribution across homes
    plt.figure(figsize=(10, 5))
    df_ds = pd.read_csv(os.path.join(METRICS_DIR, "dataset_summary.csv"))
    sns.barplot(data=df_ds, x="home_id", y="rows", palette="Blues_d")
    plt.title("Raw CASAS Dataset Event Counts by Home", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Events", fontsize=12)
    plt.xlabel("Smart Home", fontsize=12)
    for idx, row in df_ds.iterrows():
        plt.text(idx, row["rows"] + 20000, f"{row['rows']:,}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "dataset_event_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Activity over time for a 24h sample period
    plt.figure(figsize=(14, 5))
    sample_series = test_sub.head(720).copy()
    sample_series["time_idx"] = range(len(sample_series))
    # Map Total_Activity to ordinal scale for visualization
    act_map = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    sample_series["act_val"] = sample_series["Total_Activity"].map(act_map).fillna(1)
    plt.plot(sample_series["time_idx"], sample_series["act_val"], color="#2ca02c", lw=1.2, label="Total Activity Level")
    plt.yticks([0, 1, 2, 3], ["None", "Low", "Medium", "High"])
    plt.title("Smart Home Activity Intensity Stream (Chronological 5-Second Windows)", fontsize=14, fontweight="bold")
    plt.xlabel("Consecutive 5-Second Windows", fontsize=12)
    plt.ylabel("Activity State", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "activity_over_time.png"), dpi=300)
    plt.close()
    
    # 5. Example posterior probability output & expected utilities
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # ax1: Posteriors
    plot_data = []
    for var, dist in sample_posteriors.items():
        for state, prob in dist.items():
            plot_data.append({"Variable": var, "State": state, "Probability": prob})
    df_plot = pd.DataFrame(plot_data)
    sns.barplot(data=df_plot, x="Variable", y="Probability", hue="State", ax=ax1)
    ax1.set_title("Posterior Marginals for Representative Window", fontsize=12, fontweight="bold")
    ax1.tick_params(axis='x', rotation=30)
    ax1.set_ylim(0, 1.05)
    
    # ax2: Expected Utilities
    eus = sample_decision["expected_utilities"]
    colors = ["#2b83ba" if a != sample_decision["selected_action"] else "#d7191c" for a in ACTIONS]
    ax2.bar(ACTIONS, [eus[a] for a in ACTIONS], color=colors)
    ax2.set_title(f"Action Expected Utilities (MEU Selected: {sample_decision['selected_action']})", fontsize=12, fontweight="bold")
    ax2.tick_params(axis='x', rotation=30)
    ax2.set_ylabel("Expected Utility", fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "example_posterior_output.png"), dpi=300)
    plt.close()
    
    # 6. Action distribution comparison
    plt.figure(figsize=(10, 5))
    df_act = pd.DataFrame({
        "Action": ACTIONS,
        "Bayesian_MEU": [bn_actions.count(a) / len(bn_actions) * 100 for a in ACTIONS],
        "Rule_Baseline": [rule_actions.count(a) / len(rule_actions) * 100 for a in ACTIONS]
    })
    df_act_melt = df_act.melt(id_vars="Action", var_name="System", value_name="Percentage")
    sns.barplot(data=df_act_melt, x="Action", y="Percentage", hue="System", palette=["#2b83ba", "#fdae61"])
    plt.title("Action Selection Distribution: Bayesian Decision Network vs Rule Baseline", fontsize=13, fontweight="bold")
    plt.ylabel("Percentage of Test Windows (%)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "action_distribution.png"), dpi=300)
    plt.close()
    
    # 7. Bayesian vs rule-based utility comparison
    plt.figure(figsize=(10, 5))
    cum_bn = np.cumsum(bn_utilities)
    cum_rule = np.cumsum(rule_utilities)
    plt.plot(cum_bn, label=f"Bayesian Decision Network (Total Utility: {sum(bn_utilities):,.0f})", color="#2b83ba", lw=2)
    plt.plot(cum_rule, label=f"Rule-Based Baseline (Total Utility: {sum(rule_utilities):,.0f})", color="#d7191c", lw=2, linestyle="--")
    plt.title("Cumulative Realized Utility Comparison on Test Set", fontsize=14, fontweight="bold")
    plt.xlabel("Test Windows Evaluated", fontsize=12)
    plt.ylabel("Cumulative Utility Payoff", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "utility_comparison.png"), dpi=300)
    plt.close()
    
    # 8. Confusion matrices for key behavioral variables
    key_vars = ["Home_Occupancy", "Resident_Sleep", "Unusual_Activity", "Resident_Activity"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for idx, var in enumerate(key_vars):
        ax = axes[idx // 2, idx % 2]
        cm, labels = confusion_matrices[var]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"Confusion Matrix: {var}", fontsize=12, fontweight="bold")
        ax.set_ylabel("True (Weak Label)")
        ax.set_xlabel("Predicted (Bayesian MAP)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"), dpi=300)
    plt.close()
    
    # 10. Inference latency distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(latencies, bins=40, color="#756bb1", kde=True)
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    plt.axvline(mean_lat, color="red", linestyle="--", label=f"Mean Latency ({mean_lat:.2f} ms)")
    plt.axvline(p95_lat, color="orange", linestyle=":", label=f"95th Percentile ({p95_lat:.2f} ms)")
    plt.title("Per-Window Bayesian Inference Latency Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Inference Latency (milliseconds)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "latency_distribution.png"), dpi=300)
    plt.close()
    
    print(f"Generated all evaluation figures in {FIGURES_DIR}")


if __name__ == "__main__":
    df_te = pd.read_csv("data/processed/test.csv", keep_default_na=False, dtype=str)
    df_tr = pd.read_csv("data/processed/train.csv", keep_default_na=False, dtype=str)
    
    eng = BayesianInferenceEngine()
    df_beh, df_dec, lat_stat = evaluate_test_set(eng, df_te, max_samples=3000)
    print("\n--- Behavioral Metrics Summary ---")
    print(df_beh.to_string(index=False))
    print("\n--- Decision Metrics Summary ---")
    print(df_dec.to_string(index=False))
    print("\n--- Latency Performance ---")
    print(lat_stat)
    
    run_cross_home_experiments(df_tr, df_te)
