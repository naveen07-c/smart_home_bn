"""
26-Node Bayesian / Decision Network Architecture and Inference Engine.
Covers Phases 16 - 20:
- Explicit 26-node architecture (18 evidence, 6 behavioral, 1 decision, 1 utility)
- Acyclic DAG design with sparse, semantically meaningful dependencies
- Graph structure validation and export to network_structure.txt & network_dag.png
- Bayesian parameter learning (CPT estimation) using Dirichlet / BDeu smoothing
- Probabilistic posterior inference for evidence windows
"""

import os
import joblib
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from pgmpy.inference import VariableElimination

METRICS_DIR = "outputs/metrics"
FIGURES_DIR = "outputs/figures"
MODELS_DIR = "outputs/models"

# 18 Observable / Evidence Variables
EVIDENCE_NODES = [
    "Bedroom_Activity", "Bathroom_Activity", "DiningRoom_Activity", "GuestRoom_Activity",
    "Kitchen_Activity", "LivingRoom_Activity", "LoungeChair_Activity", "OtherRoom_Activity",
    "OutsideDoor_Activity", "WorkArea_Activity", "Hall_Activity",
    "OutsideDoor_Open", "OutsideDoor_Close",
    "Time_Of_Day", "Day_Of_Week", "Is_Night",
    "Total_Activity", "Total_Contact_Events"
]

# 6 Inferred / Behavioral Variables
BEHAVIORAL_NODES = [
    "Home_Occupancy", "Resident_Activity", "Resident_Sleep",
    "Leaving_Home", "Returning_Home", "Unusual_Activity"
]

# 1 Decision Node
DECISION_NODES = ["Smart_Home_Action"]

# 1 Utility Node
UTILITY_NODES = ["Homeowner_Utility"]

# Exact 26 Nodes
ALL_NODES = EVIDENCE_NODES + BEHAVIORAL_NODES + DECISION_NODES + UTILITY_NODES
assert len(ALL_NODES) == 26, f"Expected 26 nodes, got {len(ALL_NODES)}"

# Semantic dependencies for the 26-node Decision Network DAG
DAG_EDGES = [
    # Temporal evidence dependencies
    ("Day_Of_Week", "Time_Of_Day"),
    ("Time_Of_Day", "Is_Night"),
    
    # Door and contact evidence dependencies
    ("OutsideDoor_Open", "Total_Contact_Events"),
    ("OutsideDoor_Close", "Total_Contact_Events"),
    ("Total_Contact_Events", "OutsideDoor_Activity"),
    
    # Major Room Activities influencing Total Activity
    ("LivingRoom_Activity", "Total_Activity"),
    ("Kitchen_Activity", "Total_Activity"),
    ("Bedroom_Activity", "Total_Activity"),
    
    # Home Occupancy dependencies
    ("Total_Activity", "Home_Occupancy"),
    ("OutsideDoor_Activity", "Home_Occupancy"),
    
    # Resident Activity dependencies
    ("Home_Occupancy", "Resident_Activity"),
    ("Total_Activity", "Resident_Activity"),
    ("Time_Of_Day", "Resident_Activity"),
    
    # Secondary Room Activities conditioned on overall Resident Activity
    ("Resident_Activity", "Bathroom_Activity"),
    ("Resident_Activity", "DiningRoom_Activity"),
    ("Resident_Activity", "GuestRoom_Activity"),
    ("Resident_Activity", "LoungeChair_Activity"),
    ("Resident_Activity", "OtherRoom_Activity"),
    ("Resident_Activity", "WorkArea_Activity"),
    ("Resident_Activity", "Hall_Activity"),
    
    # Resident Sleep dependencies
    ("Home_Occupancy", "Resident_Sleep"),
    ("Is_Night", "Resident_Sleep"),
    ("Bedroom_Activity", "Resident_Sleep"),
    
    # Leaving Home dependencies
    ("Home_Occupancy", "Leaving_Home"),
    ("OutsideDoor_Activity", "Leaving_Home"),
    ("Total_Contact_Events", "Leaving_Home"),
    
    # Returning Home dependencies
    ("Home_Occupancy", "Returning_Home"),
    ("OutsideDoor_Activity", "Returning_Home"),
    ("Total_Contact_Events", "Returning_Home"),
    
    # Unusual Activity dependencies
    ("Home_Occupancy", "Unusual_Activity"),
    ("Resident_Activity", "Unusual_Activity"),
    ("Is_Night", "Unusual_Activity"),
    ("OutsideDoor_Activity", "Unusual_Activity"),
    
    # Decision node informational influences
    ("Unusual_Activity", "Smart_Home_Action"),
    ("Home_Occupancy", "Smart_Home_Action"),
    ("Resident_Sleep", "Smart_Home_Action"),
    
    # Utility node dependencies (payoff depends on state + action)
    ("Smart_Home_Action", "Homeowner_Utility"),
    ("Unusual_Activity", "Homeowner_Utility"),
    ("Home_Occupancy", "Homeowner_Utility")
]


def build_full_dag() -> nx.DiGraph:
    """Constructs the complete 26-node directed graph."""
    G = nx.DiGraph()
    G.add_nodes_from(ALL_NODES)
    G.add_edges_from(DAG_EDGES)
    return G


def validate_network_structure(output_metrics_dir: str = METRICS_DIR, output_figures_dir: str = FIGURES_DIR):
    """
    Phase 18: Graph Validation.
    1. Check every node exists (count == 26).
    2. Check every edge references valid nodes.
    3. Check the graph is acyclic.
    4. Check non-root node parent sets.
    5. Save outputs/metrics/network_structure.txt.
    6. Save outputs/figures/network_dag.png.
    """
    os.makedirs(output_metrics_dir, exist_ok=True)
    os.makedirs(output_figures_dir, exist_ok=True)
    
    G = build_full_dag()
    
    # Validations
    assert len(G.nodes) == 26, f"Error: Graph has {len(G.nodes)} nodes, expected exactly 26."
    for u, v in G.edges:
        assert u in ALL_NODES, f"Invalid source node: {u}"
        assert v in ALL_NODES, f"Invalid target node: {v}"
    assert nx.is_directed_acyclic_graph(G), "Error: Network contains cycles!"
    
    in_degrees = dict(G.in_degree())
    max_in_degree = max(in_degrees.values())
    assert max_in_degree <= 6, f"Parent set size too large: {max_in_degree}"
    
    # Structure report
    structure_file = os.path.join(output_metrics_dir, "network_structure.txt")
    with open(structure_file, "w") as f:
        f.write("=== 26-NODE PROBABILISTIC SMART HOME DECISION NETWORK ===\n\n")
        f.write(f"Total Nodes: {len(G.nodes)}\n")
        f.write(f"Total Edges: {len(G.edges)}\n")
        f.write(f"Is Acyclic (DAG): {nx.is_directed_acyclic_graph(G)}\n")
        f.write(f"Max In-Degree (Max Parents): {max_in_degree}\n\n")
        
        f.write("--- NODE CATEGORIES ---\n")
        f.write(f"1. Evidence Nodes ({len(EVIDENCE_NODES)}):\n")
        for node in EVIDENCE_NODES:
            f.write(f"   - {node}\n")
        f.write(f"\n2. Behavioral Nodes ({len(BEHAVIORAL_NODES)}):\n")
        for node in BEHAVIORAL_NODES:
            f.write(f"   - {node}\n")
        f.write(f"\n3. Decision Nodes ({len(DECISION_NODES)}):\n")
        for node in DECISION_NODES:
            f.write(f"   - {node}\n")
        f.write(f"\n4. Utility Nodes ({len(UTILITY_NODES)}):\n")
        for node in UTILITY_NODES:
            f.write(f"   - {node}\n")
            
        f.write("\n--- NODE PARENTS (CONDITIONAL DEPENDENCIES) ---\n")
        for node in sorted(G.nodes):
            parents = list(G.predecessors(node))
            if parents:
                f.write(f"P({node} | {', '.join(parents)})\n")
            else:
                f.write(f"P({node}) [Root / Prior Node]\n")
                
    # Visual graph generation
    plt.figure(figsize=(18, 14))
    pos = nx.spring_layout(G, seed=42, k=1.8, iterations=100)
    
    # Color nodes by category
    node_colors = []
    for node in G.nodes:
        if node in EVIDENCE_NODES:
            node_colors.append("#6baed6")      # Soft blue for evidence
        elif node in BEHAVIORAL_NODES:
            node_colors.append("#74c476")      # Green for behavioral
        elif node in DECISION_NODES:
            node_colors.append("#fd8d3c")      # Orange for decision
        else:
            node_colors.append("#9e9ac8")      # Purple for utility
            
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3200, edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=18, min_source_margin=15, min_target_margin=15)
    
    plt.title("26-Node Probabilistic Smart Home Decision Network DAG", fontsize=16, fontweight="bold", pad=20)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Evidence (18)', markerfacecolor='#6baed6', markersize=14),
        plt.Line2D([0], [0], marker='o', color='w', label='Behavioral (6)', markerfacecolor='#74c476', markersize=14),
        plt.Line2D([0], [0], marker='o', color='w', label='Decision (1)', markerfacecolor='#fd8d3c', markersize=14),
        plt.Line2D([0], [0], marker='o', color='w', label='Utility (1)', markerfacecolor='#9e9ac8', markersize=14)
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    
    dag_fig_path = os.path.join(output_figures_dir, "network_dag.png")
    plt.savefig(dag_fig_path, dpi=300)
    plt.close()
    
    print(f"Network validation passed: {structure_file} and {dag_fig_path}")
    return G


def train_bayesian_network(df_train: pd.DataFrame, model_output_path: str = os.path.join(MODELS_DIR, "bayesian_network.pkl")):
    """
    Phase 19: CPT Learning.
    Trains the 24 probabilistic nodes (18 evidence + 6 behavioral) on training data.
    Uses Dirichlet Bayesian parameter estimation with BDeu prior smoothing.
    """
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # Filter edges between probabilistic nodes
    prob_nodes = set(EVIDENCE_NODES + BEHAVIORAL_NODES)
    prob_edges = [(u, v) for u, v in DAG_EDGES if u in prob_nodes and v in prob_nodes]
    
    model = DiscreteBayesianNetwork(prob_edges)
    
    # Extract only required columns with clean string types
    train_data = df_train[list(prob_nodes)].astype(str)
    
    print("Learning Bayesian CPTs using DiscreteBayesianEstimator (prior_type='BDeu')...")
    estimator = DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=5)
    model.fit(train_data, estimator=estimator)
    
    assert model.check_model(), "Learned model failed validity checks!"
    print(f"Learned CPDs for {len(model.get_cpds())} variables.")
    
    joblib.dump(model, model_output_path)
    print(f"Saved trained Bayesian Network to {model_output_path}")
    return model


class BayesianInferenceEngine:
    """
    Phase 20: Bayesian Inference Engine.
    Computes exact posterior marginals for the 6 behavioral variables given observed evidence.
    """
    def __init__(self, model_path: str = os.path.join(MODELS_DIR, "bayesian_network.pkl")):
        if isinstance(model_path, str):
            self.model = joblib.load(model_path)
        else:
            self.model = model_path
        self.infer = VariableElimination(self.model)
        self.target_vars = BEHAVIORAL_NODES
        
    def query(self, evidence: dict, variables: list = None) -> dict:
        """
        Executes inference for specified variables given observed evidence.
        Filters evidence to only include valid variables and recognized states.
        Warns when evidence is dropped due to invalid variables or states.
        """
        import warnings
        if variables is None:
            variables = self.target_vars
            
        clean_evidence = {}
        dropped_evidence = []
        for k, v in evidence.items():
            if k in self.model.nodes and k not in variables:
                val_str = str(v)
                # Verify that state exists in CPD for node
                try:
                    cpd = self.model.get_cpds(k)
                    if val_str in cpd.state_names[k]:
                        clean_evidence[k] = val_str
                    else:
                        dropped_evidence.append(f"{k}={val_str} (valid: {list(cpd.state_names[k])})")
                except Exception as e:
                    dropped_evidence.append(f"{k}={val_str} (error: {e})")
            elif k not in self.model.nodes:
                dropped_evidence.append(f"{k}={v} (unknown variable)")
            elif k in variables:
                dropped_evidence.append(f"{k}={v} (target variable)")
                
        if dropped_evidence:
            warnings.warn(f"Dropped evidence: {', '.join(dropped_evidence)}", UserWarning)
                    
        # Variable elimination query
        res = self.infer.query(variables=variables, evidence=clean_evidence, joint=False)
        
        posteriors = {}
        for var in variables:
            factor = res[var]
            states = factor.state_names[var]
            probs = factor.values.flatten()
            posteriors[var] = {state: float(prob) for state, prob in zip(states, probs)}
            
        return posteriors


if __name__ == "__main__":
    validate_network_structure()
    df_tr = pd.read_csv("data/processed/train.csv", keep_default_na=False, dtype=str)
    trained_model = train_bayesian_network(df_tr)
    
    engine = BayesianInferenceEngine(trained_model)
    test_evidence = {
        "Bedroom_Activity": "Low",
        "Kitchen_Activity": "None",
        "LivingRoom_Activity": "None",
        "Is_Night": "Yes",
        "Total_Activity": "Low",
        "Time_Of_Day": "Night"
    }
    result = engine.query(test_evidence)
    print("\nSample Posterior Marginals:")
    for var, dist in result.items():
        print(f"  {var}: {dist}")
