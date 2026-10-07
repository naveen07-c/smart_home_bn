"""
Live Dashboard Backend for Probabilistic Smart Home Decision Network.
Serves the 26-node network structure, runs inference on test data, and streams results via WebSocket.
"""
import os
import sys
import json
import asyncio
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    EVIDENCE_NODES, BEHAVIORAL_NODES, DECISION_NODES, UTILITY_NODES,
    ALL_NODES, DAG_EDGES, BayesianInferenceEngine, build_full_dag
)
from src.decision import select_action_meu, ACTIONS, UTILITY_MATRIX, SITUATIONAL_STATES

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data exactly once at startup (replaces deprecated on_event)."""
    load_model_and_data()
    yield


app = FastAPI(title="Smart Home Decision Network - Live Dashboard", lifespan=lifespan)

# Mount static files
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Read template at startup
INDEX_HTML = (BASE_DIR / "templates" / "index.html").read_text()

# Global state
engine: Optional[BayesianInferenceEngine] = None
test_data: Optional[pd.DataFrame] = None
current_index = 0
connected_clients: List[WebSocket] = []
playback_task: Optional[asyncio.Task] = None
playback_speed = 1.0  # seconds per window
is_playing = False


def load_model_and_data():
    """Load the trained model and test dataset."""
    global engine, test_data, current_index
    
    model_path = BASE_DIR.parent / "outputs" / "models" / "bayesian_network.pkl"
    test_path = BASE_DIR.parent / "data" / "processed" / "test.csv"
    
    engine = BayesianInferenceEngine(str(model_path))
    test_data = pd.read_csv(test_path, keep_default_na=False, dtype=str)
    current_index = 0
    print(f"Loaded model and {len(test_data)} test windows")


# Deterministic pipeline-flow layout: stage columns ordered left -> right to
# mirror the project's inference pipeline. Column membership follows the DAG
# in src/model.py so every arrow either moves forward across columns or arcs
# within one. Distributed across 8 columns to prevent overlap.
#   col 0  Temporal inputs
#   col 1  Door sensors
#   col 2  Primary rooms
#   col 3  Secondary rooms
#   col 4  Aggregates
#   col 5  Behavioral (inferred)
#   col 6  Decision
#   col 7  Utility
STAGE_COLUMNS = [
    ("Temporal Inputs", ["Day_Of_Week", "Time_Of_Day", "Is_Night"]),
    ("Door Sensors", ["OutsideDoor_Open", "OutsideDoor_Close", "Total_Contact_Events", "OutsideDoor_Activity"]),
    ("Primary Rooms", ["Bedroom_Activity", "Kitchen_Activity", "LivingRoom_Activity"]),
    ("Secondary Rooms", ["Bathroom_Activity", "DiningRoom_Activity", "GuestRoom_Activity",
                           "LoungeChair_Activity", "OtherRoom_Activity", "WorkArea_Activity", "Hall_Activity"]),
    ("Aggregates", ["Total_Activity"]),
    ("Behavioral (Inferred)", ["Home_Occupancy", "Resident_Activity", "Resident_Sleep",
                                 "Leaving_Home", "Returning_Home", "Unusual_Activity"]),
    ("Decision", ["Smart_Home_Action"]),
    ("Utility", ["Homeowner_Utility"]),
]


def get_network_structure() -> Dict[str, Any]:
    """Get the 26-node network structure for visualization.

    Uses a deterministic stage-column layout (no spring randomness) so nodes
    are ordered along the pipeline flow: evidence -> behavioral -> decision
    -> utility, with generous spacing between columns and rows.
    """
    placed = [n for _, col_nodes in STAGE_COLUMNS for n in col_nodes]
    assert sorted(placed) == sorted(ALL_NODES), (
        "STAGE_COLUMNS must place every node exactly once"
    )

    n_cols = len(STAGE_COLUMNS)
    x_pad = 0.06   # horizontal margin around the outermost columns
    y_pad = 0.16   # vertical margin above/below multi-node columns

    pos: Dict[str, tuple] = {}
    col_x: List[float] = []
    for c, (_, col_nodes) in enumerate(STAGE_COLUMNS):
        x = x_pad + (c / (n_cols - 1)) * (1 - 2 * x_pad)
        col_x.append(x)
        n = len(col_nodes)
        for r, node in enumerate(col_nodes):
            if n == 1:
                y = 0.5
            else:
                y = y_pad + (r / (n - 1)) * (1 - 2 * y_pad)
            pos[node] = (x, y)

    nodes = []
    for node in ALL_NODES:
        x_norm, y_norm = pos[node]

        # Determine category
        if node in EVIDENCE_NODES:
            category = "evidence"
            color = "#6baed6"
        elif node in BEHAVIORAL_NODES:
            category = "behavioral"
            color = "#74c476"
        elif node in DECISION_NODES:
            category = "decision"
            color = "#fd8d3c"
        else:
            category = "utility"
            color = "#9e9ac8"

        nodes.append({
            "id": node,
            "label": node.replace("_", " "),
            "category": category,
            "color": color,
            "x": x_norm,
            "y": y_norm,
        })

    edges = [{"source": u, "target": v} for u, v in DAG_EDGES]
    stages = [
        {"label": label, "x": col_x[c], "count": len(col_nodes)}
        for c, (label, col_nodes) in enumerate(STAGE_COLUMNS)
    ]

    return {"nodes": nodes, "edges": edges, "stages": stages}


def run_inference_on_window(window_data: pd.Series) -> Dict[str, Any]:
    """Run Bayesian inference and MEU decision on a single window."""
    if engine is None:
        return {}
    
    # Build evidence dict (exclude Unavailable)
    evidence = {}
    for col in EVIDENCE_NODES:
        val = str(window_data[col])
        if val != "Unavailable":
            evidence[col] = val
    
    # Run inference
    posteriors = engine.query(evidence)
    
    # Run MEU decision
    decision = select_action_meu(posteriors, engine=engine, evidence=evidence)
    
    # Get true weak labels for comparison
    true_labels = {var: str(window_data[var]) for var in BEHAVIORAL_NODES}
    
    # Get MAP predictions
    predictions = {}
    for var in BEHAVIORAL_NODES:
        dist = posteriors[var]
        predictions[var] = max(dist, key=dist.get)
    
    return {
        "window_id": int(window_data.name) if hasattr(window_data, 'name') else current_index,
        "timestamp": str(window_data["window_start"]),
        "home_id": str(window_data["home_id"]),
        "evidence": evidence,
        "posteriors": posteriors,
        "decision": decision,
        "true_labels": true_labels,
        "predictions": predictions
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the main dashboard page."""
    return HTMLResponse(INDEX_HTML)


@app.get("/api/network")
async def get_network():
    """Return the 26-node network structure."""
    return get_network_structure()


@app.get("/api/nodes")
async def get_nodes():
    """Return node definitions with categories."""
    return {
        "evidence": EVIDENCE_NODES,
        "behavioral": BEHAVIORAL_NODES,
        "decision": DECISION_NODES,
        "utility": UTILITY_NODES,
        "all": ALL_NODES
    }


@app.get("/api/actions")
async def get_actions():
    """Return available actions and utility matrix."""
    return {
        "actions": ACTIONS,
        "utility_matrix": UTILITY_MATRIX,
        "situational_states": SITUATIONAL_STATES
    }


@app.get("/api/window/{idx}")
async def get_window(idx: int):
    """Get inference results for a specific window index."""
    if test_data is None or idx < 0 or idx >= len(test_data):
        return {"error": "Window not found"}
    
    window = test_data.iloc[idx]
    result = run_inference_on_window(window)
    result["index"] = idx
    result["total_windows"] = len(test_data)
    return result


@app.get("/api/next-window")
async def get_next_window():
    """Get the next window in sequence (for playback)."""
    global current_index
    if test_data is None:
        return {"error": "No data loaded"}
    
    if current_index >= len(test_data):
        current_index = 0  # Loop back
    
    window = test_data.iloc[current_index]
    result = run_inference_on_window(window)
    result["index"] = current_index
    result["total_windows"] = len(test_data)
    
    current_index += 1
    return result


@app.get("/api/stats")
async def get_stats():
    """Get overall statistics."""
    if test_data is None:
        return {}
    
    return {
        "total_windows": len(test_data),
        "current_index": current_index,
        "homes": test_data["home_id"].value_counts().to_dict(),
        "is_playing": is_playing,
        "playback_speed": playback_speed
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live streaming inference results."""
    global is_playing, playback_task, playback_speed, current_index
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            # Keep connection alive, handle client messages
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue
            
            if msg.get("type") == "play":
                is_playing = True
                try:
                    playback_speed = float(msg.get("speed", 1.0))
                except (TypeError, ValueError):
                    playback_speed = 1.0
                if playback_task is None or playback_task.done():
                    playback_task = asyncio.create_task(playback_loop())
            
            elif msg.get("type") == "pause":
                is_playing = False
            
            elif msg.get("type") == "speed":
                try:
                    playback_speed = float(msg.get("speed", 1.0))
                except (TypeError, ValueError):
                    playback_speed = 1.0
            
            elif msg.get("type") == "jump":
                if test_data is None or len(test_data) == 0:
                    continue
                idx = min(max(0, msg.get("index", 0)), len(test_data) - 1)
                window = test_data.iloc[idx]
                result = run_inference_on_window(window)
                result["index"] = idx
                result["total_windows"] = len(test_data)
                # Playback continues AFTER the jumped-to window.
                current_index = idx + 1
                await websocket.send_text(json.dumps({"type": "update", "data": result}))
    
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


async def playback_loop():
    """Background task to stream windows at configured speed."""
    global current_index, is_playing, connected_clients
    
    while is_playing and connected_clients:
        if test_data is None:
            break
        
        if current_index >= len(test_data):
            current_index = 0
        
        window = test_data.iloc[current_index]
        result = run_inference_on_window(window)
        result["index"] = current_index
        result["total_windows"] = len(test_data)
        
        message = json.dumps({"type": "update", "data": result})
        
        # Send to all connected clients
        disconnected = []
        for client in connected_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected.append(client)
        
        for client in disconnected:
            connected_clients.remove(client)
        
        current_index += 1
        await asyncio.sleep(playback_speed)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)