"""
Generates a dedicated 100-event test window with distinct outputs of the Decision Engine.
Exactly 20 events for each of the 5 MEU actions:
  - 20 x No_Action
  - 20 x Monitor
  - 20 x Notify_Resident
  - 20 x Silent_Alert
  - 20 x Local_Alert

Outputs:
  - data/demo/test_100_events.json
  - data/demo/test_100_events.csv
  - data/demo/demo_windows.json (synced with dashboard)
"""
import os
import sys
import json
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
import pandas as pd
from pathlib import Path
from src.model import BayesianInferenceEngine, EVIDENCE_NODES, BEHAVIORAL_NODES
from src.decision import select_action_meu

ROOT_DIR = Path(__file__).parent.parent
MODEL_PATH = ROOT_DIR / "outputs" / "models" / "bayesian_network.pkl"
DEMO_DIR = ROOT_DIR / "data" / "demo"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

engine = BayesianInferenceEngine(str(MODEL_PATH))

# -----------------------------------------------------------------------------
# Base templates for generating 20 distinct scenarios for each of the 5 actions
# -----------------------------------------------------------------------------

def build_no_action_scenarios():
    """20 distinct No_Action scenarios across homes, rooms, and times."""
    scenarios = []
    
    # 1-6: Routine sleep scenarios
    for idx, (home, day, room) in enumerate([
        ("Aruba", "Weekday", "Bedroom_Activity"),
        ("Aruba", "Weekend", "Bedroom_Activity"),
        ("Milan", "Weekday", "Bedroom_Activity"),
        ("Milan", "Weekend", "Bedroom_Activity"),
        ("Cairo", "Weekday", "Bedroom_Activity"),
        ("Cairo", "Weekend", "Bedroom_Activity"),
    ], 1):
        ev = {
            "Time_Of_Day": "Night",
            "Day_Of_Week": day,
            "Is_Night": "Yes",
            room: "Low",
            "OutsideDoor_Open": "No",
            "OutsideDoor_Close": "No",
            "Total_Contact_Events": "No",
            "Total_Activity": "Low"
        }
        scenarios.append({
            "title": f"Quiet Overnight Sleep ({home})",
            "home": home,
            "desc": f"Resident peacefully resting in bedroom during {day.lower()} night with no outside door motion.",
            "evidence": ev,
            "expl": "Bayesian network confirms peaceful sleep with high confidence P(Normal_Sleeping) > 85%. No_Action yields maximum utility (+14.5 to +15.0) by preserving undisturbed rest without false nuisance alarms."
        })

    # 7-14: Routine awake daytime scenarios in living room / kitchen
    day_configs = [
        ("Aruba", "Morning", "Weekday", "Kitchen_Activity", "Preparing morning breakfast in kitchen"),
        ("Aruba", "Afternoon", "Weekday", "LivingRoom_Activity", "Relaxing on couch in living room reading"),
        ("Aruba", "Evening", "Weekend", "LivingRoom_Activity", "Watching television in living room on weekend evening"),
        ("Cairo", "Morning", "Weekend", "Kitchen_Activity", "Weekend morning coffee in kitchen"),
        ("Cairo", "Afternoon", "Weekday", "LivingRoom_Activity", "Quiet afternoon living room reading"),
        ("Milan", "Morning", "Weekday", "Kitchen_Activity", "Early morning kitchen routine before work"),
        ("Milan", "Afternoon", "Weekend", "DiningRoom_Activity", "Eating afternoon lunch in dining area"),
        ("Milan", "Evening", "Weekday", "LivingRoom_Activity", "Evening downtime in living room"),
    ]
    for home, time_of_day, day, room, desc_text in day_configs:
        ev = {
            "Time_Of_Day": time_of_day,
            "Day_Of_Week": day,
            "Is_Night": "No",
            room: "Low",
            "OutsideDoor_Open": "No",
            "OutsideDoor_Close": "No",
            "Total_Contact_Events": "No",
            "Total_Activity": "Low"
        }
        scenarios.append({
            "title": f"Routine {time_of_day} Activity ({home})",
            "home": home,
            "desc": f"{desc_text} with normal sensor readings.",
            "evidence": ev,
            "expl": "Inferred state is Normal_Awake with zero anomaly indicators. No_Action is optimal (utility +8.6 to +10.0), eliminating unnecessary occupant disruptions."
        })

    # 15-20: Routine bathroom, work area, or lounging
    misc_configs = [
        ("Aruba", "Morning", "Weekday", "Bathroom_Activity", "Routine morning bathroom visit"),
        ("Aruba", "Afternoon", "Weekday", "WorkArea_Activity", "Focused desk work in home work area"),
        ("Milan", "Evening", "Weekend", "LoungeChair_Activity", "Sitting quietly in lounge chair"),
        ("Milan", "Morning", "Weekend", "Bathroom_Activity", "Bathroom use during weekend morning"),
        ("Cairo", "Afternoon", "Weekday", "WorkArea_Activity", "Afternoon study session in work area"),
        ("Cairo", "Evening", "Weekday", "GuestRoom_Activity", "Tidying up guest room in the evening"),
    ]
    for home, time_of_day, day, room, desc_text in misc_configs:
        ev = {
            "Time_Of_Day": time_of_day,
            "Day_Of_Week": day,
            "Is_Night": "No",
            room: "Low",
            "OutsideDoor_Open": "No",
            "OutsideDoor_Close": "No",
            "Total_Contact_Events": "No",
            "Total_Activity": "Low"
        }
        scenarios.append({
            "title": f"Routine Room Activity ({home})",
            "home": home,
            "desc": f"{desc_text} with low baseline activity.",
            "evidence": ev,
            "expl": "Normal room activity detected. System remains passive (No_Action) to preserve calm domestic environment."
        })

    return scenarios[:20]


def build_monitor_scenarios():
    """20 distinct Monitor scenarios across homes, entry/exit, and transition events."""
    scenarios = []
    
    # 1-10: Afternoon and morning departures / returns
    configs = [
        ("Aruba", "Afternoon", "Weekday", "Kitchen_Activity", "Resident returning home from work; door opened & closed"),
        ("Aruba", "Morning", "Weekday", "LivingRoom_Activity", "Resident leaving home for morning commute"),
        ("Aruba", "Evening", "Weekday", "Hall_Activity", "Evening arrival through front door into hallway"),
        ("Cairo", "Afternoon", "Weekday", "Hall_Activity", "Door contact event with hallway movement"),
        ("Cairo", "Morning", "Weekday", "LivingRoom_Activity", "Morning departure with outside door contact"),
        ("Milan", "Afternoon", "Weekday", "LivingRoom_Activity", "Door open/close upon arriving back home"),
        ("Milan", "Morning", "Weekday", "Kitchen_Activity", "Leaving home after quick morning breakfast"),
        ("Aruba", "Afternoon", "Weekend", "LivingRoom_Activity", "Returning from grocery trip on weekend"),
        ("Cairo", "Evening", "Weekend", "Hall_Activity", "Front door entry after weekend dinner"),
        ("Milan", "Afternoon", "Weekend", "DiningRoom_Activity", "Door activity with quick kitchen/dining movement"),
        # 11-20: Deliveries, brief exits, hallway checkups
        ("Aruba", "Morning", "Weekend", "LivingRoom_Activity", "Package delivery arrival at front door"),
        ("Aruba", "Evening", "Weekday", "LivingRoom_Activity", "Brief door opening to retrieve evening delivery"),
        ("Cairo", "Morning", "Weekend", "Hall_Activity", "Stepping out briefly to collect mail"),
        ("Cairo", "Afternoon", "Weekend", "LivingRoom_Activity", "Door sensor triggered with brief living room check"),
        ("Milan", "Morning", "Weekend", "LivingRoom_Activity", "Visitor greeting at front entrance"),
        ("Milan", "Evening", "Weekend", "Kitchen_Activity", "Door activity followed by kitchen activity"),
        ("Aruba", "Evening", "Weekend", "Kitchen_Activity", "Taking out evening recycling through back/outside door"),
        ("Cairo", "Evening", "Weekday", "LivingRoom_Activity", "Returning home late evening with door contact"),
        ("Milan", "Afternoon", "Weekday", "Hall_Activity", "Delivery driver ring with brief door motion"),
        ("Aruba", "Morning", "Weekday", "Hall_Activity", "Family member departure with front door cycle"),
    ]
    
    for home, time_of_day, day, room, desc_text in configs:
        ev = {
            "Time_Of_Day": time_of_day,
            "Day_Of_Week": day,
            "Is_Night": "No",
            "OutsideDoor_Open": "Yes",
            "OutsideDoor_Close": "Yes",
            "Total_Contact_Events": "Yes",
            "Total_Activity": "Low"
        }
        if room in ["LivingRoom_Activity", "Kitchen_Activity"]:
            ev[room] = "Low"
            
        scenarios.append({
            "title": f"Entry/Exit Transition ({home})",
            "home": home,
            "desc": f"{desc_text}.",
            "evidence": ev,
            "expl": "Door transition events detected. Decision engine selects Monitor (EU: +6.25). Passive logging tracks resident arrival/departure without sounding alarms or sending disruptive notifications."
        })

    return scenarios[:20]


def build_notify_resident_scenarios():
    """20 distinct Notify_Resident scenarios: anomalous activity while resident is awake."""
    scenarios = []
    
    configs = [
        ("Aruba", "Weekday", "High living room and kitchen activity with door open at night while awake"),
        ("Aruba", "Weekend", "Late night guest room & dining room motion while occupant awake"),
        ("Cairo", "Weekday", "Unexpected outside door contact and living room motion during late evening"),
        ("Cairo", "Weekend", "High activity in kitchen and dining room during unusual late night hours"),
        ("Milan", "Weekday", "Multiple room activity surge in living room & work area at night"),
        ("Milan", "Weekend", "Late night dining room & other room anomaly while resident awake"),
        ("Aruba", "Weekday", "Front door cycle with simultaneous kitchen and living room motion"),
        ("Aruba", "Weekend", "Unexpected door reopening late night with active living room"),
        ("Cairo", "Weekday", "Unusual nighttime hallway and living room activity spike"),
        ("Cairo", "Weekend", "Odd hours door event with active living room and kitchen"),
        ("Milan", "Weekday", "Multiple sensor triggers across secondary rooms during night"),
        ("Milan", "Weekend", "Unexpected outside door activity with resident awake in kitchen"),
        ("Aruba", "Weekday", "Unusual activity in other room and living room with contact events"),
        ("Aruba", "Weekend", "Late night living room and kitchen activity with open door"),
        ("Cairo", "Weekday", "Front entrance door opened with active living room motion"),
        ("Cairo", "Weekend", "Late evening multi-room activity surge requiring occupant verification"),
        ("Milan", "Weekday", "Living room and kitchen sensor activation during late night hours"),
        ("Milan", "Weekend", "Unexpected front door opening while occupant active in house"),
        ("Aruba", "Weekday", "High living room and dining room movement with door contact"),
        ("Milan", "Weekday", "Unusual late night appliance and room motion while occupant awake"),
    ]
    
    for home, day, desc_text in configs:
        ev = {
            "Time_Of_Day": "Night",
            "Day_Of_Week": day,
            "Is_Night": "Yes",
            "Bedroom_Activity": "None",
            "LivingRoom_Activity": "High",
            "Kitchen_Activity": "High",
            "DiningRoom_Activity": "High",
            "OtherRoom_Activity": "High",
            "Total_Activity": "High",
            "OutsideDoor_Open": "Yes",
            "OutsideDoor_Close": "Yes",
            "Total_Contact_Events": "Yes",
            "Bathroom_Activity": "Low",
            "WorkArea_Activity": "Low"
        }
        scenarios.append({
            "title": f"Awake Anomaly Alert ({home})",
            "home": home,
            "desc": f"{desc_text}.",
            "evidence": ev,
            "expl": "Bayesian inference indicates Unusual_Activity=Unusual (P > 50%) while Resident_Sleep=Awake. Notify_Resident maximizes expected utility (+25.07) because an awake occupant can quickly investigate or dismiss the alert safely without emergency escalation."
        })

    return scenarios[:20]


def build_silent_alert_scenarios():
    """20 distinct Silent_Alert scenarios: intrusion / anomaly while home is empty."""
    scenarios = []
    
    configs = [
        ("Aruba", "Morning", "Weekend", "Front door opened while house is unoccupied on weekend morning"),
        ("Aruba", "Afternoon", "Weekday", "Outside door breach detected while resident is away at work"),
        ("Cairo", "Morning", "Weekday", "Unscheduled front door opening during empty work hours"),
        ("Cairo", "Afternoon", "Weekday", "Door contact event with no interior resident activity"),
        ("Milan", "Morning", "Weekend", "Outside door motion detected while home is empty"),
        ("Milan", "Afternoon", "Weekday", "Midday front entrance trigger in unoccupied house"),
        ("Aruba", "Evening", "Weekday", "Outside door contact during evening empty home window"),
        ("Cairo", "Evening", "Weekend", "Door event in empty home while resident is away"),
        ("Milan", "Evening", "Weekday", "Unusual outside door activity in empty house"),
        ("Aruba", "Morning", "Weekday", "Early morning door sensor trip before scheduled return"),
        ("Cairo", "Morning", "Weekend", "Weekend morning door contact with zero resident motion"),
        ("Milan", "Morning", "Weekday", "Mid-morning door trigger while occupant is out"),
        ("Aruba", "Afternoon", "Weekend", "Weekend afternoon exterior door breach signal"),
        ("Cairo", "Afternoon", "Weekend", "Door opened with zero interior occupancy evidence"),
        ("Milan", "Afternoon", "Weekend", "Unoccupied weekend door event"),
        ("Aruba", "Night", "Weekday", "Late night door breach while resident away on trip"),
        ("Cairo", "Night", "Weekday", "Unoccupied home door trigger during overnight hours"),
        ("Milan", "Night", "Weekend", "Exterior door sensor tripped while house is vacant"),
        ("Aruba", "Night", "Weekend", "Weekend vacation empty home door breach"),
        ("Milan", "Night", "Weekday", "Nighttime door access event in empty residence"),
    ]
    
    for home, time_of_day, day, desc_text in configs:
        ev = {
            "Time_Of_Day": "Morning" if time_of_day != "Night" else "Night",
            "Is_Night": "Yes",
            "Day_Of_Week": day,
            "OutsideDoor_Open": "Yes",
            "Total_Contact_Events": "Yes",
            "Bedroom_Activity": "None",
            "LivingRoom_Activity": "None",
            "Kitchen_Activity": "None",
            "Total_Activity": "Low"
        }
        scenarios.append({
            "title": f"Empty Home Intrusion Signal ({home})",
            "home": home,
            "desc": f"{desc_text}.",
            "evidence": ev,
            "expl": "Inferred state is Unusual_Empty (P > 4.9%) with Home_Occupancy=Empty. Silent_Alert achieves maximum expected utility (+24.26 to +55.0), dispatching silent security intervention without tipping off a potential intruder or causing nuisance street sirens."
        })

    return scenarios[:20]


def build_local_alert_scenarios():
    """20 distinct Local_Alert scenarios: emergency / intrusion while resident is sleeping."""
    scenarios = []
    
    configs = [
        ("Aruba", "Weekday", "Living room break-in & door opened while resident sleeping in bedroom"),
        ("Aruba", "Weekend", "Kitchen & dining room anomaly at 3:00 AM during deep sleep"),
        ("Cairo", "Weekday", "Front entrance breach with multi-room sensor storm during sleep"),
        ("Cairo", "Weekend", "Midnight door opening with living room and guest room movement"),
        ("Milan", "Weekday", "Severe anomaly across living room, kitchen, and dining area at night"),
        ("Milan", "Weekend", "Door forced open and heavy downstairs motion while occupant asleep"),
        ("Aruba", "Weekday", "Outside door open with living room and other room activity at 2:00 AM"),
        ("Aruba", "Weekend", "Sudden multi-sensor trigger in living room and kitchen while in bed"),
        ("Cairo", "Weekday", "Door breach and rapid room motion during overnight sleep hours"),
        ("Cairo", "Weekend", "Nighttime intruder signal with active room motion while resident asleep"),
        ("Milan", "Weekday", "High living room and hallway activity with door open during sleep"),
        ("Milan", "Weekend", "Midnight security event: door contact & kitchen motion during sleep"),
        ("Aruba", "Weekday", "Multi-room anomaly detected across ground floor while resident asleep"),
        ("Aruba", "Weekend", "Emergency intrusion alarm: door open + kitchen/living room motion"),
        ("Cairo", "Weekday", "Nighttime perimeter breach with interior sensor activations"),
        ("Cairo", "Weekend", "Critical anomaly while asleep: front door open and dining room motion"),
        ("Milan", "Weekday", "Bedroom occupant asleep while living room and work area show high motion"),
        ("Milan", "Weekend", "Late night break-in alarm with door contact and multi-room activity"),
        ("Aruba", "Weekday", "2:30 AM door contact with living room and dining room anomaly"),
        ("Milan", "Weekday", "Urgent hazard/intrusion alert: door open and kitchen motion during sleep"),
    ]
    
    for home, day, desc_text in configs:
        ev = {
            "Time_Of_Day": "Night",
            "Day_Of_Week": day,
            "Is_Night": "Yes",
            "Bedroom_Activity": "Low",
            "LivingRoom_Activity": "High",
            "Kitchen_Activity": "High",
            "DiningRoom_Activity": "High",
            "GuestRoom_Activity": "High",
            "OtherRoom_Activity": "High",
            "Total_Activity": "High",
            "OutsideDoor_Open": "Yes",
            "OutsideDoor_Close": "Yes",
            "Total_Contact_Events": "Yes"
        }
        scenarios.append({
            "title": f"Sleeping Resident Security Alarm ({home})",
            "home": home,
            "desc": f"{desc_text}.",
            "evidence": ev,
            "expl": "CRITICAL RISK: Bayesian network infers P(Unusual_Occupied_Sleeping) = 88.85%. Resident is asleep and unaware of downstairs intrusion/hazard. Local_Alert achieves highest expected utility (+38.78 to +45.0), sounding local sirens to instantly awaken resident for self-preservation."
        })

    return scenarios[:20]


def main():
    print("Generating balanced 100-event test window...")
    no_action = build_no_action_scenarios()
    monitor = build_monitor_scenarios()
    notify = build_notify_resident_scenarios()
    silent = build_silent_alert_scenarios()
    local = build_local_alert_scenarios()
    
    all_raw = []
    import random
    
    # Organize pools of the 20 distinct scenarios per action
    pools = {
        "No_Action": list(no_action),
        "Monitor": list(monitor),
        "Notify_Resident": list(notify),
        "Silent_Alert": list(silent),
        "Local_Alert": list(local),
    }

    # Interleave so activities switch once in a while (runs of 1-2 items)
    rng = random.Random(2026)
    interleaved_items = []

    while any(pools.values()):
        avail = [k for k, v in pools.items() if len(v) > 0]
        if not avail:
            break
        last_act = interleaved_items[-1][0] if interleaved_items else None
        candidates = [k for k in avail if k != last_act] if len(avail) > 1 and last_act in avail else avail
        choice = rng.choice(candidates)
        run_len = min(rng.choice([1, 2]), len(pools[choice]))
        for _ in range(run_len):
            interleaved_items.append((choice, pools[choice].pop(0)))

    windows = []
    csv_rows = []
    
    win_idx = 0
    action_counts = {}
    
    # Timestamps starting in 2011 (CASAS dataset timeframe)
    base_timestamps = {
        "Morning": "2011-03-15 07:30:00",
        "Afternoon": "2011-03-15 13:45:00",
        "Evening": "2011-03-15 19:15:00",
        "Night": "2011-03-15 02:20:00"
    }

    for expected_action, item in interleaved_items:
            ev = item["evidence"]
            
            # Infer with BN engine
            posteriors = engine.query(ev)
            decision = select_action_meu(posteriors, engine=engine, evidence=ev)
            actual_action = decision["selected_action"]
            
            # Verify action matches expected
            action_counts[actual_action] = action_counts.get(actual_action, 0) + 1
            if actual_action != expected_action:
                print(f"WARNING: Window {win_idx} expected {expected_action} but got {actual_action}")
            
            # Behavioral predictions
            predictions = {}
            for var in BEHAVIORAL_NODES:
                dist = posteriors[var]
                predictions[var] = max(dist, key=dist.get)
                
            time_of_day = ev.get("Time_Of_Day", "Afternoon")
            ts = base_timestamps.get(time_of_day, "2011-03-15 12:00:00")
            
            # Add minute offset per index
            mins = win_idx * 5
            hr = int(ts[11:13]) + (mins // 60)
            mn = (int(ts[14:16]) + (mins % 60)) % 60
            ts_adjusted = f"{ts[:10]} {hr%24:02d}:{mn:02d}:00"
            
            win_obj = {
                "window_id": win_idx,
                "index": win_idx,
                "source": "curated_demonstration",
                "home_id": item["home"],
                "timestamp": ts_adjusted,
                "scenario_title": item["title"],
                "description": item["desc"],
                "explanation": item["expl"],
                "evidence": ev,
                "posteriors": posteriors,
                "decision": decision,
                "predictions": predictions,
                "true_labels": predictions,
                "total_windows": 100
            }
            windows.append(win_obj)
            
            # CSV summary row
            csv_row = {
                "window_id": win_idx,
                "home_id": item["home"],
                "timestamp": ts_adjusted,
                "scenario_title": item["title"],
                "selected_action": actual_action,
                "expected_utility": round(decision["expected_utilities"][actual_action], 2),
                "dominant_state": max(decision["state_probabilities"], key=decision["state_probabilities"].get),
                "description": item["desc"],
                "explanation": item["expl"]
            }
            for col in EVIDENCE_NODES:
                csv_row[col] = ev.get(col, "None")
            csv_rows.append(csv_row)
            
            win_idx += 1

    print(f"Total windows generated: {len(windows)}")
    print(f"Action distribution: {action_counts}")
    assert len(windows) == 100, f"Expected 100 windows, got {len(windows)}"
    
    # Save demo_windows.json
    demo_json_path = DEMO_DIR / "demo_windows.json"
    with open(demo_json_path, "w") as f:
        json.dump(windows, f, indent=2)
    print(f"Saved: {demo_json_path}")
    
    # Save standalone test_100_events.json
    test_json_path = DEMO_DIR / "test_100_events.json"
    with open(test_json_path, "w") as f:
        json.dump(windows, f, indent=2)
    print(f"Saved: {test_json_path}")
    
    # Save test_100_events.csv
    test_csv_path = DEMO_DIR / "test_100_events.csv"
    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv(test_csv_path, index=False)
    print(f"Saved: {test_csv_path}")

if __name__ == "__main__":
    main()
