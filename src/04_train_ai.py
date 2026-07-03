
import numpy as np
import pandas as pd
import os
import sys
import json
import pickle
import logging
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from physics_engine import get_network_state
from config import (
    NUM_TRAJECTORIES, STEPS_PER_TRAJECTORY, WINDOW_SIZE,
    AREA_W, AREA_H, SPEED_MIN, SPEED_MAX,
    AI_DATA_DIR, MODEL_FILE, HEX_FILE, ROUTE_DIR,
    N_ESTIMATORS, MAX_DEPTH, N_JOBS, RANDOM_STATE,
    TEST_SIZE, RLF_SINR_THRESHOLD,
    LOG_FILE, LOG_LEVEL,
)

# ==================== LOGGING ====================
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==================== HO-MINIMIZATION ORACLE PARAMETERS ====================
AI_STABILITY_MARGIN = 5.0    # dB gain required before switching (high = conservative)
MIN_STAY_SAMPLES    = 15     # Force minimum dwell before any switch
CRITICAL_SINR       = -10.0  # Emergency switch threshold (dB) — prevents RLF

# ==================== SETUP ====================
os.makedirs(AI_DATA_DIR, exist_ok=True)

if not os.path.exists(HEX_FILE):
    raise FileNotFoundError(f"HEX file missing: {HEX_FILE}")

bs_coords = pd.read_csv(HEX_FILE).values
num_cells = len(bs_coords) * 3

print(f"Loaded {len(bs_coords)} sites → {num_cells} cells")

# ==================== LOAD ROUTE WAYPOINTS FOR REALISTIC SEEDING ====================
# Using real OSM-route start positions closes the distribution gap between
# straight-line training walks and the road-constrained evaluation routes.
route_files      = sorted(glob.glob(os.path.join(ROUTE_DIR, "route_*.json")))
route_start_pts  = []
for rf in route_files:
    try:
        with open(rf) as f:
            rd = json.load(f)
        pts = rd.get("route_points", [])
        if len(pts) >= STEPS_PER_TRAJECTORY:
            route_start_pts.append(np.array(pts))
    except Exception:
        pass

use_real_routes = len(route_start_pts) >= 10
if use_real_routes:
    print(f"Using {len(route_start_pts)} real OSM routes as training position seeds.")
    logger.info(f"Training seeded from {len(route_start_pts)} real routes")
else:
    print("No real routes found — falling back to random-walk trajectories.")
    logger.warning("Training falling back to random walks; run 02_generate_routes.py first.")

print(f"Generating {NUM_TRAJECTORIES} trajectories for HO-minimisation training...")
logger.info(f"Starting training data generation: {NUM_TRAJECTORIES} trajectories")

X_train_data = []
y_labels     = []

# ==================== DATA GENERATION ====================
for traj_idx in range(NUM_TRAJECTORIES):

    if use_real_routes:
        # Pick a random real route and walk along its waypoints
        route_pts  = route_start_pts[traj_idx % len(route_start_pts)]
        max_start  = max(0, len(route_pts) - STEPS_PER_TRAJECTORY - WINDOW_SIZE - 1)
        start_idx  = np.random.randint(0, max_start + 1)
        waypoints  = route_pts[start_idx: start_idx + STEPS_PER_TRAJECTORY + WINDOW_SIZE]
        speed_scalar = np.random.uniform(SPEED_MIN, SPEED_MAX)
    else:
        # Fallback: straight-line random walk
        pos         = np.random.uniform(0, AREA_W, size=2)
        velocity    = np.random.uniform(SPEED_MIN, SPEED_MAX, size=2)
        speed       = np.linalg.norm(velocity)
        direction   = velocity / speed if speed > 0 else np.array([1.0, 0.0])
        speed_scalar = np.random.uniform(SPEED_MIN, SPEED_MAX)
        velocity    = direction * speed_scalar
        waypoints   = None

    temp_shadow   = None
    current_loads = np.full(num_cells, 0.5)
    traj_buffer   = []

    for step_i in range(STEPS_PER_TRAJECTORY + WINDOW_SIZE):
        if use_real_routes:
            pos = waypoints[step_i] if step_i < len(waypoints) else waypoints[-1]
        else:
            pos = np.clip(pos + velocity * 0.5, 0, AREA_W)

        state       = get_network_state(pos, bs_coords, temp_shadow, current_loads)
        temp_shadow = state["shadowing"]

        features = list(state["rsrp"]) + list(state["sinr"]) + [speed_scalar]
        traj_buffer.append({"features": features, "sinr": state["sinr"]})

    # ==================== STABILITY-FIRST ORACLE LABELING ====================
    current_serving = int(np.argmax(traj_buffer[0]["sinr"]))
    stay_counter    = 0

    for i in range(WINDOW_SIZE, len(traj_buffer)):
        window = np.array(
            [traj_buffer[j]["features"] for j in range(i - WINDOW_SIZE, i)]
        ).flatten()
        X_train_data.append(window)

        current_sinrs = np.array(traj_buffer[i - 1]["sinr"])
        best_cell     = int(np.argmax(current_sinrs))
        current_sinr  = current_sinrs[current_serving]
        best_sinr     = current_sinrs[best_cell]
        gain          = best_sinr - current_sinr

        # Rule 1 — Emergency: SINR below RLF floor → immediate switch
        if current_sinr < RLF_SINR_THRESHOLD:
            current_serving = best_cell
            stay_counter    = 0

        # Rule 2 — Enforce minimum dwell time
        
        elif stay_counter < MIN_STAY_SAMPLES:
          stay_counter += 1

        # Rule 3 — Switch only if gain is very large AND SINR is degraded
        elif gain > AI_STABILITY_MARGIN and current_sinr < CRITICAL_SINR:
            current_serving = best_cell
            stay_counter    = 0

        # Rule 4 — Default: stay
        else:
            stay_counter += 1

        y_labels.append(current_serving)

    if (traj_idx + 1) % max(1, NUM_TRAJECTORIES // 10) == 0:
        print(f"  Trajectory {traj_idx+1}/{NUM_TRAJECTORIES} complete "
              f"({len(X_train_data)} samples so far)")

# ==================== TRAIN MODEL ====================
X = np.array(X_train_data)
y = np.array(y_labels)

print(f"\nDataset: {len(X)} samples, {X.shape[1]} features, {num_cells} classes")
logger.info(f"Dataset ready: {len(X)} samples")

unique, counts = np.unique(y, return_counts=True)
rare_labels    = unique[counts < 2]
if len(rare_labels) > 0:
    logger.warning(f"Rare labels (< 2 samples), disabling stratify: {rare_labels}")
    stratify_arg = None
else:
    stratify_arg = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=stratify_arg,
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Fitting Random Forest ({N_ESTIMATORS} trees, max_depth={MAX_DEPTH}, "
      f"n_jobs={N_JOBS})...")

clf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    n_jobs=N_JOBS,
    random_state=RANDOM_STATE,
)
clf.fit(X_train, y_train)

# ==================== EVALUATION ====================
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)

y_pred   = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

# Weighted metrics (handles class imbalance correctly)
f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Weighted F1   : {f1:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print("\nPer-class Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save metrics to CSV for paper
import pandas as pd, os
ml_metrics = pd.DataFrame([{
    "Accuracy": round(test_acc, 4),
    "Weighted_F1": round(f1, 4),
    "Precision": round(precision, 4),
    "Recall": round(recall, 4)
}])
os.makedirs(AI_DATA_DIR, exist_ok=True)
ml_metrics.to_csv(os.path.join(AI_DATA_DIR, "ml_metrics.csv"), index=False)
print(f"ML metrics saved.")

# ==================== SAVE MODEL ====================
with open(MODEL_FILE, "wb") as f:
    pickle.dump(clf, f)

print(f"HO-Optimised AI model saved to {MODEL_FILE}")
logger.info(f"Model saved to {MODEL_FILE}")