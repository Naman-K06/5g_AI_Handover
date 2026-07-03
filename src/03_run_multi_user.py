
import numpy as np
import pandas as pd
import os
import sys
import json
import logging
import matplotlib.pyplot as plt
import osmnx as ox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics_engine import get_network_state
from config import (
    NUM_UES, CELL_CAPACITY, SIM_DURATION, TIME_STEP,
    HYSTERESIS_DB, TTT_LIMIT, PING_PONG_THRESHOLD,
    HOF_SINR_THRESHOLD, RLF_SINR_THRESHOLD, RLF_TIME_LIMIT,
    GRAPH_FILE, HEX_FILE, L3_FILTER_COEFFICIENT,
    UE_SPEED_MIN_KMH, UE_SPEED_MAX_KMH, LOG_FILE, LOG_LEVEL
)

# Setup logging
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SAVE_PATH = os.getenv("CURRENT_RUN_PATH", "outputs/Default_Run")
os.makedirs(SAVE_PATH, exist_ok=True)

# -------------------
# UE AGENT CLASS
# -------------------
class UserEquipment:
    def __init__(self, ue_id, route_data):
        self.ue_id        = ue_id
        self.points       = np.array(route_data['route_points'])
        self.cum_dist     = np.array(route_data['cum_dist'])
        self.total_len    = route_data['total_length']
        self.current_dist = 0
        self.speed_kmh    = np.random.uniform(UE_SPEED_MIN_KMH, UE_SPEED_MAX_KMH)

        self.serving_cell = -1
        self.prev_shadow  = None
        self.filtered_rsrp = None
        self.k_filter     = L3_FILTER_COEFFICIENT

        self.candidate_cell     = -1
        self.ttt_timer          = 0
        self.ho_count           = 0
        self.ping_pong_count    = 0
        self.hof_count          = 0
        self.rlf_count          = 0
        self.rlf_timer          = 0
        self.last_serving_cell  = -1
        self.time_since_last_ho = 999.0
        self.history            = []

    def apply_l3_filter(self, new_rsrps):
        if self.filtered_rsrp is None:
            self.filtered_rsrp = new_rsrps
        else:
            self.filtered_rsrp = (
                (1 - self.k_filter) * self.filtered_rsrp + self.k_filter * new_rsrps
            )
        return self.filtered_rsrp

    def move(self, time_step):
        self.current_dist       += (self.speed_kmh / 3.6) * time_step
        self.time_since_last_ho += time_step
        return self.current_dist < self.total_len

    def get_coords(self):
        ux = np.interp(self.current_dist, self.cum_dist, self.points[:, 0])
        uy = np.interp(self.current_dist, self.cum_dist, self.points[:, 1])
        return np.array([ux, uy])


# -------------------
# CORE SIMULATION
# -------------------
if not os.path.exists(GRAPH_FILE):
    logger.error(f"Graph file not found: {GRAPH_FILE}")
    raise FileNotFoundError(f"OSM network not found: {GRAPH_FILE}")

if not os.path.exists(HEX_FILE):
    logger.error(f"Hex file not found: {HEX_FILE}")
    raise FileNotFoundError(f"Hex centers file not found: {HEX_FILE}")

logger.info(f"Loading OSM network from {GRAPH_FILE}")
# Load G once — reused for plotting later (avoids double load)
G             = ox.load_graphml(GRAPH_FILE)
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

logger.info(f"Loading base station coordinates from {HEX_FILE}")
bs_coords = pd.read_csv(HEX_FILE).values
num_cells = len(bs_coords) * 3

logger.info(f"Initializing {NUM_UES} UEs...")
ues = []
for i in range(NUM_UES):
    route_file = f"data/routes/route_{i+1}.json"
    try:
        if not os.path.exists(route_file):
            logger.warning(f"Route file missing: {route_file}")
            continue
        with open(route_file, 'r') as f:
            route_data = json.load(f)
        required = {'route_points', 'cum_dist', 'total_length'}
        if not required.issubset(route_data):
            logger.warning(f"Route {i+1} has invalid structure, skipping")
            continue
        ues.append(UserEquipment(i, route_data))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse route {i+1}: {e}")
    except Exception as e:
        logger.error(f"Failed to load route {i+1}: {e}")

if len(ues) == 0:
    logger.error("No UEs could be loaded from route files")
    raise RuntimeError("No valid routes found")

logger.info(f"Loaded {len(ues)} UEs successfully")
print(f"Simulating {len(ues)} UEs with variable speeds "
      f"({UE_SPEED_MIN_KMH}-{UE_SPEED_MAX_KMH} km/h)...")
print(f"  Steps: {SIM_DURATION}  |  Time step: {TIME_STEP}s  |  "
      f"Total sim time: {SIM_DURATION * TIME_STEP:.0f}s")

PRINT_INTERVAL = max(1, SIM_DURATION // 10)  # print ~10 progress lines

for t in range(SIM_DURATION):
    cell_counts  = np.zeros(num_cells)
    for ue in ues:
        if ue.serving_cell != -1:
            cell_counts[ue.serving_cell] += 1
    current_loads = np.clip(cell_counts / CELL_CAPACITY, 0, 1.0)

    for ue in ues:
        if not ue.move(TIME_STEP):
            continue
        pos = ue.get_coords()

        state          = get_network_state(pos, bs_coords, ue.prev_shadow, current_loads)
        ue.prev_shadow = state['shadowing']
        rsrps          = ue.apply_l3_filter(state['rsrp'])

        if ue.serving_cell == -1:
            ue.serving_cell = np.argmax(rsrps)
            continue

        # --- 3GPP A3 EVENT LOGIC & HOF ---
        best_rsrp_idx = np.argmax(rsrps)
        if (best_rsrp_idx != ue.serving_cell and
                rsrps[best_rsrp_idx] > rsrps[ue.serving_cell] + HYSTERESIS_DB):
            if best_rsrp_idx == ue.candidate_cell:
                ue.ttt_timer += TIME_STEP
            else:
                ue.candidate_cell = best_rsrp_idx
                ue.ttt_timer      = TIME_STEP

            if ue.ttt_timer >= TTT_LIMIT:
                target_sinr = state['sinr'][best_rsrp_idx]
                if target_sinr < HOF_SINR_THRESHOLD:
                    ue.hof_count += 1
                    ue.ttt_timer, ue.candidate_cell = 0, -1
                else:
                    if (best_rsrp_idx == ue.last_serving_cell and
                            ue.time_since_last_ho <= PING_PONG_THRESHOLD):
                        ue.ping_pong_count += 1
                    ue.last_serving_cell    = ue.serving_cell
                    ue.serving_cell         = ue.candidate_cell
                    ue.ho_count            += 1
                    ue.time_since_last_ho   = 0
                    ue.ttt_timer, ue.candidate_cell = 0, -1
        else:
            ue.ttt_timer, ue.candidate_cell = 0, -1

        # --- Radio Link Failure (RLF) Monitoring ---
        serving_sinr = state['sinr'][ue.serving_cell]
        if serving_sinr < RLF_SINR_THRESHOLD:
            ue.rlf_timer += TIME_STEP
            if ue.rlf_timer >= RLF_TIME_LIMIT:
                ue.rlf_count   += 1
                ue.rlf_timer    = 0
                ue.serving_cell = np.argmax(rsrps)
        else:
            ue.rlf_timer = 0

        ue.history.append({
            "x":         pos[0],
            "y":         pos[1],
            "serving":   ue.serving_cell,
            "speed":     ue.speed_kmh,
            "ping_pong": ue.ping_pong_count,
            "hof":       ue.hof_count,
            "rlf":       ue.rlf_count,
        })

    # Progress print every ~10% of steps
    if (t + 1) % PRINT_INTERVAL == 0 or t == SIM_DURATION - 1:
        active = sum(1 for ue in ues if ue.current_dist < ue.total_len)
        ho_so_far = sum(ue.ho_count for ue in ues)
        print(f"  t={t+1:4d}/{SIM_DURATION}  active UEs={active:3d}  "
              f"HOs so far={ho_so_far}")
        logger.info(f"Simulation progress t={t+1}/{SIM_DURATION}, active={active}, HOs={ho_so_far}")

# -------------------
# KPI SUMMARY
# -------------------
UE_LOGS_DIR = os.path.join(SAVE_PATH, "ue_logs")
os.makedirs(UE_LOGS_DIR, exist_ok=True)

for ue in ues:
    if ue.history:
        ue_df              = pd.DataFrame(ue.history)
        ue_df['ue_id']     = ue.ue_id
        ue_df['speed_kmh'] = ue.speed_kmh
        ue_df.to_csv(os.path.join(UE_LOGS_DIR, f"ue_{ue.ue_id}.csv"), index=False)

total_ho  = sum(ue.ho_count          for ue in ues)
total_pp  = sum(ue.ping_pong_count   for ue in ues)
total_hof = sum(ue.hof_count         for ue in ues)
total_rlf = sum(ue.rlf_count         for ue in ues)

print(f"\n--- Global Simulation Results ---")
print(f"  Total Handovers : {total_ho}  |  Ping-Pongs: {total_pp}")
print(f"  Total HOF       : {total_hof}  |  Total RLF: {total_rlf}")
print(f"  Individual logs saved to: {UE_LOGS_DIR}")

# -------------------
# ENHANCED RESEARCH-GRADE MAPPING
# (reuses G / edges_gdf already loaded above — no second file load)
# -------------------
def draw_detailed_site(ax, x, y, radius):
    angles = np.linspace(np.pi / 6, 2 * np.pi + np.pi / 6, 7)
    ax.plot(x + radius * np.cos(angles), y + radius * np.sin(angles),
            linewidth=1.0, color='#FF8C00', alpha=0.3, zorder=2)
    for angle in [0, 120, 240]:
        rad = np.radians(angle)
        ax.plot([x, x + (radius * 0.8) * np.cos(rad)],
                [y, y + (radius * 0.8) * np.sin(rad)],
                color='red', alpha=0.2, linestyle='--', linewidth=0.8, zorder=3)

with open("data/map_metadata.json", 'r') as f:
    meta = json.load(f)

print("Rendering High-Resolution Mobility Map...")
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 16), facecolor='#0D1117')
ax.set_facecolor('#0D1117')

# Roads (use edges_gdf already loaded at top)
for geom in edges_gdf.geometry:
    xs, ys = geom.xy
    xs_p = (np.array(xs) - meta['minx']) * meta['scale_factor'] + meta['offset_x']
    ys_p = (np.array(ys) - meta['miny']) * meta['scale_factor'] + meta['offset_y']
    ax.plot(xs_p, ys_p, color='#1B2129', linewidth=0.7, alpha=0.4, zorder=1)

# Base Stations
hex_radius = meta['ISD'] / np.sqrt(3)
for i, (bx, by) in enumerate(bs_coords):
    draw_detailed_site(ax, bx, by, hex_radius)
    ax.scatter(bx, by, c='red', marker='h', s=80, edgecolors='white', zorder=5)
    ax.text(bx, by - 120, f"Site {i}", color='gray', fontsize=8, ha='center', zorder=6)

# UE Trajectories
cmap = plt.get_cmap('tab20')
for i, ue in enumerate(ues):
    if not ue.history:
        continue
    h     = pd.DataFrame(ue.history)
    color = cmap(i % 20)
    ax.plot(h['x'], h['y'], color=color, linewidth=2, alpha=0.8,
            label=f"UE {ue.ue_id+1} ({ue.speed_kmh:.0f} km/h)", zorder=10)
    ho_events = h[h['serving'].diff() != 0]
    ax.scatter(ho_events['x'], ho_events['y'], color=color, marker='x', s=40, zorder=11)

ax.set_xlim(0, 5000)
ax.set_ylim(0, 5000)
ax.axis('off')
place_name = meta.get('PLACE', 'Chandigarh')
ax.set_title(f"3GPP Baseline Mobility: {place_name} Digital Twin",
             color='white', fontsize=18, pad=20)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small',
          title="UE Speeds & IDs", facecolor='#111', edgecolor='#444',
          labelcolor='white', ncol=2)

plt.savefig(os.path.join(SAVE_PATH, "baseline_mobility_map.png"),
            dpi=300, bbox_inches='tight')
print(f"Baseline Map saved to: {SAVE_PATH}")