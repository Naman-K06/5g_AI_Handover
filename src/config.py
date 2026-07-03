
import os
import datetime

# ==================== PHYSICS PARAMETERS ====================
# 3GPP TR 38.901 UMa Constants
FC_GHZ = 3.5                    # Frequency (GHz)
H_BS = 25.0                     # Base station height (m)
H_UT = 1.5                      # User terminal height (m)
TX_POWER_DBM = 46.0             # Transmit power (dBm)
NOISE_FLOOR_DBM = -104.0        # Noise floor (dBm)
C_LIGHT = 3e8                   # Speed of light (m/s)

# Antenna Pattern Parameters
ANTENNA_GAIN_MAX = 8.0          # Maximum antenna gain (dBi)
PHI_3DB = 65                    # Horizontal 3dB beamwidth (degrees)
THETA_3DB = 65                  # Vertical 3dB beamwidth (degrees)
A_MAX = 30                      # Maximum attenuation (dB)
DOWNTILT_DEG = 10.0             # Mechanical downtilt (degrees)

# Path Loss Shadowing
SHADOWING_STD_LOS = 4.0         # LOS shadowing std dev (dB)
SHADOWING_STD_NLOS = 6.0        # NLOS shadowing std dev (dB)
SHADOWING_CORRELATION = 0.82    # Autocorrelation coefficient (0-1)

# ==================== 3GPP HANDOVER PARAMETERS ====================
HYSTERESIS_DB = 3.0             # Handover hysteresis (dB)
TTT_LIMIT = 1.5                 # Time-to-trigger (seconds)
PING_PONG_THRESHOLD = 3.0       # Ping-pong detection window (seconds)

# ==================== RELIABILITY THRESHOLDS ====================
HOF_SINR_THRESHOLD = -6.0       # Handover Failure limit (dB)
RLF_SINR_THRESHOLD = -12.0      # Radio Link Failure limit (dB)
RLF_TIME_LIMIT = 2.0            # RLF trigger duration (seconds) — T310 timer

# ==================== AI TRAINING PARAMETERS ====================
# Trajectory Generation
NUM_TRAJECTORIES = 2000          # Training trajectories (600 old val)
STEPS_PER_TRAJECTORY = 100       # Steps per trajectory (50 old val)
WINDOW_SIZE = 20                # Sliding window size for features (10 old val)

# Speed Range (m/s)
SPEED_MIN = 5.5                 # Minimum speed (m/s) ~20 km/h
SPEED_MAX = 27.7                # Maximum speed (m/s) ~100 km/h

# Safety & Robustness
HOF_TRAINING_LIMIT = -4.0       # Training safety margin (dB above HOF threshold)
CONGESTION_HIGH_THRESHOLD = 0.75  # High load threshold
CONGESTION_LOW_THRESHOLD = 0.6    # Low load threshold (for traffic steering)

# Random Forest Hyperparameters
N_ESTIMATORS = 300              # Number of trees (200 old val)
MAX_DEPTH = 30                  # Maximum tree depth (20 old val)
CPU_COUNT = os.cpu_count() or 1
N_JOBS = min(10, CPU_COUNT)     # Respects available CPUs
RANDOM_STATE = 42               # Reproducibility seed

# ==================== SIMULATION PARAMETERS ====================
# Multi-User Simulation
NUM_UES = 100                # Number of simultaneous UEs
CELL_CAPACITY = 25              # Max UEs per cell
SIM_DURATION = 400              # Simulation time steps
TIME_STEP = 0.5                 # Time step (seconds)

# UE Speed Range
UE_SPEED_MIN_KMH = 20           # Minimum UE speed (km/h)
UE_SPEED_MAX_KMH = 100          # Maximum UE speed (km/h)

# L3 Filtering (for RSRP smoothing)
L3_FILTER_COEFFICIENT = 0.3     # Kalman filter gain (higher = more responsive)

# ==================== ROUTE GENERATION ====================
NUM_ROUTES = 100                # Number of routes to generate
MIN_ROUTE_LEN = 2000            # Minimum route length (meters)
# Pipeline aborts route step if fewer than this fraction succeed
MIN_ROUTE_SUCCESS_RATE = 0.8    # 80% minimum success rate

# ==================== GEOMETRY ====================
AREA_W = 5000                   # Simulation area width (meters)
AREA_H = 5000                   # Simulation area height (meters)
ISD = 950                       # Inter-Site Distance (meters)
# Correct for a flat-topped hex: R = ISD/sqrt(3)
HEX_RADIUS = ISD / (3 ** 0.5)

# ==================== DATA PATHS ====================
GRAPH_FILE    = "data/road_network.graphml"
HEX_FILE      = "data/hex_centers.csv"
METADATA_FILE = "data/map_metadata.json"
ROUTE_DIR     = "data/routes"
MAPS_DIR      = "data/maps"

# ==================== RUN PATH ====================
_timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M")
_default_path = os.path.join("outputs", "new_routes", f"Standalone_Run_{_timestamp}")

# Matches os.environ["CURRENT_RUN_PATH"] set by main.py (if used)
SAVE_PATH = os.environ.get("CURRENT_RUN_PATH", _default_path)
os.makedirs(SAVE_PATH, exist_ok=True)

# ==================== MODEL & DATA OUTPUT ====================
AI_DATA_DIR = "outputs/AI_Data"
MODEL_FILE  = os.path.join(AI_DATA_DIR, "ai_handover_model.pkl")

# ==================== LOGGING ====================
# Defined once, inside per-run SAVE_PATH for proper isolation.
LOG_FILE  = os.path.join(SAVE_PATH, "pipeline.log")
LOG_LEVEL = "DEBUG"             # DEBUG, INFO, WARNING, ERROR, CRITICAL

# ==================== CROSS-VALIDATION ====================
CV_FOLDS  = 5                   # K-fold cross-validation folds
TEST_SIZE = 0.2                 # Test set fraction

# Feature Validation
VALIDATE_FEATURES = True        # Enable feature dimension checking
VALIDATE_ROUTES   = True        # Enable route validation

# ==================== PLACE INFO ====================
PLACE = "Chandigarh, India"

