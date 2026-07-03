# System Architecture & Design
**Project:** 5G Wireless Network Handover Optimization Using AI

---

## 1. Project Overview

In cellular networks, a "handover" (or handoff) is the process of transferring an active connection of a User Equipment (UE) from one Base Station (BS) to another. Traditional 5G networks rely on fixed, rule-based logic defined by 3GPP standards. However, in complex urban environments, these static rules often lead to sub-optimal connections, delayed handovers, or "ping-pong" effects (rapidly bouncing between two cells).

This project creates a **Digital Twin** of an urban 5G network to simulate realistic UE movement and signal propagation. It then builds an **AI-driven Handover Engine** to autonomously decide the optimal cell connection, comparing its performance against standard 3GPP baseline rules.

---

## 2. Core Components & Architecture

The project is modularized into four primary domains:

### A. Geographic & Mobility Environment (The Digital Twin)
- **Map Generation:** Uses `OSMnx` to fetch real-world street graphs of Chandigarh, India.
- **Simulation Area:** A 5000m × 5000m defined bounding box.
- **UE Mobility:** Simulates a user driving along actual road networks. The UE calculates precise coordinate updates based on an assumed fixed speed (e.g., 60 km/h) over a specific time step (0.5s).

### B. Network Topology
- **Hexagonal Grid:** Generates 19 base stations arranged in a standard hexagonal grid, typical of cellular network planning.
- **Inter-Site Distance (ISD):** 950 meters between base stations.
- **Sectorization:** Models omnidirectional or sectored antennas (future upgrade roadmap).

### C. Physics & Signal Propagation Engine
To train the AI realistically, the signal simulation must reflect real-world physics. The engine calculates the **RSRP (Reference Signal Received Power)** using:
1. **Free-Space Path Loss (FSPL):** Signal degradation over distance.
2. **Urban Shadowing:** Log-normal shadowing to simulate obstacles like buildings blocking the line-of-sight (LoS).
3. **Fast Fading:** Rayleigh/Rician fading effects representing multipath interference.
4. **Interference & SINR (Planned):** Signal-to-Interference-plus-Noise Ratio modeling to evaluate network congestion.

### D. Decision Engines
The framework simultaneously runs two decision engines for fair A/B comparison:
- **Baseline Engine:** Emulates standard telecommunication switches.
- **AI Engine:** A Machine Learning classifier.

---

## 3. How the Handover Logic Works

### The Baseline Logic (Traditional 3GPP)
The telecom standard uses two primary parameters to prevent bouncing back and forth between cells:
*   **Hysteresis Margin (e.g., 3 dB):** The target cell must be stronger than the current cell by at least this margin to be considered.
*   **Time-to-Trigger (TTT) (e.g., 2s):** The target cell must consistently stay above the hysteresis margin for the duration of the TTT.

**Workflow:**
1. UE reports signal strength (Measurement Report).
2. Network checks if `RSRP_target > RSRP_serving + Hysteresis`.
3. If true, starts a timer.
4. If the condition holds for the entirety of the TTT, the handover executes.

### The AI Logic (Machine Learning)
Instead of relying on timers and rigid margins, the AI Engine looks at the *pattern* of all available signals simultaneously.

*   **Model:** Random Forest Classifier (or Deep Neural Network).
*   **Inputs (Features):**
    *   Current RSRP of all 19 cells.
    *   Signal Gradients (Is the signal from cell 4 increasing or decreasing?).
    *   Current connected cell ID.
*   **Outputs:** The predicted best cell ID to connect to at this exact millisecond.
*   **Ping-Pong Mitigation (Persistence Buffer):** Instead of a static TTT timer, the AI uses a sliding window (e.g., 6 steps). A handover is only physically triggered if the AI consistently predicts a new cell over the majority of the buffer, heavily punishing ping-pong oscillations.

---

## 4. Execution Pipeline & Data Flow

The project operates via a linear pipeline orchestrated by `main.py`.

### Step 1: Preprocessing (`01_preprocess.py`)
- Fetches OpenStreetMap data.
- Generates base station coordinates.
- Pre-computes static parameters and saves them to `/data`.

### Step 2: Route Generation (`02_generate_routes.py`)
- Samples random Origin-Destination pairs on the map.
- Computes shortest paths using Dijkstra's algorithm.
- Saves multiple testing routes (`route_1.csv`, `route_2.csv`) to ensure the AI generalizes.

### Step 3: Simulation & Data Collection (`03_run_multi_user.py`)
- Runs simulated UE along the routes using Baseline Handover logic.
- Logs millions of data points: `[Timestamp, X, Y, Serving_Cell, RSRP_1 ... RSRP_19]`.
- This data serves as the ground truth training environment.

### Step 4: AI Training (`04_train_ai.py`)
- Ingests the baseline simulation data.
- Formats data into ML features (Signal strengths, top-N cells, variance).
- Trains a Machine Learning model (Random Forest).
- Exports the serialized model to the `/outputs` folder.

### Step 5: Final Evaluation & Comparison (`05_final_comparison.py`)
- Ingests an unseen route.
- Processes the route simultaneously through both the Baseline rules and the AI model.
- Generates comparative statistics (Total handovers, average signal quality, ping-pong events).
- Outputs statistical graphs and significance testing (T-Tests).

---

## 5. Key Performance Indicators (KPIs)

When the final comparison executes, it measures success using the following metrics:

1. **Total Handover Count:** Fewer handovers are better (reduces signaling overhead and latency).
2. **Average RSRP / SINR:** Higher is better. It guarantees that reducing handovers didn't come at the cost of dropping the call or poor internet speeds.
3. **Ping-Pong Rate:** Measures how many times the UE swaps between the same two cells within a 5-second window.
4. **Inference Latency:** Ensures the AI can make a decision in <1ms, making it viable for real-time Edge network deployment.

---

## 6. Future Expansion & Roadmap

The architecture is designed to scale. Future modules that fit into this design include:

- **Variable Mobility:** Adding realistic traffic patterns, stoplights, and variable speeds (Acceleration/Deceleration).
- **Multi-User Congestion:** Modeling hundreds of UEs simultaneously to dynamically balance load across cells, requiring the AI to factor in *Cell Utilization* alongside signal strength.
- **LoS/NLoS Algorithms (Blockage Modeling):** Integrating actual building footprints from OSM to block signals instantly if a UE turns a corner.
- **Deep Learning:** Exchanging Random Forest for LSTMs (Long Short-Term Memory) to predict UE trajectory and make proactive handovers before signal degrades.