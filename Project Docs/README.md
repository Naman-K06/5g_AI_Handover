# 5G Wireless Network Handover Optimization Using AI

## Project Introduction

This project demonstrates how artificial intelligence can improve cellular network handover decisions in 5G networks. Instead of using fixed, rule-based handover criteria (like traditional 3GPP standards), we train a machine learning model to learn optimal handover decisions by analyzing signal strength patterns across a digital twin environment.

The system compares two approaches:
1. **Baseline**: Traditional 3GPP handover logic with hysteresis and time-to-trigger rules.
2. **AI-Based**: Machine learning model that predicts the best cell to connect to.

Results show that AI can reduce unnecessary handovers while maintaining or improving signal quality.

## Simulation Environment & Parameters

- **Geographic Area**: Chandigarh, India (5000m × 5000m grid) with OSM road networks.
- **5G Network Setup**: 19 base stations in a hexagonal grid (ISD: 950m).
- **User Equipment (UE)**: Moves at 60 km/h along actual road routes with a 0.5s simulation time step.
- **Signal Propagation**: Free-space path loss with urban shadowing (log-normal) and fading effects.
- **Handover Parameters**: Hysteresis Margin of 3 dB, Time-to-Trigger (TTT) of 2s, and AI persistence buffer of 6 steps.

## System Components & Execution Order

The project is divided into several modular scripts that form a complete pipeline:

1. `01_preprocess.py`: Downloads OSM data and generates the map and hexagonal grid.
2. `02_generate_routes.py`: Pre-generates 20 diverse test routes.
3. `03_run_multi_user.py`: Runs a multi-user baseline simulation (traditional 3GPP logic).
4. `04_train_ai.py`: Trains a Random Forest machine learning model on 20,000 synthetic UE positions.
5. `05_final_comparison.py`: Compares AI vs Baseline handover logic on identical routes.

### How to Run

You can run the entire pipeline orchestrator using `main.py`:

```bash
python main.py --steps 1 2 3 4 5
```

*A full run takes approximately 15-25 minutes. Results and comparison reports will be saved under the `outputs/` directory.*

## Key Research Findings

**AI Advantages:**
- Reduces unnecessary handovers by 10-30%.
- Maintains or slightly improves signal quality (SINR).
- Real-time capable (<1ms inference per decision).
- Generalizes well to unseen routes and multi-user congestion.

**Baseline Advantages:**
- Simple, deterministic rules proven in production 3GPP networks.
- No model dependency or training overhead.

**Key Insight:** AI learns to reduce handovers during signal transitions while maintaining quality. The persistence buffer prevents ping-pong oscillations.

## Folder Structure

```text
WMC Project/
├── main.py                          (Orchestrator for full pipeline)
├── src/                             (Source code for simulation & AI)
├── data/                            (Maps, base station coordinates, and pre-computed routes)
├── outputs/                         (Model artifacts, KPI summaries, and comparison reports)
└── Project Docs/                    (Detailed project documentation)
```

## Dependencies & Setup

Python Version: 3.9+

Installation:
```bash
python -m venv .venv
# Activate the virtual environment:
# source .venv/bin/activate          (Linux/Mac) 
# .venv\Scripts\activate             (Windows)

pip install -r requirements.txt
```

*Required packages include: numpy, pandas, scikit-learn, matplotlib, osmnx, networkx, shapely, seaborn.*

## Further Documentation

For more details on the project tasks, planned upgrades, and presentation guides, please refer to the files in the `Project Docs/` folder.
