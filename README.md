# AI-Optimized 5G Handover Using Digital Twins

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green.svg)]()
[![5G Networks](https://img.shields.io/badge/Domain-5G%20Wireless-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Active%20Research-success.svg)]()

## Overview

This project presents an **AI-driven handover optimization framework for 5G cellular networks** using a realistic **Digital Twin** of an urban environment.

Traditional 3GPP handover algorithms rely on static rules such as:

- Hysteresis Margin
- Time-to-Trigger (TTT)

Although these methods are robust, they often lead to:

- Excessive handovers
- Ping-pong effects
- Delayed cell transitions
- Sub-optimal signal quality

This project investigates whether **Machine Learning can learn better handover decisions** directly from network conditions and user mobility patterns.

The framework compares:

| Method | Description |
|---------|-------------|
| Baseline | Traditional 3GPP Handover Logic |
| AI-Based | Random Forest-based Intelligent Handover Engine |

---

# Research Objectives

- Build a realistic Digital Twin of a 5G network.
- Simulate user movement on real road networks.
- Train an AI model to predict optimal handover decisions.
- Reduce unnecessary handovers.
- Minimize ping-pong events.
- Maintain or improve signal quality.
- Evaluate statistical significance of AI improvements.

---

# Digital Twin Environment

### Geographic Area
- Chandigarh, India
- 5000 m × 5000 m simulation area

### Network Topology
- 19 Base Stations
- Hexagonal cellular layout
- Inter-Site Distance (ISD): 950 m

### User Mobility
- Real road networks from OpenStreetMap
- Shortest path routing
- User speed: 60 km/h
- Simulation timestep: 0.5 seconds

---

# System Architecture

```text
OpenStreetMap
       │
       ▼
Preprocessing
       │
       ▼
Route Generation
       │
       ▼
Baseline Simulation
       │
       ▼
Training Dataset
       │
       ▼
Machine Learning Model
       │
       ▼
AI Handover Engine
       │
       ▼
Performance Comparison
```

---

# Signal Propagation Model

The simulation includes realistic radio propagation effects:

- Free Space Path Loss (FSPL)
- Log-Normal Shadowing
- Urban Fading Effects
- Multi-route mobility

Planned extensions:

- SINR Modeling
- LoS/NLoS propagation
- Multipath fading
- Interference modeling

---

# AI Handover Engine

## Machine Learning Model

- Algorithm: Random Forest Classifier
- Features:
  - RSRP from all cells
  - Signal gradients
  - Serving Cell ID
  - Signal variance
  - Top-N strongest cells

## Prediction Output

```text
Input Features
        ↓
Random Forest Model
        ↓
Predicted Best Cell ID
```

To avoid ping-pong effects, the AI engine uses a:

- Persistence Buffer
- Majority Voting Strategy

instead of fixed TTT timers.

---

# Project Pipeline

## Step 1 — Preprocessing

```bash
python src/01_preprocess.py
```

- Download OSM map
- Generate base stations
- Save environment data

---

## Step 2 — Route Generation

```bash
python src/02_generate_routes.py
```

- Generate test routes
- Create reproducible trajectories

---

## Step 3 — Baseline Simulation

```bash
python src/03_run_multi_user.py
```

- Run traditional 3GPP handover
- Generate training dataset

---

## Step 4 — AI Training

```bash
python src/04_train_ai.py
```

- Train Random Forest model
- Save trained model

---

## Step 5 — Final Comparison

```bash
python src/05_final_comparison.py
```

Compare:

- Total Handovers
- Average RSRP
- Ping-Pong Events
- Inference Time

---

# Run Entire Pipeline

```bash
python main.py --steps 1 2 3 4 5
```

---

# Repository Structure

```text
5g_AI_Handover/
│
├── src/
│   ├── 01_preprocess.py
│   ├── 02_generate_routes.py
│   ├── 03_run_multi_user.py
│   ├── 04_train_ai.py
│   ├── 05_final_comparison.py
│   └── config.py
│
├── outputs/
│   └── new_routes/
│
├── Project Docs/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Naman-K06/5g_AI_Handover.git
cd 5g_AI_Handover
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate:

### Windows

```bash
.venv\Scripts\activate
```

### Linux/Mac

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Key Performance Indicators (KPIs)

| Metric | Goal |
|---------|------|
| Total Handovers | Lower |
| Ping-Pong Rate | Lower |
| Average RSRP | Higher |
| Average SINR | Higher |
| Inference Time | < 1 ms |

---

# Current Results

Experiments have already been performed using:

- 10 User Equipment (UE)
- 100 User Equipment (UE)

with all outputs generated and saved for analysis. :contentReference[oaicite:2]{index=2}

---

# Research Roadmap

### Completed
- Digital Twin Generation
- Baseline Handover Simulation
- AI Handover Engine
- Route Reproducibility
- Multi-Route Testing

### In Progress
- Statistical Significance Testing
- Feature Engineering
- Multi-User Congestion Modeling
- Cross-Validation on Unseen Routes

### Future Work
- Deep Learning (LSTM)
- Reinforcement Learning
- Graph Neural Networks
- Real-world Network Trace Validation
- Multi-User Load Balancing

The roadmap also prioritizes cross-validation, statistical testing, and publication-quality plots for research dissemination. :contentReference[oaicite:3]{index=3}

---

# Research Paper

**Proposed Title**

> AI-Optimized Handover Decisions in 5G Networks:
> A Machine Learning Approach Using Digital Twins

---

# Author

**Naman Kamboj**

Electronics and Communication Engineering

Research Areas:

- 5G/6G Wireless Networks
- Artificial Intelligence in Telecommunications
- Digital Twins
- Network Optimization
- Embedded Systems

---

# License

This project is released under the MIT License.

---

# Citation

```bibtex
@software{kamboj2026aihandover,
  author = {Naman Kamboj},
  title = {AI-Optimized 5G Handover Using Digital Twins},
  year = {2026},
  url = {https://github.com/Naman-K06/5g_AI_Handover}
}
```
