# MediNav
## Risk-Aware SLAM Navigation for Autonomous Hospital Supply Delivery

**Course:** BITS F327 — Artificial Intelligence for Robotics  
**Institution:** BITS Pilani, Goa Campus  
**Supervisor:** Dr. Prasad Vinayak Patil  

---

## Team

| Name | ID | Module |
|---|---|---|
| Aaditya H Vernenker | 2023A7PS1027G | GraphSLAM, SLAM uncertainty integration, dynamic replanning |
| Ojas Sahu | 2023A3PS0861G | Risk map generation, Dijkstra comparison, visualisation |
| Ayush Chopra | 2023A7PS1018G | Risk-aware A* planner, parameter sensitivity analysis |
| Hansel Cisil Sunny | 2023A4PS0271G | Synthetic environment, Monte Carlo evaluation, path smoothing |

---

## Project Overview

MediNav finds the *safest* path through a hospital ward — not just the shortest one. It integrates GraphSLAM-based localisation with a risk-aware A* planner that penalises proximity to walls and areas of high localisation uncertainty.

```
Sensor Data → [GraphSLAM] → Occupancy Grid + Uncertainty Map
                                    ↓
                           [Risk Module] → Risk Field
                                    ↓
                           [Risk-Aware A*] → Safe Path
                                    ↓
                           [Corner Fillet] → Smooth Trajectory
```

---

## Key Results (30-trial Monte Carlo, seed=42)

| Metric | Standard A* | Risk-Aware A* | Improvement |
|---|---|---|---|
| Safety violation rate | 49.1% | 3.1% | **93.7% reduction** |
| Mean min clearance (cells) | 2.25 | 3.42 | **+52%** |
| Mean safety violations | 29.7 | 1.9 | **93.6% reduction** |
| Path length overhead | 0% | 2.9% | cost of safety |

SLAM RMSE: 0.821 m (dead reckoning) → 0.055 m (GraphSLAM) — **93.3% improvement**

---

## Setup

```bash
pip install numpy matplotlib scipy
```

No ROS. No Gazebo. Pure Python.

---

## How to Run

Run all scripts from the project root directory:

```bash
python slam/graphslam.py              # SLAM trajectory + occupancy grid
python risk/risk_map.py               # Risk field generation
python risk/visualize_risk.py         # 4-panel risk visualisation
python planner/path_compare.py        # Path comparison figure
python planner/lambda_analysis.py     # Lambda sensitivity sweep
python planner/param_sensitivity.py   # Alpha/w1 sensitivity sweep
python planner/path_smooth.py         # Corner fillet smoothing
python planner/dynamic_replan.py      # Dynamic obstacle replanning
python simulation/evaluate.py         # 30-trial Monte Carlo evaluation
python simulation/evaluate_extended.py  # 3-planner comparison
```

All outputs are saved to `outputs/`.

---

## Folder Structure

```
MediNav/
├── slam/
│   ├── graphslam.py           # GraphSLAM with per-pose covariance export
│   └── occupancy_grid.py      # Occupancy grid from SLAM trajectory
├── risk/
│   ├── risk_map.py            # Risk field: obstacle decay + SLAM uncertainty
│   └── visualize_risk.py      # 4-panel risk visualisation
├── planner/
│   ├── astar_risk.py          # Risk-aware A* implementation
│   ├── path_compare.py        # Side-by-side path comparison
│   ├── lambda_analysis.py     # Safety weight sensitivity
│   ├── param_sensitivity.py   # Alpha and w1/w2 sensitivity
│   ├── path_smooth.py         # Corner fillet post-processor
│   └── dynamic_replan.py      # 5-cell lookahead replanning
├── simulation/
│   ├── synthetic_env.py       # Hospital ward environment generator
│   ├── evaluate.py            # 30-trial Monte Carlo evaluation
│   └── evaluate_extended.py   # Dijkstra vs A* vs Risk-Aware A*
└── outputs/                   # All generated figures and data files
```
