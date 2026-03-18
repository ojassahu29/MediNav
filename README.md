# MediNav 🏥🤖
### Autonomous Contactless Medical Supply Delivery Robot Using Risk-Aware SLAM
 
**Course:** BITS F327 — Artificial Intelligence for Robotics  
**Institution:** BITS Pilani, Goa Campus  
**Supervisor:** Dr. Prasad Vinayak Patil  
**Submission Deadline:** 22 April 2026
 
---
 
## Team
 
| Name | ID | Module | Branch |
|------|----|--------|--------|
| Aaditya H Vernenker | 2023A7PS1027G | SLAM (GraphSLAM + Occupancy Grid) | `slam` |
| Ojas Sahu | 2023A3PS0861G | Risk Map + Visualization | `risk` |
| Ayush Chopra | 2023A7PS1018G | A* Planner + Path Comparison | `planner` |
| Hansel Cisil Sunny | 2023A4PS0271G | Synthetic Environment + Evaluation | `simulation` |
 
---
 
## Project Overview
 
MediNav is a Python-based simulation of a risk-aware autonomous navigation system for hospital delivery robots. Instead of finding the *shortest* path, MediNav finds the *safest* path — staying away from walls, equipment, and areas where the robot might lose track of its position.
 
The system has four components that work together:
 
```
Sensor Data → [SLAM] → Map → [Risk Module] → Risk Map → [A* Planner] → Safe Path → Robot Moves
```
 
1. **SLAM** builds a map of the hospital while tracking the robot's position
2. **Risk Module** assigns danger scores to areas of the map (high near walls, low in open corridors)
3. **A\* Planner** finds a path that minimises both distance *and* risk
4. **Evaluation** runs both planners 20 times and compares safety statistics
 
---
 
## Folder Structure
 
```
MediNav/
├── README.md               ← this file
├── requirements.txt        ← Python dependencies
├── outputs/                ← all generated plots/images go here (auto-created)
├── slam/                   ← Aaditya's module
│   ├── graphslam.py
│   └── occupancy_grid.py
├── risk/                   ← Ojas's module
│   ├── risk_map.py
│   └── visualize_risk.py
├── planner/                ← Ayush's module
│   ├── astar_risk.py
│   └── path_compare.py
└── simulation/             ← Hansel's module
    ├── synthetic_env.py
    └── evaluate.py
```
 
---
 
## Setup — Run This Once
 
```bash
# Install dependencies (run this once on your machine)
pip install numpy matplotlib scipy
```
 
No ROS. No Gazebo. Everything runs as plain Python.
 
---
 
## How to Run
 
```bash
# Each script is independent — run them in any order
python slam/graphslam.py          # generates outputs/slam_trajectory.png
python slam/occupancy_grid.py     # generates outputs/occupancy_grid.png
python risk/risk_map.py           # generates outputs/risk_map.npy
python risk/visualize_risk.py     # generates outputs/risk_visualization.png
python planner/astar_risk.py      # prints path length comparison
python planner/path_compare.py    # generates outputs/path_comparison.png
python simulation/evaluate.py     # prints stats table + outputs/evaluation_results.png
```
 
---
 
---
 
# 📋 GitHub Guide for Team Members
 
**Read this carefully before touching any code.**
 
---
 
## The Branch System
 
Think of the repo like a Google Doc with version history, but safer.
 
- **`main`** = the clean, working version. **You never commit directly to main.**
- **Your branch** = your personal workspace. You do all your work here.
- When your code works, Aaditya reviews it and **merges** it into main.
 
```
main ──────────────────────────────────── (clean, working code)
       ↑ merge         ↑ merge
slam ──────────        risk ─────────────  (each person's workspace)
```
 
---
 
## Step-by-Step: First Time Setup (everyone does this once)
 
### 1. Install Git
- **Windows:** Download from https://git-scm.com/download/win → install with all defaults
- **Mac:** Open Terminal, type `git --version` → if not installed, it will prompt you to install
- **Linux:** `sudo apt install git`
 
### 2. Set your identity (one time only)
Open a terminal and run:
```bash
git config --global user.name "Your Full Name"
git config --global user.email "your_github_email@gmail.com"
```
 
### 3. Clone the repository
This downloads the project to your computer:
```bash
git clone https://github.com/Aadityahv/MediNav.git
cd MediNav
```

 
### 4. Create your personal branch
Replace `yourname` with your actual branch name from the table above (`risk`, `planner`, or `simulation`):
```bash
git checkout -b yourname
```
Example for Ojas:
```bash
git checkout -b risk
```
 
You are now on your own branch. Confirm it worked:
```bash
git branch
```
You should see `* risk` (or your branch name) with a star next to it.
 
---
 
## Day-to-Day Workflow (do this every time you work on the project)
 
### Before you start coding — pull latest changes
```bash
git checkout yourname-branch    # make sure you're on your branch
git pull origin main            # get any updates from main
```
 
### After you write/edit code — save your progress
```bash
git add .                       # stage all changed files
git commit -m "describe what you did"   # save a snapshot
git push origin yourname-branch         # upload to GitHub
```
 
**Good commit messages** (be specific):
```bash
git commit -m "Add risk heatmap generation using distance transform"
git commit -m "Fix path comparison plot — paths now visually distinct"
git commit -m "Add evaluation loop with 20 Monte Carlo trials"
```
 
**Bad commit messages** (useless):
```bash
git commit -m "stuff"
git commit -m "fixed"
git commit -m "update"
```
 
---
 
## What Each Person Needs to Do
 
---
 
### 🧠 Ojas Sahu — `risk` branch
 
**Your files:** `risk/risk_map.py` and `risk/visualize_risk.py`
 
**Your output:** `outputs/risk_visualization.png` — a 3-panel figure showing occupancy grid, risk heatmap, and overlay.
 
**Steps:**
```bash
# After cloning and creating your branch:
git checkout -b risk
 
# Write your scripts in the risk/ folder
# Test them:
python risk/risk_map.py
python risk/visualize_risk.py
 
# Save your work:
git add risk/
git commit -m "Add risk map computation and 3-panel visualization"
git push origin risk
 
# Then message Aaditya: "risk branch ready to merge"
```
 
**What your scripts must produce:**
- `risk_map.py` → prints min/max/mean risk values, saves `outputs/risk_map.npy`
- `visualize_risk.py` → saves `outputs/risk_visualization.png` (3 subplots: occupancy | risk heatmap | overlay)
 
---
 
### 🗺️ Ayush Chopra — `planner` branch
 
**Your files:** `planner/astar_risk.py` and `planner/path_compare.py`
 
**Your output:** `outputs/path_comparison.png` — side-by-side showing standard A* (red, hugs walls) vs risk-aware A* (blue, stays centre).
 
**Steps:**
```bash
git checkout -b planner
 
# Write your scripts in the planner/ folder
python planner/astar_risk.py
python planner/path_compare.py
 
git add planner/
git commit -m "Add risk-aware A* planner and path comparison visualization"
git push origin planner
# Message Aaditya when done
```
 
**What your scripts must produce:**
- `astar_risk.py` → prints path lengths for both planners on a test grid
- `path_compare.py` → saves `outputs/path_comparison.png` showing two distinct paths
 
---
 
### 📊 Hansel Cisil Sunny — `simulation` branch
 
**Your files:** `simulation/synthetic_env.py` and `simulation/evaluate.py`
 
**Your output:** A printed statistics table + `outputs/evaluation_results.png`
 
**Steps:**
```bash
git checkout -b simulation
 
python simulation/synthetic_env.py
python simulation/evaluate.py
 
git add simulation/
git commit -m "Add hospital environment generator and Monte Carlo evaluation"
git push origin simulation
# Message Aaditya when done
```
 
**What your scripts must produce:**
- `synthetic_env.py` → prints grid summary (free cells, landmark count)
- `evaluate.py` → prints formatted stats table, saves `outputs/evaluation_results.png`
 
---
 
## For Aaditya — Merging Branches into Main
 
When a team member says their branch is ready:
 
1. Go to **github.com/[your-username]/MediNav**
2. Click **Pull requests** → **New pull request**
3. Set: base = `main`, compare = `risk` (or whoever's branch)
4. Click **Create pull request** → scroll down → **Merge pull request**
5. Done. Their code is now in main.
 
OR via terminal:
```bash
git checkout main
git merge risk         # merge Ojas's branch
git push origin main
```
 
---
 
## Deadlines
 
| Date | Who | What |
|------|-----|------|
| 18 Mar (today) | Aaditya | Push folder structure + slam/ to main |
| 19 Mar (Wed) | Ojas | `risk` branch ready |
| 19 Mar (Wed) | Hansel | `simulation` branch ready |
| 20 Mar (Thu) | Ayush | `planner` branch ready |
| 20 Mar (Thu) | Aaditya | Merge all branches, final README |
| 22 Mar 23:59 | Aaditya | Submit report (link repo in report) |
 
---
 
## Common Problems & Fixes
 
**"I accidentally committed to main"**
```bash
git checkout -b yourname-branch    # create your branch from current state
git checkout main
git reset --hard HEAD~1            # undo the commit on main
git push origin main --force       # update GitHub
```
Then message Aaditya immediately.
 
**"My push was rejected"**
```bash
git pull origin yourname-branch    # get latest version first
# fix any conflicts, then:
git push origin yourname-branch
```
 
**"I don't know which branch I'm on"**
```bash
git branch                         # star = current branch
git status                         # also shows current branch at top
```
 
**"I want to see what changed before committing"**
```bash
git diff                           # shows line-by-line changes
git status                         # shows which files changed
```
 
---
 
## Questions?
 
Message Aaditya on WhatsApp before touching anything you're unsure about. Do not guess with Git — it is easier to ask first than to untangle a broken repo.
