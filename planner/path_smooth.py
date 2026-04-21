import heapq
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

os.makedirs("outputs", exist_ok=True)


def _h(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def astar(grid, risk_map, start, goal, lambda_weight=0.0):
    rows, cols = grid.shape
    open_heap = [(_h(start, goal), 0.0, 0.0, start[0], start[1])]
    best = {start: (0.0, 0.0)}
    came_from = {start: None}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        f, g_d, g_r, r, c = heapq.heappop(open_heap)
        node = (r, c)

        if node == goal:
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        if node in best:
            bd, br = best[node]
            if g_d > bd + 1e-9 or g_r > br + 1e-9:
                continue

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue

            nb = (nr, nc)
            step_d = math.sqrt(dr * dr + dc * dc)
            step_r = float(risk_map[nr, nc])
            new_gd = g_d + step_d
            new_gr = g_r + step_r

            new_f = new_gd + lambda_weight * new_gr + _h(nb, goal)
            if nb in best:
                bd, br = best[nb]
                old_f = bd + lambda_weight * br + _h(nb, goal)
                if new_f >= old_f - 1e-9:
                    continue

            best[nb] = (new_gd, new_gr)
            came_from[nb] = node
            heapq.heappush(open_heap, (new_f, new_gd, new_gr, nr, nc))

    return None


def build_risk_map(grid, alpha=1.5):
    free_mask = (grid == 0).astype(float)
    dist = distance_transform_edt(free_mask)
    risk = np.exp(-alpha * dist)
    return np.clip(risk, 0, 1)


def build_standard_grid():
    g = np.ones((100, 100), dtype=int)
    g[10:21, 0:100] = 0
    g[70:81, 0:100] = 0
    g[20:71, 70:81] = 0
    g[40:51, 0:81]  = 0
    g[0, :]  = 1;  g[99, :] = 1
    g[:, 0]  = 1;  g[:, 99] = 1
    g[43:45, 20:22] = 1
    g[43:45, 40:42] = 1
    g[45:47, 58:60] = 1
    g[42:44, 70:72] = 1
    return g

np.random.seed(0)
grid_std = build_standard_grid()
risk_std  = build_risk_map(grid_std, alpha=1.5)

START = (15, 5)
GOAL  = (75, 75)

raw_path = astar(grid_std, risk_std, START, GOAL, lambda_weight=8.0)
if raw_path is None:
    raise RuntimeError("A* found no path on standard grid.")

raw_rows = np.array([p[0] for p in raw_path], dtype=float)
raw_cols = np.array([p[1] for p in raw_path], dtype=float)

# Subsample raw path to key waypoints only (direction changes)
def extract_waypoints(path, min_dist=6):
    """Keep only points where direction changes, spaced >= min_dist apart."""
    if len(path) < 3:
        return path
    wpts = [path[0]]
    for i in range(1, len(path) - 1):
        pr, pc = path[i-1]
        cr, cc = path[i]
        nr, nc = path[i+1]
        d1 = (cr-pr, cc-pc)
        d2 = (nr-cr, nc-cc)
        dist_from_last = abs(cr - wpts[-1][0]) + abs(cc - wpts[-1][1])
        if d1 != d2 and dist_from_last >= min_dist:
            wpts.append(path[i])
    wpts.append(path[-1])
    return wpts

def fillet_path(waypoints, radius=12):
    """
    Returns a list of (row, col) float points representing:
    - Exact straight segments between corners (only endpoints stored)
    - Smooth Bézier arc at each corner
    No resampling is applied — straight lines stay perfectly straight.
    """
    pts = [np.array(w, dtype=float) for w in waypoints]
    result = []

    for i in range(len(pts)):
        if i == 0:
            result.append(tuple(pts[0]))
            continue
        if i == len(pts) - 1:
            result.append(tuple(pts[-1]))
            continue

        prev = pts[i - 1]
        curr = pts[i]
        nxt  = pts[i + 1]

        v1 = curr - prev
        v2 = nxt  - curr
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 < 1e-6 or len2 < 1e-6:
            result.append(tuple(curr))
            continue

        u1 = v1 / len1
        u2 = v2 / len2

        # Fillet radius, capped to 40% of each segment length
        r = min(radius, len1 * 0.4, len2 * 0.4)

        arc_start = curr - u1 * r   # where straight segment ends
        arc_end   = curr + u2 * r   # where next straight segment begins

        # Straight segment endpoint — matplotlib draws a perfect line to here
        result.append(tuple(arc_start))

        # Quadratic Bézier arc: control point is the corner itself
        n_arc = max(20, int(r * 3))
        for t in np.linspace(0, 1, n_arc):
            b = ((1-t)**2 * arc_start
                 + 2*(1-t)*t * curr
                 + t**2 * arc_end)
            result.append(tuple(b))

        # arc_end becomes the start of the next straight segment
        result.append(tuple(arc_end))

    return result


waypoints = extract_waypoints(raw_path, min_dist=6)
filleted  = fillet_path(waypoints, radius=12)

smooth_path_r = [p[0] for p in filleted]
smooth_path_c = [p[1] for p in filleted]

print(f"[Part 1] Raw path    : {len(raw_path)} cells")
print(f"[Part 1] Smoothed    : {len(filleted)} points")

H, W = 100, 100
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 7))
fig1.patch.set_facecolor("#1a1a2e")

wall_disp = np.where(grid_std == 1, 0.0, 0.2)

risk_rgba = np.zeros((H, W, 4), dtype=float)
risk_rgba[:, :, 0] = risk_std
risk_rgba[:, :, 3] = risk_std * 0.3

for ax, title, path_r, path_c, color, lw, annotation in [
    (axes1[0], "Raw A* Path",
     raw_rows, raw_cols, "#e63946", 1.5, None),
    (axes1[1], "Smoothed Path (Corner Fillet)",
     smooth_path_r, smooth_path_c, "#00f5d4", 2.5,
     "Physically realisable trajectory"),
]:
    ax.imshow(wall_disp, cmap="gray", vmin=0, vmax=1,
              origin="upper", interpolation="nearest")
    ax.imshow(risk_rgba, origin="upper", interpolation="nearest")
    ax.plot(path_c, path_r, color=color, linewidth=lw, zorder=5,
            solid_capstyle="round")
    ax.plot(START[1], START[0], "o", color="#06d6a0", markersize=9,
            markeredgecolor="white", markeredgewidth=1.2, zorder=6, label="Start")
    ax.plot(GOAL[1],  GOAL[0],  "*", color="#ffd166", markersize=13,
            markeredgecolor="white", markeredgewidth=1.0, zorder=6, label="Goal")
    if annotation:
        ax.text(50, 93, annotation, ha="center", va="center", fontsize=9,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#4895ef",
                          alpha=0.85, edgecolor="white", linewidth=0.8))
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#0d1b2a")
    ax.legend(loc="upper left", fontsize=8,
              facecolor="#0d1b2a", edgecolor="gray", labelcolor="white")

fig1.suptitle("MediNav Path Smoothing — Corner Fillet Post-Processing",
              color="white", fontsize=13, fontweight="bold", y=0.98)
fig1.text(0.5, 0.01,
          f"Raw path: {len(raw_path)} cells  |  Smoothed path: {len(filleted)} points",
          ha="center", va="bottom", fontsize=9.5, color="white",
          bbox=dict(boxstyle="round,pad=0.5", facecolor="#16213e",
                    edgecolor="#4895ef", linewidth=1.2))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig1.savefig("outputs/path_smoothed.png", dpi=150, bbox_inches="tight",
             facecolor=fig1.get_facecolor())
plt.close(fig1)
print("Saved -> outputs/path_smoothed.png")


# ---------------------------------------------------------------------------
# Part 2 — Corner Fillet Smoothing across 5 Hospital Layouts
# ---------------------------------------------------------------------------

def add_border(g):
    g[0, :] = 1;  g[99, :] = 1
    g[:, 0] = 1;  g[:, 99] = 1
    return g

def layout_standard():
    g = np.ones((100, 100), dtype=int)
    g[10:21, 0:100] = 0
    g[70:81, 0:100] = 0
    g[20:71, 70:81] = 0
    g[40:51, 0:81]  = 0
    g[43:45, 20:22] = 1
    g[43:45, 40:42] = 1
    g[45:47, 58:60] = 1
    g[42:44, 70:72] = 1
    return add_border(g)

def layout_tjunction():
    g = np.ones((100, 100), dtype=int)
    g[45:56, 0:100] = 0
    g[0:46,  45:56] = 0
    g[45:100,45:56] = 0
    g[10:21, 0:46]  = 0
    g[75:86, 0:46]  = 0
    g[10:21, 55:100]= 0
    g[75:86, 55:100]= 0
    return add_border(g)

def layout_lshape():
    g = np.ones((100, 100), dtype=int)
    g[5:96,  5:20]  = 0   # left vertical arm
    g[80:96, 5:96]  = 0   # bottom horizontal arm
    g[5:21,  5:96]  = 0   # top horizontal arm
    return add_border(g)

def layout_zigzag():
    g = np.ones((100, 100), dtype=int)
    g[5:21,  5:65]  = 0   # top horizontal
    g[5:56,  52:66] = 0   # first vertical drop
    g[44:60, 20:66] = 0   # middle horizontal
    g[44:96, 20:34] = 0   # second vertical drop
    g[80:96, 20:95] = 0   # bottom horizontal
    return add_border(g)

def layout_cross():
    g = np.ones((100, 100), dtype=int)
    g[45:56, 5:96]  = 0   # horizontal corridor
    g[5:96,  45:56] = 0   # vertical corridor
    return add_border(g)

layouts = [
    ("Standard Ward",  layout_standard(),  (15, 5),  (75, 75)),
    ("T-Junction",     layout_tjunction(), (12, 5),  (12, 92)),
    ("L-Shape",        layout_lshape(),    (12, 10), (88, 88)),
    ("Zigzag",         layout_zigzag(),    (12, 10), (88, 60)),
    ("Cross",          layout_cross(),     (50, 8),  (8, 50)),
]

fig2, axes2 = plt.subplots(2, 5, figsize=(26, 11))
fig2.patch.set_facecolor("#1a1a2e")

for col, (name, g, s, goal) in enumerate(layouts):
    risk = build_risk_map(g, alpha=1.5)
    raw  = astar(g, risk, s, goal, lambda_weight=8.0)

    ax_raw = axes2[0, col]
    ax_smo = axes2[1, col]

    wall_disp = np.where(g == 1, 0.0, 0.2)
    r_rgba = np.zeros((100, 100, 4), dtype=float)
    r_rgba[:, :, 0] = risk
    r_rgba[:, :, 3] = risk * 0.25

    for ax in (ax_raw, ax_smo):
        ax.imshow(wall_disp, cmap="gray", vmin=0, vmax=1,
                  origin="upper", interpolation="nearest")
        ax.imshow(r_rgba, origin="upper", interpolation="nearest")
        ax.plot(s[1],    s[0],    "o", color="#06d6a0", markersize=8,
                markeredgecolor="white", markeredgewidth=1.0, zorder=6)
        ax.plot(goal[1], goal[0], "*", color="#ffd166", markersize=12,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        ax.set_xlim(-0.5, 99.5); ax.set_ylim(99.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#0d1b2a")

    if raw is None:
        print(f"WARNING: no path found for '{name}'")
        ax_raw.set_title(name, color="white", fontsize=9, fontweight="bold", pad=5)
        ax_smo.set_title("(no path)", color="#e63946", fontsize=9, pad=5)
        continue

    # --- Top row: raw A* path ---
    ax_raw.plot([c for _, c in raw], [r for r, _ in raw],
                color="#e63946", linewidth=2.0, zorder=5,
                solid_capstyle="round", solid_joinstyle="round")
    ax_raw.set_title(name, color="white", fontsize=10, fontweight="bold", pad=5)

    # --- Bottom row: smoothed path ---
    wpts     = extract_waypoints(raw, min_dist=6)
    filleted = fillet_path(wpts, radius=12)
    sm_r = [p[0] for p in filleted]
    sm_c = [p[1] for p in filleted]

    ax_smo.plot(sm_c, sm_r, color="#00f5d4", linewidth=2.2, zorder=5,
                solid_capstyle="round", solid_joinstyle="round")
    ax_smo.set_title(f"Smoothed  ({len(filleted)} pts)",
                     color="#00f5d4", fontsize=8, pad=5)

    print(f"[{name}]  raw={len(raw)} cells   smoothed={len(filleted)} pts")

# Row labels on left edge
for row_ax, label in zip(axes2[:, 0], ["Raw A* Path", "Corner Fillet Smoothed"]):
    row_ax.set_ylabel(label, color="white", fontsize=10,
                      fontweight="bold", labelpad=8)
    row_ax.yaxis.label.set_visible(True)

fig2.suptitle(
    "MediNav Corner Fillet Smoothing — 5 Hospital Layouts",
    color="white", fontsize=14, fontweight="bold", y=1.01
)
plt.tight_layout(pad=1.5)
fig2.savefig("outputs/path_smooth_multi.png", dpi=150, bbox_inches="tight",
             facecolor=fig2.get_facecolor())
plt.close(fig2)
print("Saved -> outputs/path_smooth_multi.png")
