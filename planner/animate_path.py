import subprocess
import sys

try:
    import PIL
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])

import os
import math
import heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import distance_transform_edt

def main():
    # 1. Inline the hospital grid
    grid = np.ones((100, 100), dtype=int)
    grid[10:21, 0:100] = 0
    grid[70:81, 0:100] = 0
    grid[20:71, 70:81] = 0
    grid[40:51, 0:81] = 0

    grid[0, :] = 1
    grid[99, :] = 1
    grid[:, 0] = 1
    grid[:, 99] = 1

    grid[43:45, 20:22] = 1
    grid[43:45, 40:42] = 1
    grid[45:47, 58:60] = 1
    grid[42:44, 70:72] = 1

    # 2. Risk map
    free_mask = (grid == 0).astype(float)
    dist = distance_transform_edt(free_mask)
    risk_map = np.clip(np.exp(-1.5 * dist), 0, 1)

    # 3. Risk-aware A* (lambda=8)
    def astar(grid, risk_map, start, goal, lambda_val):
        rows, cols = grid.shape
        def _h(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
            
        open_heap = []
        heapq.heappush(open_heap, (_h(start, goal), 0.0, start[0], start[1])) # f, g, r, c
        
        best = {start: 0.0}
        came_from = {start: None}
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
                      
        while open_heap:
            f, g, r, c = heapq.heappop(open_heap)
            node = (r, c)
            
            if node == goal:
                path = []
                cur = goal
                while cur is not None:
                    path.append(cur)
                    cur = came_from[cur]
                path.reverse()
                return path
                
            if g > best.get(node, float('inf')):
                continue
                
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if grid[nr, nc] == 1:
                    continue
                    
                nb = (nr, nc)
                step_dist = math.sqrt(dr*dr + dc*dc)
                step_cost = step_dist + lambda_val * risk_map[nr, nc]
                new_g = g + step_cost
                
                if new_g < best.get(nb, float('inf')):
                    best[nb] = new_g
                    came_from[nb] = node
                    new_f = new_g + _h(nb, goal)
                    heapq.heappush(open_heap, (new_f, new_g, nr, nc))
                    
        return []

    start = (15, 5)
    goal = (75, 75)
    path = astar(grid, risk_map, start, goal, 8.0)

    # 4. Create matplotlib animation
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Static map (walls black, free space dark grey)
    wall_display = np.where(grid == 1, 0.0, 0.2)
    ax.imshow(wall_display, cmap='gray', vmin=0, vmax=1, origin='upper')

    # Faint red risk heatmap
    risk_rgba = np.zeros((100, 100, 4), dtype=float)
    risk_rgba[:, :, 0] = risk_map  # red channel
    risk_rgba[:, :, 3] = risk_map * 0.4  # alpha
    ax.imshow(risk_rgba, origin='upper')

    # Start and goal
    ax.plot(start[1], start[0], 'go', markersize=10, label='Start') # green circle
    ax.plot(goal[1], goal[0], 'y*', markersize=15, label='Goal')    # yellow star

    # Pre-extract coordinates
    if len(path) > 0:
        xs = [c for r, c in path]
        ys = [r for r, c in path]
    else:
        xs, ys = [], []

    # Dynamic elements
    line, = ax.plot([], [], color='tab:blue', linewidth=2.5)
    robot_dot, = ax.plot([], [], 'wo', markersize=8) # white filled circle for robot

    ax.set_title("MediNav — Risk-Aware Navigation", color='white')
    ax.tick_params(colors='white')
    
    def init():
        line.set_data([], [])
        robot_dot.set_data([], [])
        return line, robot_dot

    def update(frame):
        if frame == 0:
            line.set_data([], [])
            robot_dot.set_data([], [])
        else:
            # Draw path travelled so far
            line.set_data(xs[:frame], ys[:frame])
            # Draw robot position
            robot_dot.set_data([xs[frame-1]], [ys[frame-1]])
        return line, robot_dot

    anim = animation.FuncAnimation(fig, update, frames=len(path)+1, init_func=init, interval=40, blit=True)

    # 5. Save as outputs/medinav_navigation.gif
    os.makedirs('outputs', exist_ok=True)
    writer = animation.PillowWriter(fps=15)
    anim.save('outputs/medinav_navigation.gif', writer=writer, dpi=100)
    
    print("Saved outputs/medinav_navigation.gif")

if __name__ == '__main__':
    main()