import os
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
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

    # 2. Risk map using scipy.ndimage.distance_transform_edt
    free_mask = (grid == 0).astype(float)
    dist = distance_transform_edt(free_mask)
    risk_map = np.clip(np.exp(-1.5 * dist), 0, 1)

    # 3. A* Implementation
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
                    
        return None

    # 4. Simulation
    start = (15, 5)
    goal = (75, 75)
    lambdas = [0, 1, 2, 4, 6, 8, 10, 15, 20]

    path_lengths = []
    safety_violations_list = []

    print(f"{'Lambda':<10} | {'Path Length':<15} | {'Safety Violations'}")
    print("-" * 50)

    for l_val in lambdas:
        path = astar(grid, risk_map, start, goal, l_val)
        if path:
            p_len = len(path)
            v_count = sum(1 for (r, c) in path if dist[r, c] < 3)
        else:
            p_len = 0
            v_count = 0
            
        path_lengths.append(p_len)
        safety_violations_list.append(v_count)
        
        print(f"{l_val:<10} | {p_len:<15} | {v_count}")

    # 5. Plotting
    os.makedirs('outputs', exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Safety Weight λ')
    ax1.set_ylabel('Path Length (cells)', color=color1)
    line1, = ax1.plot(lambdas, path_lengths, color=color1, marker='o', label='Path Length')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Safety Violations (count)', color=color2)
    line2, = ax2.plot(lambdas, safety_violations_list, color=color2, marker='s', label='Safety Violations')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Vertical line at lambda=8
    vmin, vmax = ax1.get_ylim()
    vline = ax1.axvline(x=8, color='grey', linestyle='--', label='MediNav setting (λ=8)')
    
    # Legends
    lines = [line1, line2, vline]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.title('Effect of Safety Weight λ on Path Length and Safety Violations')
    fig.tight_layout()
    plt.savefig('outputs/lambda_analysis.png', dpi=150)
    print("Saved dual-axis plot to outputs/lambda_analysis.png")

if __name__ == '__main__':
    main()