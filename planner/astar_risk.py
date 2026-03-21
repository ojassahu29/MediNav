
import heapq
import math
import numpy as np


def astar_risk(grid, risk_map, start, goal, lambda_weight=5.0):
    

    rows, cols = grid.shape

    def h(node):
        
        return math.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

    
    start_f = h(start)
   
    open_heap = [(start_f, 0.0, 0.0, start[0], start[1])]


    best = {}  
    best[start] = (0.0, 0.0)

   
    came_from = {start: None}

    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_heap:
        f, g_d, g_r, r, c = heapq.heappop(open_heap)
        node = (r, c)

        
        if node in best:
            bd, br = best[node]
            if g_d > bd + 1e-9 or g_r > br + 1e-9:
               
                if g_d >= bd - 1e-9 and g_r >= br - 1e-9:
                    continue

        if node == goal:
            
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue  

            neighbour = (nr, nc)

            
            step_dist = math.sqrt(dr * dr + dc * dc)
            
            step_risk = float(risk_map[nr, nc])

            new_g_d = g_d + step_dist
            new_g_r = g_r + step_risk

            if neighbour in best:
                bd, br = best[neighbour]
                
                new_f = new_g_d + lambda_weight * new_g_r + h(neighbour)
                old_f = bd + lambda_weight * br + h(neighbour)
                if new_f >= old_f - 1e-9:
                    continue

            best[neighbour] = (new_g_d, new_g_r)
            came_from[neighbour] = node
            new_f = new_g_d + lambda_weight * new_g_r + h(neighbour)
            heapq.heappush(open_heap, (new_f, new_g_d, new_g_r, nr, nc))

    return None  


def astar_standard(grid, start, goal):
   
    zero_risk = np.zeros_like(grid, dtype=float)
    return astar_risk(grid, zero_risk, start, goal, lambda_weight=0.0)



if __name__ == "__main__":
    np.random.seed(42)

    SIZE = 20
    grid = np.zeros((SIZE, SIZE), dtype=int)

   
    grid[10, :15] = 1
    grid[10, 16:] = 1 

    
    risk = np.zeros((SIZE, SIZE), dtype=float)
    for r in range(SIZE):
        for c in range(SIZE):
            dist_to_wall = abs(r - 10)
            risk[r, c] = max(0.0, 1.0 - dist_to_wall / 5.0)

    start = (2, 2)
    goal = (18, 18)

    path_std = astar_standard(grid, start, goal)
    path_risk = astar_risk(grid, risk, start, goal, lambda_weight=5.0)

    if path_std:
        print(f"Standard A*      — path length: {len(path_std)} cells")
    else:
        print("Standard A* — no path found")

    if path_risk:
        print(f"Risk-Aware A*    — path length: {len(path_risk)} cells")
    else:
        print("Risk-Aware A* — no path found")

    if path_std and path_risk:
        overhead = (len(path_risk) - len(path_std)) / len(path_std) * 100
        print(f"Safety overhead  : {overhead:.1f}%")
