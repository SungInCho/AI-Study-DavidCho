import numpy as np

class GridWorld:
    def __init__(self, size=5, obstacles=None):
        assert size > 1, f"Grid size must be greater than 1. Received size:{size}."
        self.size = size
        self.agent_pos = None
        self.goal_pos = (size - 1, size - 1)
        self.obstacles = obstacles if obstacles is not None else []
        assert not any(pos in self.obstacles for pos in [(0, 0), (size - 1, size - 1)]), f"Obstacles at start or goal position."

    def reset(self):
        self.agent_pos = (0, 0)
        return self.agent_pos
    
    def step(self, action):
        # 0: up, 1: down, 2: left, 3: right
        actions = [(1, 0), (-1, 0), (0, -1), (0, 1)]  
        r, c = self.agent_pos
        dr, dc = actions[action]

        next_r = max(0, min(self.size - 1, r + dr))
        next_c = max(0, min(self.size - 1, c + dc))

        next_agent_pos = (next_r, next_c)

        if next_agent_pos in self.obstacles:
            return self.agent_pos, -1, False
        elif next_agent_pos == self.goal_pos:
            self.agent_pos = next_agent_pos
            return self.agent_pos, +1, True
        else:
            self.agent_pos = next_agent_pos
            return self.agent_pos, -0.1, False
        
    def render(self):
        grid = np.full((self.size, self.size), '.')
        for obs in self.obstacles:
            grid[obs] = 'X'
        grid[self.goal_pos] = 'G'
        grid[self.agent_pos] = 'A'
        print('\n'.join([' '.join(row) for row in grid]))
        print()