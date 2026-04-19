import numpy as np
import random

class GridWorld:
    def __init__(self, size=5, obstacles=None, icy_floors=None):
        assert size > 1, f"Grid size must be greater than 1. Received size:{size}."
        self.size = size
        self.agent_pos = None
        self.goal_pos = (size - 1, size - 1)
        # 0: up, 1: down, 2: left, 3: right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
        self.states = [(r, c) for r in range(size) for c in range(size)]
        
        self.obstacles = obstacles if obstacles is not None else []
        assert not any(pos in self.obstacles for pos in [(0, 0), (size - 1, size - 1)]), f"Obstacles at the start or goal position."
        
        self.icy_floors = icy_floors if icy_floors is not None else []
        assert not any(pos in self.obstacles for pos in self.icy_floors), f"Obstacles above icy floors."
    
    def state_to_index(self, state):
        r, c = state
        return int(r * self.size + c)
    
    def index_to_state(self, ind):
        return (int(ind // self.size), int(ind % self.size))
    
    def reset(self):
        self.agent_pos = (0, 0)
        return self.agent_pos  
    
    def get_transition_prob(self, state, action):
        action_list = ["Up", "Down", "Left", "Right", "None"]
        transition = {}
        r, c = state
        dr, dc = self.actions[action]
        next_state = (r + dr, c + dc)

        if state in self.icy_floors:
            p_main, p_i1, p_i2 = 0.8, 0.1, 0.1
            
            if action in [0, 1]:
                action_i1, action_i2 = 2, 3
            else:
                action_i1, action_i2 = 0, 1
            dr_i1, dc_i1 = self.actions[action_i1]
            dr_i2, dc_i2 = self.actions[action_i2]
            next_state_i1 = (r + dr_i1, c + dc_i1)
            next_state_i2 = (r + dr_i2, c + dc_i2)

            candidates = [
                (next_state, p_main, action),
                (next_state_i1, p_i1, action_i1),
                (next_state_i2, p_i2, action_i2)
            ]

            for s_next, prob, move in candidates:
                r_next, c_next = s_next
                is_out = not (0 <= r_next <= self.size - 1 and 0 <= c_next <= self.size -1)
                is_obstacle = s_next in self.obstacles
                is_goal = s_next == self.goal_pos

                if is_out or is_obstacle:
                    old_prob, _, _ = transition.get(state, (0, -1, action_list[4]))
                    transition[state] = (old_prob + prob, -1, action_list[4])
                elif is_goal:
                    old_prob, _, _ = transition.get(s_next, (0, 1, action_list[move]))
                    transition[s_next] = (old_prob + prob, 1, action_list[move])
                else:
                    old_prob, _, _ = transition.get(s_next, (0, -0.1, action_list[move]))
                    transition[s_next] = (old_prob + prob, -0.1, action_list[move])
        else:
            r_next, c_next = next_state
            is_out = not (0 <= r_next <= self.size - 1 and 0 <= c_next <= self.size -1)
            is_obstacle = next_state in self.obstacles
            is_goal = next_state == self.goal_pos

            if is_out or is_obstacle:
                transition[state] = (1, -1, action_list[4])
            elif is_goal:
                transition[next_state] = (1, 1, action_list[action])
            else:
                transition[next_state] = (1, -0.1, action_list[action])

        return transition

    def get_dynamics_rewards(self):
        dynamics = np.zeros((len(self.states), len(self.actions), len(self.states)))
        rewards = np.zeros((len(self.states), len(self.actions), len(self.states)))
        for r in range(self.size):
            for c in range(self.size):
                for act in range(len(self.actions)):
                    transition = self.get_transition_prob((r,c), act)
                    for state, (prob, reward, _) in transition.items():
                        dynamics[self.state_to_index((r,c)), act, self.state_to_index(state)] = prob
                        rewards[self.state_to_index((r,c)), act, self.state_to_index(state)] = reward
        return dynamics, rewards
    
    def step(self, action):
        transition = self.get_transition_prob(self.agent_pos, action)
        next_states = list(transition.keys())
        probs = list(v[0] for v in transition.values())

        next_state = random.choices(next_states, probs, k=1)[0]

        self.agent_pos = next_state
        reward = transition[next_state][1]
        move = transition[next_state][2]

        if next_state == self.goal_pos:
            return next_state, reward, move, True
        
        else:
            return next_state, reward, move, False
        
    def render(self):
        grid = np.full((self.size, self.size), '.')
        for obs in self.obstacles:
            grid[obs] = 'X'
        for obs in self.icy_floors:
            grid[obs] = 'O'
        grid[self.goal_pos] = 'G'
        grid[self.agent_pos] = 'A'
        print('\n'.join([' '.join(row) for row in grid]))
        print()