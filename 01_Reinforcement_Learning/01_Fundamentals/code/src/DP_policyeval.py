import numpy as np
from GridWorld import GridWorld

class DP_policyeval:
    def __init__(self, env, states, actions):
        self.env = env

        self.states = states
        self.actions = actions

        rows, _ = zip(*states) 
        self.size = max(rows) + 1
        
        self.policy = np.full((len(self.states), len(self.actions), 1), 1 / len(self.actions))
        self.V = np.zeros(len(self.states))

    def state_to_index(self, state):
        r, c = state
        return int(r * self.size + c)

    def index_to_state(self, ind):
        return (int(ind // self.size), int(ind % self.size))
    
    def get_dynamics_rewards(self):
        dynamics = np.zeros((len(self.states), len(self.actions), len(self.states)))
        rewards = np.zeros((len(self.states), len(self.actions), len(self.states)))
        for r in range(self.size):
            for c in range(self.size):
                for act in self.actions:
                    transition = self.env.get_transition_prob((r,c), act)
                    for state, (prob, reward, _) in transition.items():
                        dynamics[self.state_to_index((r,c)), act, self.state_to_index(state)] = prob
                        rewards[self.state_to_index((r,c)), act, self.state_to_index(state)] = reward
        return dynamics, rewards
    
    def eval(self, gamma = 0.99, theta = 1e-6):
        delta = np.inf
        dynamics, rewards = self.get_dynamics_rewards()
        while delta >= theta:
            new_V = np.sum(self.policy * dynamics * (rewards + gamma * self.V), axis=(1, 2))
            delta = max(abs(new_V - self.V))
            self.V = new_V
        return self.V