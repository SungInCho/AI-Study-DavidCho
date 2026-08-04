import numpy as np

class DP_valueiter:
    def __init__(self, env):
        self.env = env
        self.dynamics, self.rewards = env.get_dynamics_rewards()
        self.states = env.states
        self.actions = env.actions
        self.V = np.zeros(len(self.states))
        self.policy = np.zeros((len(self.states), len(self.actions), 1))
    
    def forward(self, gamma = 0.99, threshold = 1e-6):
        delta = np.inf
        while delta > threshold:
            Q = np.sum(self.dynamics * (self.rewards + gamma * self.V), axis = -1)
            next_V = np.max(Q, axis = -1)
            delta = max(abs(next_V - self.V))
            self.V = next_V

        greedy_policy = np.argmax(Q, axis = -1)
        for ind, act in enumerate(greedy_policy):
            self.policy[ind, act] = 1

        return self.policy, self.V

