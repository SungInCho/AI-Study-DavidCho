import numpy as np

class DP_policyeval:
    def __init__(self, env):
        self.env = env
        self.dynamics, self.rewards = env.get_dynamics_rewards()
        self.states = env.states
        self.actions = env.actions

        self.V = np.zeros(len(self.states))
    
    def eval(self, policy, gamma = 0.99, theta = 1e-6):
        delta = np.inf
        while delta >= theta:
            new_V = np.sum(policy * self.dynamics * (self.rewards + gamma * self.V), axis=(1, 2))
            delta = max(abs(new_V - self.V))
            self.V = new_V
        return self.V