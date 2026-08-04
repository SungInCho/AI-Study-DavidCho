import numpy as np
from DP_policyeval import DP_policyeval

class DP_policyiter:
    def __init__(self, env):
        self.env = env
        self.dynamics, self.rewards = env.get_dynamics_rewards()
        self.states = env.states
        self.actions = env.actions
        self.eval = DP_policyeval(env)

    def improvement(self, V, gamma=0.99):
        # (S, A, S') * [(S, A, S') + (S',)] -> (S, A, S') 
        Q = np.sum(self.dynamics * (self.rewards + gamma * V), axis=-1) # (S, A,)
        greedy_actions = np.argmax(Q, axis=1) # (S,)
        greedy_policy = np.zeros((len(self.states), len(self.actions), 1))
        for i in range(len(self.states)):
            greedy_policy[i, greedy_actions[i]] = 1

        return greedy_policy

    def forward(self, gamma=0.99, theta = 1e-6):
        policy = np.full((len(self.states), len(self.actions), 1), 1 / len(self.actions))
        delta = np.inf

        while delta != 0:
            V = self.eval.eval(policy, gamma=gamma, theta=theta)
            new_policy = self.improvement(V, gamma=gamma)
            delta = np.sum(np.abs(policy - new_policy))
            policy = new_policy.copy()

        return policy, V