import numpy as np
import random 

class TD0:
    def __init__(self, env):
        self.env = env
        self.states = env.states
        self.actions = env.actions

    def gen_episodes(self, policy, num = 100, max_steps = 50):
        episodes = {}
        for i in range(num):
            episode = []
            self.env.reset()
            
            done = False
            while not done:
                if len(episode) > max_steps:
                    break

                cur_state_idx = self.state_id[self.env.cur_state]
                direction = np.random.choice(len(self.actions), p = policy[cur_state_idx])
                
                _, reward, _, done = self.env.step(direction)
                episode.append((cur_state_idx, direction, reward))
                
            episodes[i] = episode
        return episodes

    def evaluation(self, policy, alpha = 0.1, gamma = 0.99, max_steps=1000, max_iter = 10000):
        V = np.zeros(len(self.states))
        
        n_iter = 0
        while n_iter < max_iter:
            self.env.reset()
            done = False
            n_steps = 0
            while (not done) and (n_steps <= max_steps):
                cur_state_idx = self.env.state_to_index(self.env.cur_state)
                a = np.random.choice(len(self.actions), p = policy[cur_state_idx])
                next_s, r, _, done = self.env.step(a)
                next_state_idx = self.env.state_to_index(next_s)
                V[cur_state_idx] += alpha * (r + gamma * V[next_state_idx] - V[cur_state_idx])
                n_steps += 1
                
            n_iter += 1

        return V
    
    def SARSA(self, alpha = 0.1, gamma = 0.99, max_steps=1000, max_iter=10000):
        Q = np.zeros((len(self.states), len(self.actions)))

        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            epsilon = 1 / n_iter
            self.env.reset()
            cur_state_idx = self.env.state_to_index(self.env.cur_state)
            
            greedy_A = np.argmax(Q, axis=1)
            policy_Q = np.full((len(self.states), len(self.actions)), epsilon / len(self.actions))
            policy_Q[np.arange(len(self.states)), greedy_A] += 1 - epsilon 
            cur_a = np.random.choice(len(self.actions), p = policy_Q[cur_state_idx])
            
            done = False
            n_steps = 0
            while not done and n_steps <= max_steps:
                n_steps += 1
                next_s, r, _, done = self.env.step(cur_a)
                next_state_idx = self.env.state_to_index(next_s)

                greedy_A = np.argmax(Q, axis=1)
                policy_Q = np.full((len(self.states), len(self.actions)), epsilon / len(self.actions))
                policy_Q[np.arange(len(self.states)), greedy_A] += 1 - epsilon  
                next_a = np.random.choice(len(self.actions), p = policy_Q[next_state_idx])

                Q[cur_state_idx, cur_a] += alpha * (r + gamma * Q[next_state_idx, next_a] - Q[cur_state_idx, cur_a])
                cur_state_idx, cur_a = next_state_idx, next_a
        
        return policy_Q, Q