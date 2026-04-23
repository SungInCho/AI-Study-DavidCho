import numpy as np
import random 
from collections import Counter, defaultdict

class MonteCarlo:
    def __init__(self, env):
        self.env = env
        self.states = env.states
        self.state_id = {state: i  for i, state in enumerate(self.states)}
        self.actions = env.actions

    def gen_episodes(self, policy, num = 100, max_steps = 50, ES = False):
        episodes = {}
        for i in range(num):
            episode = []
            self.env.reset()

            if ES:
                self.env.cur_state = random.choice(self.env.valid_states)
                direction = random.randrange(len(self.actions))
            else:
                cur_state_idx = self.state_id[self.env.cur_state]
                direction = np.random.choice(len(self.actions), p = policy[cur_state_idx])
            
            done = False
            while not done:
                if len(episode) > max_steps:
                    break

                cur_state_idx = self.state_id[self.env.cur_state]
                
                _, reward, _, done = self.env.step(direction)
                episode.append((cur_state_idx, direction, reward))

                direction = np.random.choice(len(self.actions), p = policy[cur_state_idx])
            
            episodes[i] = episode
        return episodes
    
    def first_visit_pred(self, policy, num = 100, max_steps = 50, gamma = 0.99):
        episodes = self.gen_episodes(policy, num, max_steps, False)

        V = np.zeros(len(self.states))
        Q = np.zeros((len(self.states), len(self.actions)))

        count_s = defaultdict(int)
        count_sa = defaultdict(int)
        
        for episode in episodes.values():
            s_counts = Counter(s for s, _, _ in episode)
            sa_counts = Counter((s, a) for s, a, _ in episode)
            G = 0
            for s, a, r in reversed(episode):
                G = r + gamma * G

                if s_counts[s] == 1:
                    count_s[s] += 1
                    V[s] += (G - V[s]) / count_s[s]
                s_counts[s] -= 1

                if sa_counts[(s, a)] == 1:
                    count_sa[(s, a)] += 1
                    Q[s, a] += (G - Q[s, a]) / count_sa[(s, a)]
                sa_counts[(s, a)] -= 1
        
        return V, Q
    
    def every_visit_pred(self, policy, num = 100, max_steps = 50, gamma = 0.99):
        episodes = self.gen_episodes(policy, num, max_steps, False)

        V = np.zeros(len(self.states))
        Q = np.zeros((len(self.states), len(self.actions)))

        count_s = defaultdict(int)
        count_sa = defaultdict(int)

        for episode in episodes.values():
            G = 0
            for s, a, r in reversed(episode):
                G = r + gamma * G

                count_s[s] += 1
                V[s] += (G - V[s]) / count_s[s]

                count_sa[(s, a)] += 1
                Q[s, a] += (G - Q[s, a]) / count_sa[(s, a)]
        
        return V, Q
    
    def ES_control(self, max_steps = 100, gamma = 0.99, visit="first", max_iter = 10000):
        assert visit in ["first", "every"], f"visit must be either \"fisrt\" or \"every\"."
        
        policy = np.full((len(self.states), len(self.actions)), 1/len(self.actions))
        Q = np.zeros((len(self.states), len(self.actions)))

        count_sa = defaultdict(int)
        num_iter = 0
        while num_iter < max_iter:
            episode = self.gen_episodes(policy, 1, max_steps, True)[0]
            G = 0
            sa_counts = Counter((s, a) for s, a, _ in episode)
            for s, a, r in reversed(episode):
                G = r + gamma * G

                if visit == "first":
                    sa_counts[(s, a)] -= 1
                    if sa_counts[(s, a)] == 0:
                        count_sa[(s, a)] += 1
                        Q[s, a] += (G - Q[s, a]) / count_sa[(s, a)]
                else:
                    count_sa[(s, a)] += 1
                    Q[s, a] += (G - Q[s, a]) / count_sa[(s, a)]
                
            greedy_A = np.argmax(Q, axis = 1)
            policy = np.zeros((len(self.states), len(self.actions)))
            for s in range(len(self.states)):
                policy[s, greedy_A[s]] = 1
            num_iter += 1
        
        return policy, Q
    
    def make_eps_greedy(self, Q, epsilon=0.1):
        policy = np.full((len(self.states), len(self.actions)), epsilon / len(self.actions))
        greedy_A = np.argmax(Q, axis=1)
        for s in range(len(self.states)):
            policy[s, greedy_A[s]] += 1 - epsilon
        return policy

    def onpolicy(self, max_steps = 100, gamma = 0.99, visit="first", max_iter = 10000, epsilon=0.1):
        assert visit in ["first", "every"], f"visit must be either \"fisrt\" or \"every\"."

        min_prob = epsilon / len(self.actions)
        remaining = 1 - min_prob * len(self.actions)
        policy = np.random.dirichlet(np.ones(len(self.actions)), size=len(self.states)) * remaining + min_prob
        Q = np.zeros((len(self.states), len(self.actions)))
        
        count_sa = defaultdict(int)
        num_iter= 0
        while num_iter < max_iter:
            episode = self.gen_episodes(policy, 1, max_steps, False)[0]
            G = 0
            sa_counts = Counter((s,a) for s, a, _ in episode)
            
            for s, a, r in reversed(episode):
                G = r + gamma * G

                if visit == "first":
                    sa_counts[(s, a)] -= 1
                    if sa_counts[(s, a)] == 0:
                        count_sa[(s, a)] += 1
                        Q[s, a] += (G - Q[s, a]) / count_sa[(s, a)]
                else:
                    count_sa[(s, a)] += 1
                    Q[s, a] += (G - Q[s, a]) / count_sa[(s, a)]
            
            policy = self.make_eps_greedy(Q, epsilon)
            num_iter += 1
        
        return policy, Q

    def offpolicy_pred(self, target_policy, max_steps = 100, gamma = 0.99, max_iter = 10000, epsilon=0.1):
        min_prob = epsilon / len(self.actions)
        remaining = 1 - min_prob * len(self.actions)
        b_policy = np.random.dirichlet(np.ones(len(self.actions)), size=len(self.states)) * remaining + min_prob
        Q = np.zeros((len(self.states), len(self.actions)))
        C = np.zeros((len(self.states), len(self.actions)))

        num_iter = 0
        while num_iter < max_iter:
            episode = self.gen_episodes(b_policy, 1, max_steps, False)[0]
            G = 0
            W = 1
            sa_counts = Counter((s, a) for s, a, _ in episode)
            for s, a, r in reversed(episode):
                G = r + gamma * G

                sa_counts[(s, a)] -= 1
                if sa_counts[(s, a)] == 0:
                    if W == 0:
                        break
                    C[s, a] += W
                    Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
                    W *=  target_policy[s, a] / b_policy[s, a]
            num_iter += 1

        return Q
                    
