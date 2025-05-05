import torch

class KLScheduler:
    def __init__(self, init_kl=0.1, final_kl=0.02, total_steps=10000):
        self.init_kl = init_kl
        self.final_kl = final_kl
        self.total_steps = total_steps

    def get_kl(self, step):
        frac = min(step / self.total_steps, 1.0)
        return self.init_kl - frac * (self.init_kl - self.final_kl)

class RewardNormalizer:
    def __init__(self, alpha=0.99):
        self.mean = 0.0
        self.alpha = alpha

    def normalize(self, reward):
        self.mean = self.alpha * self.mean + (1 - self.alpha) * float(reward)
        return reward - self.mean

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.01):
        self.best_score = None
        self.counter = 0
        self.patience = patience
        self.min_delta = min_delta

    def step(self, current_score):
        if self.best_score is None or current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    next_value = 0.0
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_value - v
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = v
    return torch.tensor(advantages, device=values.device)
