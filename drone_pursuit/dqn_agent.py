import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta        = beta_start
        self.beta_frames = beta_frames
        self.frame       = 1
        self.buffer      = []
        self.priorities  = np.zeros(capacity, dtype=np.float32)
        self.pos         = 0

    def store(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        self.beta = min(1.0, self.beta + (1.0 - 0.4) / self.beta_frames)
        self.frame += 1

        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = p + 1e-5

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, cfg):
        self.state_dim        = cfg['state_dim']
        self.action_dim       = cfg['action_dim']
        self.gamma            = cfg.get('gamma', 0.99)
        self.lr               = cfg.get('lr', 0.0005)
        self.epsilon          = cfg.get('epsilon_start', 1.0)
        self.epsilon_min      = cfg.get('epsilon_end', 0.05)
        self.epsilon_decay    = cfg.get('epsilon_decay', 0.9998)
        self.batch_size       = cfg.get('batch_size', 64)
        self.target_update    = cfg.get('target_update_freq', 500)
        hidden                = tuple(cfg.get('hidden_sizes', [256, 256]))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[DQN] Using device: {self.device}')

        self.q_net      = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.memory    = PrioritizedReplayBuffer(cfg.get('memory_size', 50000))
        self.steps     = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(s).argmax().item())

    def store(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.batch_size)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).to(self.device)
        w  = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q = self.q_net(s).gather(1, a).squeeze(1)

        # Double DQN target:
        # Use online net to SELECT best action
        # Use target net to EVALUATE that action
        # This reduces overestimation bias
        with torch.no_grad():
            next_actions = self.q_net(ns).argmax(1, keepdim=True)
            next_q       = self.target_net(ns).gather(1, next_actions).squeeze(1)
            target_q     = r + (1 - d) * self.gamma * next_q

        # Weighted loss (importance sampling weights from PER)
        td_errors = (current_q - target_q).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss = (w * nn.functional.smooth_l1_loss(
            current_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

        # Sync target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            print(f'[DQN] Target network synced at step {self.steps}', flush=True)

    def save(self, path):
        torch.save({
            'q_net':     self.q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon':   self.epsilon,
            'steps':     self.steps,
        }, path)
        print(f'[DQN] Saved checkpoint: {path}', flush=True)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['q_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt.get('epsilon', self.epsilon_min)
        self.steps   = ckpt.get('steps', 0)
        print(f'[DQN] Loaded checkpoint: {path} (step {self.steps})', flush=True)
