import torch
import torch.nn as nn
import numpy as np
import random
import os


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(256, 256)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):
            self.priorities[i] = abs(float(err)) + 1e-6

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    # 6 discrete actions: ±x ±y ±z
    ACTIONS = np.array([
        [ 1,  0,  0],
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1],
    ], dtype=np.float32)

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[DQN] Using device: {self.device}')

        sd = cfg['state_dim']
        ad = cfg['action_dim']
        hidden = tuple(cfg['hidden_sizes'])

        self.q_net     = QNetwork(sd, ad, hidden).to(self.device)
        self.target_net = QNetwork(sd, ad, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg['lr'])
        self.memory    = PrioritizedReplayBuffer(cfg['memory_size'])
        self.epsilon   = cfg['epsilon_start']
        self.steps     = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.cfg['action_dim'])
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(s).argmax().item())

    def action_to_velocity(self, action_idx: int, speed: float = 1.0) -> np.ndarray:
        return self.ACTIONS[action_idx] * speed

    def store(self, s, a, r, s_, done):
        self.memory.push(
            np.array(s, dtype=np.float32),
            int(a),
            float(r),
            np.array(s_, dtype=np.float32),
            float(done),
        )

    def update(self):
        if len(self.memory) < self.cfg['batch_size']:
            return None

        states, actions, rewards, next_states, dones, idxs, weights = \
            self.memory.sample(self.cfg['batch_size'])

        S  = torch.FloatTensor(states).to(self.device)
        A  = torch.LongTensor(actions).to(self.device)
        R  = torch.FloatTensor(rewards).to(self.device)
        S_ = torch.FloatTensor(next_states).to(self.device)
        D  = torch.FloatTensor(dones).to(self.device)
        W  = torch.FloatTensor(weights).to(self.device)

        q_vals = self.q_net(S).gather(1, A.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_a  = self.q_net(S_).argmax(1)
            q_next  = self.target_net(S_).gather(1, best_a.unsqueeze(1)).squeeze(1)
            targets = R + self.cfg['gamma'] * q_next * (1 - D)

        td_errors = (q_vals - targets).detach().cpu().numpy()
        loss = (W * nn.functional.huber_loss(
            q_vals, targets, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.memory.update_priorities(idxs, td_errors)
        self.steps += 1

        if self.steps % self.cfg['target_update_freq'] == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            print(f'[DQN] Target network synced at step {self.steps}')

        self.epsilon = max(
            self.cfg['epsilon_end'],
            self.epsilon * self.cfg['epsilon_decay'],
        )
        return float(loss.item())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_net':     self.q_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon':   self.epsilon,
            'steps':     self.steps,
        }, path)
        print(f'[DQN] Saved checkpoint: {path}')

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['q_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt['epsilon']
        self.steps   = ckpt['steps']
        print(f'[DQN] Loaded checkpoint: {path} (step {self.steps})')
