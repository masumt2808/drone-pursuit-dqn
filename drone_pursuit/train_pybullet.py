import sys
sys.path.insert(0, '/home/masum/drone_pursuit_ws/src/drone_pursuit')

import numpy as np
import yaml, time, os
from torch.utils.tensorboard import SummaryWriter
from drone_pursuit.dqn_agent import DQNAgent

CFG_PATH = os.path.expanduser(
    '~/drone_pursuit_ws/src/drone_pursuit/config/dqn_config.yaml')
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

STEP             = 0.5
MAX_EPISODES     = 3000
MAX_STEPS        = 200
SAVE_EVERY       = 100
INTERCEPT_THRESH = 0.8
BOUNDARY         = 8.0
EVADER_SPEED     = 0.5
CKPT_DIR         = os.path.expanduser('~/drone_pursuit_ws/models/pybullet/')
os.makedirs(CKPT_DIR, exist_ok=True)

ACTION_OFFSETS = np.array([
    [ STEP, 0,     0    ],
    [-STEP, 0,     0    ],
    [0,     STEP,  0    ],
    [0,    -STEP,  0    ],
    [0,     0,     STEP ],
    [0,     0,    -STEP ],
], dtype=np.float32)


def get_vision_bit(chaser_pos, chaser_vel, evader_pos):
    rel = evader_pos - chaser_pos
    dist = np.linalg.norm(rel)
    speed = np.linalg.norm(chaser_vel)
    if dist < 0.01 or speed < 0.01:
        return 0
    forward = chaser_vel / speed
    direction_to_evader = rel / dist
    angle = np.arccos(np.clip(np.dot(forward, direction_to_evader), -1.0, 1.0))
    return 1 if angle < np.radians(30) and dist < 8.0 else 0


def reset_sim():
    chaser_pos = np.array([0.0, 0.0, 3.0])
    angle      = np.random.uniform(0, 2 * np.pi)
    dist       = np.random.uniform(2.0, 6.0)
    evader_pos = np.array([
        dist * np.cos(angle),
        dist * np.sin(angle),
        3.0,
    ])
    return chaser_pos.copy(), evader_pos.copy()


def random_evader_vel():
    d = np.random.randn(3)
    d[2] = 0.0
    n = np.linalg.norm(d)
    if n > 0:
        d /= n
    return d * EVADER_SPEED


def step_evader(evader_pos, evader_vel, dt=0.05):
    evader_pos = evader_pos + evader_vel * dt
    if np.linalg.norm(evader_pos[:2]) > 15.0:
        evader_vel[:2] *= -1.0
    evader_pos[2] = 3.0
    return evader_pos, evader_vel


def get_state(chaser_pos, chaser_vel, evader_pos, vision_bit):
    rel         = evader_pos - chaser_pos
    dist        = float(np.linalg.norm(rel))
    speed       = np.linalg.norm(chaser_vel)
    if dist > 0.01 and speed > 0.01:
        dot         = np.clip(np.dot(rel/dist, chaser_vel/speed), -1, 1)
        heading_err = float(np.arccos(dot))
    else:
        heading_err = 0.0
    alt_err = float(evader_pos[2] - chaser_pos[2])
    return np.array([
        rel[0], rel[1], rel[2],
        chaser_vel[0], chaser_vel[1], chaser_vel[2],
        dist, heading_err, alt_err, float(vision_bit),
    ], dtype=np.float32)


def compute_reward(prev_dist, curr_dist, vision_bit, chaser_pos):
    rcfg = cfg['reward']
    dcfg = cfg['drone']
    r = rcfg['step_dist_penalty'] * curr_dist
    r += 5.0 * (prev_dist - curr_dist)
    if vision_bit == 1:
        r += 5.0
    else:
        r += rcfg['vision_penalty']
    if curr_dist < 5.0:
        r += rcfg['heading_bonus']
    if curr_dist < 2.0:
        r += rcfg['proximity_bonus']
    if chaser_pos[2] < 1.0 or chaser_pos[2] > 7.0:
        return r - 100.0, True
    if curr_dist < INTERCEPT_THRESH:
        return r + rcfg['terminal_reward'], True
    if curr_dist > BOUNDARY:
        return r - 50.0, True
    return r, False


agent  = DQNAgent(cfg['agent'])
writer = SummaryWriter(
    os.path.expanduser('~/drone_pursuit_ws/runs/pybullet_dqn'))

print('[PyBullet] Training started', flush=True)
ep_rewards, intercepts = [], 0

for episode in range(MAX_EPISODES):
    chaser_pos, evader_pos = reset_sim()
    chaser_vel    = np.zeros(3, dtype=np.float32)
    evader_vel    = random_evader_vel()
    direction_timer = 0

    vision_bit = get_vision_bit(chaser_pos, chaser_vel, evader_pos)
    state      = get_state(chaser_pos, chaser_vel, evader_pos, vision_bit)
    prev_dist  = float(np.linalg.norm(evader_pos - chaser_pos))
    ep_reward, done, step = 0.0, False, 0

    while not done and step < MAX_STEPS:
        action  = agent.select_action(state)
        offset  = ACTION_OFFSETS[action]
        new_pos = chaser_pos + offset
        new_pos[0] = np.clip(new_pos[0], -15, 15)
        new_pos[1] = np.clip(new_pos[1], -15, 15)
        new_pos[2] = np.clip(new_pos[2], 1.5, 7.0)

        chaser_vel = (new_pos - chaser_pos) / 0.05
        chaser_pos = new_pos

        direction_timer += 1
        if direction_timer >= 40:
            evader_vel      = random_evader_vel()
            direction_timer = 0
        evader_pos, evader_vel = step_evader(evader_pos, evader_vel)

        curr_dist  = float(np.linalg.norm(evader_pos - chaser_pos))
        vision_bit = get_vision_bit(chaser_pos, chaser_vel, evader_pos)
        next_state = get_state(chaser_pos, chaser_vel, evader_pos, vision_bit)
        reward, done = compute_reward(prev_dist, curr_dist, vision_bit, chaser_pos)

        agent.store(state, action, reward, next_state, float(done))
        agent.update()

        state, prev_dist = next_state, curr_dist
        ep_reward += reward
        step += 1

    intercepted = curr_dist < INTERCEPT_THRESH
    if intercepted:
        intercepts += 1

    ep_rewards.append(ep_reward)
    avg50 = float(np.mean(ep_rewards[-50:]))

    writer.add_scalar('reward/episode',         ep_reward,              episode)
    writer.add_scalar('reward/avg50',            avg50,                  episode)
    writer.add_scalar('agent/epsilon',           agent.epsilon,          episode)
    writer.add_scalar('training/intercept_rate',
                      intercepts / (episode + 1),                       episode)

    print(f'[PyBullet] Ep {episode+1:4d} | steps={step:3d} | '
          f'dist={curr_dist:.2f}m | reward={ep_reward:8.2f} | '
          f'avg50={avg50:8.2f} | eps={agent.epsilon:.3f} | '
          f'intercepts={intercepts}'
          f'{" INTERCEPT!" if intercepted else ""}', flush=True)

    if (episode + 1) % SAVE_EVERY == 0:
        ckpt = os.path.join(CKPT_DIR, f'ep{episode+1}.pt')
        agent.save(ckpt)

writer.close()
print('[PyBullet] Training done', flush=True)
