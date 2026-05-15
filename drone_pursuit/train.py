import os
import time
import yaml
import argparse
import subprocess
from threading import Thread

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from torch.utils.tensorboard import SummaryWriter

from drone_pursuit.env import PursuitEnv
from drone_pursuit.dqn_agent import DQNAgent
from drone_pursuit.perception import HSVDetector


STEP = 0.5
HOME = np.array([0.0, 0.0, 3.0], dtype=np.float32)

ACTION_OFFSETS = np.array([
    [ STEP, 0.0,  0.0],
    [-STEP, 0.0,  0.0],
    [0.0,   STEP, 0.0],
    [0.0,  -STEP, 0.0],
    [0.0,   0.0,  STEP],
    [0.0,   0.0, -STEP],
], dtype=np.float32)

CFG_PATH = os.path.expanduser(
    '~/drone_pursuit_ws/src/drone_pursuit/config/dqn_config.yaml'
)

CKPT_DIR = os.path.expanduser('~/drone_pursuit_ws/models/')
os.makedirs(CKPT_DIR, exist_ok=True)


def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def spawn_dist(episode):
   
    if episode < 200:
        return np.random.uniform(1.5, 3.5)
    elif episode < 600:
        return np.random.uniform(2.5, 5.0)
    else:
        return np.random.uniform(3.0, 6.0)


def goto_position(pub, env, x, y, z, wait=3.0, tol=0.8):
    msg = PoseStamped()
    msg.header.frame_id = 'map'
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = float(z)
    msg.pose.orientation.w = 1.0

    target = np.array([x, y, z], dtype=np.float32)
    t0 = time.time()

    while time.time() - t0 < wait:
        msg.header.stamp = env.get_clock().now().to_msg()
        pub.publish(msg)
        time.sleep(0.1)

        if np.linalg.norm(env.chaser_pos - target) < tol:
            break


def teleport_evader(evader_reset_client, ex, ey, ez=3.0):
    req = (
        f'name: "crazyflie" '
        f'position: {{x: {ex:.2f}, y: {ey:.2f}, z: {ez:.2f}}} '
        f'orientation: {{w: 1}}'
    )

    try:
        subprocess.run(
            [
                'gz', 'service',
                '-s', '/world/pursuit_world/set_pose',
                '--reqtype', 'gz.msgs.Pose',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '500',
                '--req', req
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.0
        )
    except Exception:
        pass

    if evader_reset_client.wait_for_service(timeout_sec=0.5):
        evader_reset_client.call_async(Trigger.Request())

    time.sleep(0.3)


def ensure_flying(env):
    if env.chaser_pos[2] > 1.5:
        return True

    print('[TRAIN] Drone too low — rearming...', flush=True)

    cmds = [
        (
            [
                'ros2', 'service', 'call', '/mavros/set_mode',
                'mavros_msgs/srv/SetMode',
                '{base_mode: 0, custom_mode: GUIDED}'
            ],
            2
        ),
        (
            [
                'ros2', 'service', 'call', '/mavros/cmd/arming',
                'mavros_msgs/srv/CommandBool',
                '{value: true}'
            ],
            2
        ),
        (
            [
                'ros2', 'service', 'call', '/mavros/cmd/takeoff',
                'mavros_msgs/srv/CommandTOL',
                '{min_pitch: 0.0, yaw: 0.0, latitude: 0.0, longitude: 0.0, altitude: 3.0}'
            ],
            6
        ),
    ]

    for cmd, delay in cmds:
        try:
            subprocess.run(
                cmd,
                timeout=5,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(delay)
        except Exception:
            pass

    print(f'[TRAIN] Rearm done. z={env.chaser_pos[2]:.2f}', flush=True)
    return env.chaser_pos[2] > 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args, _ = parser.parse_known_args()

    cfg = load_cfg()

    rclpy.init()

    perception = HSVDetector()
    env = PursuitEnv(cfg, perception)

    executor = MultiThreadedExecutor()
    executor.add_node(env)
    Thread(target=executor.spin, daemon=True).start()

    pos_pub = env.create_publisher(
        PoseStamped,
        '/mavros/setpoint_position/local',
        10
    )

    evader_reset_client = env.create_client(Trigger, '/evader/reset')

    agent = DQNAgent(cfg['agent'])

    start_ep = 0
    if args.checkpoint:
        agent.load(args.checkpoint)
        try:
            start_ep = int(
                os.path.basename(args.checkpoint)
                .replace('ep', '')
                .replace('.pt', '')
            )
        except Exception:
            start_ep = 0
        print(f'[TRAIN] Resumed from ep{start_ep}', flush=True)
    else:
        print('[TRAIN] Starting from scratch', flush=True)

    writer = SummaryWriter(
        os.path.expanduser('~/drone_pursuit_ws/runs/pursuit_dqn')
    )

    max_episodes = cfg['training']['max_episodes']
    max_steps = cfg['training']['max_steps_per_episode']
    save_every = cfg['training']['save_every']

    dcfg = cfg['drone']
    boundary = float(dcfg['boundary_radius'] - 1.0)

    print('[TRAIN] Waiting for drone + evader...', flush=True)

    t0 = time.time()
    while not env.is_ready():
        time.sleep(0.2)
        if time.time() - t0 > 30:
            print('[TRAIN] ERROR: timeout waiting for odometry')
            rclpy.shutdown()
            return

    print(f'[TRAIN] Ready! dist={env.get_distance():.2f}m', flush=True)

    ep_rewards = []
    intercepts = 0

    for episode in range(start_ep, max_episodes):

        if not ensure_flying(env):
            print('[TRAIN] Could not rearm — skipping episode', flush=True)
            continue

        env.stop()
        time.sleep(0.5)

        goto_position(pos_pub, env, *HOME, wait=5.0, tol=0.5)

        angle = np.random.uniform(0, 2 * np.pi)
        dist = spawn_dist(episode)

        ex = float(dist * np.cos(angle))
        ey = float(dist * np.sin(angle))

        teleport_evader(evader_reset_client, ex, ey)

        # Let Gazebo, MAVROS, and perception settle before RL actions begin.
        time.sleep(3.0)
        env.stop()

        state = env.get_state()
        prev_dist = env.get_distance()

        ep_reward = 0.0
        done = False
        step = 0

        print(
            f'[EP {episode + 1:4d}] evader=({ex:.1f},{ey:.1f}) '
            f'dist={prev_dist:.2f}m vision={env.vision_bit}',
            flush=True
        )

        while not done and step < max_steps:
            action = agent.select_action(state)
            offset = ACTION_OFFSETS[action]

            new_pos = env.chaser_pos.copy() + offset

            new_pos[0] = float(np.clip(new_pos[0], -boundary, boundary))
            new_pos[1] = float(np.clip(new_pos[1], -boundary, boundary))
            new_pos[2] = float(np.clip(new_pos[2], 1.5, 6.0))

            goto_position(pos_pub, env, *new_pos, wait=0.5, tol=0.8)

            curr_dist = env.get_distance()
            next_state = env.get_state()

            reward, done = env.compute_reward(prev_dist, curr_dist)

            agent.store(state, action, reward, next_state, float(done))
            agent.update()

            state = next_state
            prev_dist = curr_dist

            ep_reward += reward
            step += 1

        env.stop()

        final_dist = env.get_distance()
        intercepted = final_dist < dcfg['intercept_threshold']

        if intercepted:
            intercepts += 1

        ep_rewards.append(ep_reward)
        avg50 = float(np.mean(ep_rewards[-50:]))

        episode_count = episode - start_ep + 1

        writer.add_scalar('reward/episode', ep_reward, episode)
        writer.add_scalar('reward/avg50', avg50, episode)
        writer.add_scalar('agent/epsilon', agent.epsilon, episode)
        writer.add_scalar('training/intercept_rate', intercepts / episode_count, episode)
        writer.add_scalar('training/final_distance', final_dist, episode)

        print(
            f'Ep {episode + 1:4d} | steps={step:3d} | '
            f'dist={final_dist:.2f}m | '
            f'reward={ep_reward:8.2f} | avg50={avg50:8.2f} | '
            f'eps={agent.epsilon:.3f} | intercepts={intercepts}'
            f'{" INTERCEPT!" if intercepted else ""}',
            flush=True
        )

        if (episode + 1) % save_every == 0:
            ckpt = os.path.join(CKPT_DIR, f'ep{episode + 1}.pt')
            agent.save(ckpt)

    writer.close()
    executor.shutdown()
    rclpy.shutdown()

    print('[TRAIN] Done', flush=True)


if __name__ == '__main__':
    main()
