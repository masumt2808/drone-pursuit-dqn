"""
Evaluation script — tests trained DQN policy against 3 evader difficulties.
No exploration (epsilon=0), greedy policy only.
Reports: intercept rate, avg reward, avg steps, avg final distance.
"""
import rclpy
import yaml, time, os, sys, numpy as np
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
import subprocess
import argparse

from drone_pursuit.env import PursuitEnv
from drone_pursuit.dqn_agent import DQNAgent
from drone_pursuit.perception import HSVDetector


def load_cfg():
    p = os.path.expanduser(
        '~/drone_pursuit_ws/src/drone_pursuit/config/dqn_config.yaml')
    with open(p) as f:
        return yaml.safe_load(f)


STEP = 0.5
ACTION_OFFSETS = np.array([
    [ STEP, 0,     0    ],
    [-STEP, 0,     0    ],
    [0,     STEP,  0    ],
    [0,    -STEP,  0    ],
    [0,     0,     STEP ],
    [0,     0,    -STEP ],
], dtype=np.float32)

EVADER_DIFFICULTIES = {
    'static': 0.0,
    'slow':   0.5,
    'fast':   1.5,
}

EVAL_EPISODES = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint .pt file')
    parser.add_argument('--episodes', type=int, default=EVAL_EPISODES)
    parser.add_argument('--difficulty', type=str, default='all',
                        choices=['all', 'static', 'slow', 'fast'],
                        help='Which difficulty to evaluate')
    parsed, _ = parser.parse_known_args()

    rclpy.init()
    cfg = load_cfg()

    perception = HSVDetector()
    env = PursuitEnv(cfg, perception)
    executor = MultiThreadedExecutor()
    executor.add_node(env)
    Thread(target=executor.spin, daemon=True).start()

    # publishers
    pos_pub = env.create_publisher(
        PoseStamped, '/mavros/setpoint_position/local', 10)
    evader_reset_client = env.create_client(Trigger, '/evader/reset')

    def goto_position(x, y, z, wait=4.0, tol=0.8):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        t0 = time.time()
        while time.time() - t0 < wait:
            msg.header.stamp = env.get_clock().now().to_msg()
            pos_pub.publish(msg)
            time.sleep(0.1)
            if np.linalg.norm(env.chaser_pos - np.array([x,y,z])) < tol:
                break

    def teleport_evader(x, y, z=3.0):
        req = (f'name: "crazyflie" '
               f'position: {{x: {x:.2f}, y: {y:.2f}, z: {z:.2f}}} '
               f'orientation: {{w: 1}}')
        try:
            subprocess.run(
                ['gz', 'service', '-s', '/world/pursuit_world/set_pose',
                 '--reqtype', 'gz.msgs.Pose',
                 '--reptype', 'gz.msgs.Boolean',
                 '--timeout', '500', '--req', req],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.0)
        except Exception:
            pass
        if evader_reset_client.wait_for_service(timeout_sec=0.5):
            evader_reset_client.call_async(Trigger.Request())
        time.sleep(0.3)

    # load agent — greedy (epsilon=0)
    agent = DQNAgent(cfg['agent'])
    agent.load(parsed.checkpoint)
    agent.epsilon = 0.0
    print(f'\n[EVAL] Loaded: {parsed.checkpoint}')
    print(f'[EVAL] epsilon={agent.epsilon} (greedy)')
    print(f'[EVAL] Episodes per difficulty: {parsed.episodes}')

    # wait for drone
    print('[EVAL] Waiting for drone...', flush=True)
    t0 = time.time()
    while not env.is_ready():
        time.sleep(0.1)
        if time.time() - t0 > 30:
            print('[EVAL] Timeout waiting for drone')
            rclpy.shutdown()
            return

    HOME = np.array([0.0, 0.0, 3.0])
    all_results = {}

    difficulties = EVADER_DIFFICULTIES if parsed.difficulty == 'all'         else {parsed.difficulty: EVADER_DIFFICULTIES[parsed.difficulty]}

    for difficulty, evader_speed in difficulties.items():
        print(f'\n{"="*50}')
        print(f'[EVAL] Difficulty: {difficulty.upper()} '
              f'(evader speed={evader_speed} m/s)')
        print(f'{"="*50}')

        intercepts   = 0
        rewards      = []
        steps_list   = []
        final_dists  = []

        # Set evader speed for this difficulty
        import subprocess as sp
        sp.run(['ros2', 'param', 'set', '/evader_node', 'speed',
                str(evader_speed)],
               capture_output=True, timeout=3)
        time.sleep(0.5)
        print(f'  Evader speed set to {evader_speed} m/s', flush=True)

        for ep in range(parsed.episodes):
            # reset
            env.stop()
            goto_position(*HOME, wait=5.0, tol=0.5)

            # random evader spawn
            angle = np.random.uniform(0, 2 * np.pi)
            dist  = np.random.uniform(2.0, 5.0)
            ex    = float(dist * np.cos(angle))
            ey    = float(dist * np.sin(angle))
            teleport_evader(ex, ey, 3.0)

            state     = env.get_state()
            prev_dist = env.get_distance()
            ep_reward = 0.0
            done      = False
            step      = 0

            while not done and step < cfg['training']['max_steps_per_episode']:
                action  = agent.select_action(state)
                offset  = ACTION_OFFSETS[action]
                drone_pos = env.chaser_pos.copy()
                new_pos = drone_pos + offset
                new_pos[0] = np.clip(new_pos[0], -12, 12)
                new_pos[1] = np.clip(new_pos[1], -12, 12)
                new_pos[2] = np.clip(new_pos[2], 1.5, 6.0)

                goto_position(*new_pos, wait=0.5, tol=0.8)

                next_state = env.get_state()
                curr_dist  = env.get_distance()
                reward, done = env.compute_reward(prev_dist, curr_dist)

                state, prev_dist = next_state, curr_dist
                ep_reward += reward
                step += 1

            env.stop()

            final_dist   = env.get_distance()
            intercepted  = final_dist < cfg['drone']['intercept_threshold']
            if intercepted:
                intercepts += 1

            rewards.append(ep_reward)
            steps_list.append(step)
            final_dists.append(final_dist)

            status = 'INTERCEPT!' if intercepted else f'dist={final_dist:.2f}m'
            print(f'  ep {ep+1:3d} | steps={step:3d} | '
                  f'reward={ep_reward:8.2f} | {status}', flush=True)

        # summary
        intercept_rate = intercepts / parsed.episodes
        all_results[difficulty] = {
            'intercept_rate':  intercept_rate,
            'avg_reward':      float(np.mean(rewards)),
            'avg_steps':       float(np.mean(steps_list)),
            'avg_final_dist':  float(np.mean(final_dists)),
        }

        print(f'\n  --- {difficulty.upper()} Summary ---')
        print(f'  Intercept rate : {intercept_rate:.1%} '
              f'({intercepts}/{parsed.episodes})')
        print(f'  Avg reward     : {np.mean(rewards):.2f}')
        print(f'  Avg steps      : {np.mean(steps_list):.1f}')
        print(f'  Avg final dist : {np.mean(final_dists):.2f}m')

    # final comparison table
    print(f'\n{"="*50}')
    print('[EVAL] FINAL RESULTS')
    print(f'{"="*50}')
    print(f'{"Difficulty":<12} {"Intercept%":<14} '
          f'{"Avg Reward":<14} {"Avg Dist":<10}')
    print('-' * 52)
    for diff, res in all_results.items():
        print(f'{diff:<12} {f"{res['intercept_rate']:.1%}":<14} '
              f'{res["avg_reward"]:<14.2f} {res["avg_final_dist"]:<10.2f}m')

    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
