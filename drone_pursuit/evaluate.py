import rclpy
from rclpy.executors import MultiThreadedExecutor
import yaml, time, os, sys
import numpy as np
from threading import Thread

from drone_pursuit.env import PursuitEnv
from drone_pursuit.dqn_agent import DQNAgent
from drone_pursuit.perception import HSVDetector

SPEEDS = {'static': 0.0, 'slow': 0.5, 'fast': 1.5}
EVAL_EPISODES = 50


def load_cfg():
    p = os.path.expanduser(
        '~/drone_pursuit_ws/src/drone_pursuit/config/dqn_config.yaml'
    )
    with open(p) as f:
        return yaml.safe_load(f)


def main(args=None):
    if len(sys.argv) < 2:
        print('Usage: ros2 run drone_pursuit evaluate <checkpoint.pt>')
        return

    checkpoint = sys.argv[1]
    rclpy.init(args=args)
    cfg = load_cfg()

    perception = HSVDetector()
    env = PursuitEnv(cfg, perception)

    executor = MultiThreadedExecutor()
    executor.add_node(env)
    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    agent = DQNAgent(cfg['agent'])
    agent.load(checkpoint)
    agent.epsilon = 0.0   # greedy — no exploration at eval time

    print(f'[EVAL] Loaded checkpoint: {checkpoint}')
    print('[EVAL] Waiting for drone...')
    while not env.is_ready():
        time.sleep(0.1)

    results = {}
    tcfg = cfg['training']
    dcfg = cfg['drone']

    for difficulty, speed in SPEEDS.items():
        print(f'\n[EVAL] --- {difficulty.upper()} evader (speed={speed}) ---')
        intercepts, steps_list, rewards = 0, [], []

        for ep in range(EVAL_EPISODES):
            state     = env.get_state()
            prev_dist = env.get_distance()
            ep_reward = 0.0
            done      = False
            step      = 0

            while not done and step < tcfg['max_steps_per_episode']:
                action = agent.select_action(state)
                vel    = agent.action_to_velocity(
                    action, dcfg['velocity_magnitude']
                )
                env.publish_velocity(vel)
                time.sleep(0.05)

                next_state = env.get_state()
                curr_dist  = env.get_distance()
                reward, done = env.compute_reward(prev_dist, curr_dist)

                state     = next_state
                prev_dist = curr_dist
                ep_reward += reward
                step      += 1

            env.stop()

            if done and env.get_distance() < dcfg['intercept_threshold']:
                intercepts += 1
            steps_list.append(step)
            rewards.append(ep_reward)

            print(f'  ep {ep+1:3d} | '
                  f'reward={ep_reward:8.2f} | '
                  f'steps={step} | '
                  f'intercepted={done and env.get_distance() < dcfg["intercept_threshold"]}')

        results[difficulty] = {
            'intercept_rate':         intercepts / EVAL_EPISODES,
            'avg_reward':             float(np.mean(rewards)),
            'avg_steps':              float(np.mean(steps_list)),
        }
        print(f'  → intercept rate : {results[difficulty]["intercept_rate"]:.2%}')
        print(f'  → avg reward     : {results[difficulty]["avg_reward"]:.2f}')
        print(f'  → avg steps      : {results[difficulty]["avg_steps"]:.1f}')

    print('\n[EVAL] Final results:')
    for d, r in results.items():
        print(f'  {d:8s}: {r["intercept_rate"]:.2%} intercepts | '
              f'avg reward {r["avg_reward"]:.2f}')

    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
