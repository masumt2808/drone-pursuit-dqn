"""
Stable DQN training using POSITION setpoints instead of velocity.
Based on best practices from AirSim RL and ArduPilot GUIDED mode docs.

Key design decisions from literature:
1. Actions = discrete POSITION OFFSETS (not velocity) — ArduPilot 
   handles stable flight to each position target automatically
2. No downward action — altitude maintained by ArduPilot
3. GUID_TIMEOUT=30 — drone holds position between commands
4. Episode = 200 steps max (shorter = faster learning)
5. Reward = exponential of distance (from IvLabs paper)
"""
import rclpy
import yaml, time, os, numpy as np
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
from torch.utils.tensorboard import SummaryWriter
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
import subprocess

from drone_pursuit.env import PursuitEnv
from drone_pursuit.dqn_agent import DQNAgent
from drone_pursuit.perception import HSVDetector


def load_cfg():
    p = os.path.expanduser(
        '~/drone_pursuit_ws/src/drone_pursuit/config/dqn_config.yaml')
    with open(p) as f:
        return yaml.safe_load(f)


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parsed, _ = parser.parse_known_args()

    rclpy.init(args=args)
    cfg = load_cfg()
    print('[TRAIN] Config loaded', flush=True)

    perception = HSVDetector()
    env = PursuitEnv(cfg, perception)
    executor = MultiThreadedExecutor()
    executor.add_node(env)
    Thread(target=executor.spin, daemon=True).start()

    # position setpoint publisher — stable ArduPilot control
    pos_pub = env.create_publisher(
        PoseStamped, '/mavros/setpoint_position/local', 10)
    evader_reset_client = env.create_client(Trigger, '/evader/reset')

    def send_position(x, y, z):
        """Send a single position setpoint."""
        msg = PoseStamped()
        msg.header.stamp    = env.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        pos_pub.publish(msg)

    def goto_position(x, y, z, wait=4.0, tol=0.5):
        """
        Fly to position and wait until arrived or timeout.
        This is stable — ArduPilot handles the flight.
        """
        t0 = time.time()
        while time.time() - t0 < wait:
            send_position(x, y, z)
            time.sleep(0.1)
            curr = env.chaser_pos
            dist = np.linalg.norm(curr - np.array([x, y, z]))
            if dist < tol:
                break

    def ensure_flying(min_z=1.5):
        """Check altitude and rearm+takeoff if drone is on ground."""
        curr_z = env.chaser_pos[2]
        if curr_z < min_z:
            print(f'[TRAIN] Drone too low (z={curr_z:.2f}) — rearming...', flush=True)
            import subprocess
            subprocess.run(['ros2', 'service', 'call', '/mavros/set_mode',
                'mavros_msgs/srv/SetMode',
                '{base_mode: 0, custom_mode: GUIDED}'],
                capture_output=True, timeout=5)
            time.sleep(1)
            subprocess.run(['ros2', 'service', 'call', '/mavros/cmd/arming',
                'mavros_msgs/srv/CommandBool', '{value: true}'],
                capture_output=True, timeout=5)
            time.sleep(2)
            subprocess.run(['ros2', 'service', 'call', '/mavros/cmd/takeoff',
                'mavros_msgs/srv/CommandTOL',
                '{min_pitch: 0.0, yaw: 0.0, latitude: 0.0, longitude: 0.0, altitude: 3.0}'],
                capture_output=True, timeout=5)
            time.sleep(8)
            print(f'[TRAIN] Rearm done. z={env.chaser_pos[2]:.2f}', flush=True)
            return env.chaser_pos[2] > min_z
        return True

    def reset_evader_random():
        angle = np.random.uniform(0, 2 * np.pi)
        dist  = np.random.uniform(3.0, 6.0)
        x = float(dist * np.cos(angle))
        y = float(dist * np.sin(angle))
        z = 3.0
        req = (f'name: "crazyflie" '
               f'position: {{x: {x:.2f}, y: {y:.2f}, z: {z:.2f}}} '
               f'orientation: {{w: 1}}')
        try:
            subprocess.run(
                ['gz', 'service', '-s', '/world/pursuit_world/set_pose',
                 '--reqtype', 'gz.msgs.Pose',
                 '--reptype', 'gz.msgs.Boolean',
                 '--timeout', '500', '--req', req],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=1.0)
        except Exception:
            pass
        if evader_reset_client.wait_for_service(timeout_sec=0.5):
            evader_reset_client.call_async(Trigger.Request())
        time.sleep(0.3)
        return x, y, z

    # Action space — position OFFSETS (not velocity)
    # Each action moves drone by STEP metres in one direction
    STEP = 1.0  # metres per action
    ACTION_OFFSETS = np.array([
        [ STEP, 0,     0    ],  # forward
        [-STEP, 0,     0    ],  # backward
        [0,     STEP,  0    ],  # left
        [0,    -STEP,  0    ],  # right
        [0,     0,     STEP ],  # up
        [0,     0,    -STEP ],  # down
    ], dtype=np.float32)

    agent = DQNAgent(cfg['agent'])
    if parsed.checkpoint and os.path.isfile(parsed.checkpoint):
        agent.load(parsed.checkpoint)
        print(f'[TRAIN] Loaded checkpoint: {parsed.checkpoint}', flush=True)
    else:
        print('[TRAIN] Starting from scratch', flush=True)
    os.makedirs(
        os.path.expanduser('~/drone_pursuit_ws/models/'), exist_ok=True)
    ckpt_dir = os.path.expanduser('~/drone_pursuit_ws/models/')
    writer   = SummaryWriter(
        os.path.expanduser('~/drone_pursuit_ws/runs/pursuit_dqn'))

    print('[TRAIN] Waiting for drone + evader...', flush=True)
    t0 = time.time()
    while not env.is_ready():
        time.sleep(0.1)
        if time.time() - t0 > 30:
            print('[TRAIN] ERROR: timeout')
            rclpy.shutdown()
            return

    print(f'[TRAIN] Ready! dist={env.get_distance():.2f}m', flush=True)

    tcfg  = cfg['training']
    dcfg  = cfg['drone']
    HOME  = np.array([0.0, 0.0, 3.0])
    ep_rewards, intercepts = [], 0

    for episode in range(tcfg['max_episodes']):

        # --- reset ---
        # 0. ensure drone is flying
        ensure_flying()
        # 1. stop any velocity
        env.stop()
        # 2. fly home
        goto_position(*HOME, wait=4.0, tol=0.8)
        # 3. random evader position
        ex, ey, ez = reset_evader_random()

        # current drone position as base for actions
        drone_pos = env.chaser_pos.copy()

        state     = env.get_state()
        prev_dist = env.get_distance()
        ep_reward, done, step = 0.0, False, 0

        print(f'[EP {episode+1:4d}] evader=({ex:.1f},{ey:.1f}) '
              f'dist={prev_dist:.2f}m vision={env.vision_bit}',
              flush=True)

        while not done and step < tcfg['max_steps_per_episode']:
            drone_pos = env.chaser_pos.copy()  # always use real pos
            action = agent.select_action(state)

            # compute new target position
            offset    = ACTION_OFFSETS[action]
            new_pos   = drone_pos + offset
            # clamp altitude 2.0–5.0m
            new_pos[2] = float(np.clip(new_pos[2], 2.0, 5.0))
            # clamp XY to boundary
            new_pos[0] = float(np.clip(new_pos[0], -12, 12))
            new_pos[1] = float(np.clip(new_pos[1], -12, 12))

            # send position setpoint and wait briefly
            goto_position(*new_pos, wait=0.5, tol=0.8)
            drone_pos = env.chaser_pos.copy()  # always use real position

            next_state = env.get_state()
            curr_dist  = env.get_distance()
            reward, done = env.compute_reward(prev_dist, curr_dist)

            agent.store(state, action, reward, next_state, float(done))
            agent.update()

            state, prev_dist = next_state, curr_dist
            ep_reward += reward
            step += 1

        env.stop()

        intercepted = env.get_distance() < dcfg['intercept_threshold']
        if intercepted:
            intercepts += 1

        ep_rewards.append(ep_reward)
        avg50 = float(np.mean(ep_rewards[-50:]))

        writer.add_scalar('reward/episode',   ep_reward,             episode)
        writer.add_scalar('reward/avg50',     avg50,                 episode)
        writer.add_scalar('agent/epsilon',    agent.epsilon,         episode)
        writer.add_scalar('training/intercept_rate',
                          intercepts/(episode+1),                    episode)

        print(f'Ep {episode+1:4d} | steps={step:3d} | '
              f'dist={env.get_distance():.2f}m | '
              f'reward={ep_reward:8.2f} | avg50={avg50:8.2f} | '
              f'eps={agent.epsilon:.3f} | intercepts={intercepts}'
              f'{" INTERCEPT!" if intercepted else ""}',
              flush=True)

        if (episode+1) % tcfg['save_every'] == 0:
            ckpt = os.path.join(ckpt_dir, f'ep{episode+1}.pt')
            agent.save(ckpt)

    writer.close()
    executor.shutdown()
    rclpy.shutdown()
    print('[TRAIN] Done', flush=True)


if __name__ == '__main__':
    main()
