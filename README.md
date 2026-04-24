# Vision-Based Autonomous Drone Pursuit Using Deep Q-Networks

**ENPM690 — Robot Learning | University of Maryland | Spring 2026**  
**Authors:** Masum Gautam Thakkar | Swathi Sree Annambhotla

---

## Overview

This project trains an Iris quadrotor (chaser) to autonomously intercept a red Crazyflie 2.x drone (evader) using a Deep Q-Network (DQN) in Gazebo Harmonic simulation. The chaser perceives the evader through a forward-facing RGB camera using HSV color detection and ground-truth odometry, learning a pursuit policy through reinforcement learning with 6 discrete position setpoint actions. Phase 1 achieved a 71.3% intercept rate over 150 training episodes, rising to 82% in the exploitation phase.

---

## System Requirements

- Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic 8.11
- ArduPilot SITL (ArduCopter V4.8.0), MAVROS2
- PyTorch 2.5.1 + CUDA, OpenCV 4.13

---

## Package Structure

```
drone_pursuit/
├── config/dqn_config.yaml        # All hyperparameters
├── drone_pursuit/
│   ├── dqn_agent.py              # DQN + Prioritized Experience Replay
│   ├── env.py                    # ROS2 pursuit environment node
│   ├── evader_node.py            # Random-walk evader with gz set_pose
│   ├── perception.py             # HSV detector + YOLOv8n detector
│   ├── train.py                  # Training loop with position setpoints
│   └── evaluate.py               # 3-tier evaluation script
├── models/
│   ├── iris_with_camera/         # Iris drone with forward RGB camera
│   ├── crazyflie_red/            # Crazyflie 2.x real mesh model (red)
│   └── meshes/                   # Collada mesh files
└── worlds/pursuit_world.sdf      # Gazebo pursuit world
```

---

## Running the Simulation

**Terminal 1 — Gazebo:**
```bash
source /opt/ros/jazzy/setup.bash
gz sim -r worlds/pursuit_world.sdf
```

**Terminal 2 — ArduPilot SITL:**
```bash
cd ~/ardupilot
sim_vehicle.py -v ArduCopter --model=JSON \
  --add-param-file=Tools/autotest/default_params/gazebo-iris.parm \
  --add-param-file=Tools/autotest/default_params/sitl_custom.parm \
  --console --map
```
Then in MAVProxy console: `output add 127.0.0.1:14551`

**Terminal 3 — MAVROS:**
```bash
source /opt/ros/jazzy/setup.bash
ros2 run mavros mavros_node --ros-args \
  -p fcu_url:=udp://127.0.0.1:14551@14555 \
  -p tgt_system:=1 -p tgt_component:=1 -p system_id:=255
```

**Terminal 4 — Arm and Takeoff:**
```bash
source /opt/ros/jazzy/setup.bash
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
  "{base_mode: 0, custom_mode: 'GUIDED'}"
sleep 2
ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"
sleep 2
ros2 service call /mavros/cmd/takeoff mavros_msgs/srv/CommandTOL \
  "{min_pitch: 0.0, yaw: 0.0, latitude: 0.0, longitude: 0.0, altitude: 3.0}"
```

**Terminal 5 — Camera Bridge and Evader:**
```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 run ros_gz_bridge parameter_bridge \
  /iris/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image &
sleep 2
ros2 run drone_pursuit evader_node \
  --ros-args -p speed:=0.5 -p start_x:=3.0 \
  -p start_y:=0.0 -p start_z:=3.0 -p use_gazebo:=True
```

**Terminal 6 — Training:**
```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash

# Fresh training
PYTHONUNBUFFERED=1 python3 -u drone_pursuit/train.py

# Resume from checkpoint
PYTHONUNBUFFERED=1 python3 -u drone_pursuit/train.py --checkpoint models/ep100.pt

# TensorBoard
tensorboard --logdir runs/pursuit_dqn --port 6006
```

---

## Architecture

**State Vector (10-D)**

| Index | Feature | Source |
|-------|---------|--------|
| 0-2 | rel_x, rel_y, rel_z | /evader/odom |
| 3-5 | vel_x, vel_y, vel_z | /mavros/local_position/odom |
| 6 | distance | Euclidean to evader |
| 7 | heading_err | Yaw error toward evader |
| 8 | alt_err | Altitude difference |
| 9 | vision_bit | HSV detection (0 or 1) |

**DQN:** 3-layer MLP (10-256-256-6) with ReLU, Prioritized Experience Replay, target network sync every 500 steps, CUDA accelerated.

**Actions:** 6 discrete position offsets of +/-1.5m in x, y, z directions.

**Reward:** Distance shaping (+10 per meter closed) + vision bonus (+5) + intercept (+200) + boundary penalty (-200) + step penalty (-0.5).

---

## Results

| Metric | Value |
|--------|-------|
| Episodes completed | 150 |
| Total intercepts | 107 / 150 |
| Overall intercept rate | 71.3% |
| Early rate (ep 1-50) | ~40% |
| Late rate (ep 51-150) | ~82% |
| Peak avg50 reward | +192 |
| Final epsilon | 0.050 |

---

## Phase 2 Plan

- Option B: YOLOv8n spatial detector replacing HSV (14-D state vector)
- Extended training using gym-pybullet-drones for faster iteration
- 3-tier evaluation across static, slow (0.3 m/s), and fast (0.7 m/s) evader
- Comparison between Option A (HSV) and Option B (YOLOv8n)
- Final report with full results and analysis

---

## References

- Chen et al. (2025). Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep Reinforcement Learning. arXiv:2409.15866.
- Panerati et al. (2021). Learning to Fly - a Gym Environment with PyBullet Physics for RL of Multi-agent Quadcopter Control. IROS 2021.
- Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature, 518, 529-533.
- Bitcraze AB. (2024). Crazyflie Simulation. github.com/bitcraze/crazyflie-simulation.
