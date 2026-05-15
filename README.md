# Vision-Based Autonomous Drone Pursuit Using Deep Q-Networks

**ENPM690 — Robot Learning | Spring 2026 | University of Maryland**

Masum Gautam Thakkar (121229076) · Swathi Sree Annambhotla (122257627)

---

## Results

| Difficulty | Speed | Intercept Rate | Avg Steps | Avg Reward |
|-----------|-------|---------------|-----------|------------|
| Static | 0.0 m/s | **100%** (20/20) | 78.2 | +940 |
| Slow | 0.3 m/s | **95%** (19/20) | 98.3 | +1137 |
| Fast | 0.5 m/s | **75%** (15/20) | 99.1 | +819 |

---

## Trained Model

The checkpoint is included in this repo at `checkpoints/ep300.pt` (832KB).

---

## Repository Structure

```
drone-pursuit-dqn/
├── drone_pursuit/
│   ├── train.py          # DQN training loop
│   ├── env.py            # ROS 2 environment node (state, reward)
│   ├── dqn_agent.py      # Double DQN + Prioritized Experience Replay
│   ├── perception.py     # HSV detector + YOLOv8n detector
│   ├── evader_node.py    # Random-walk evader drone node
│   └── evaluate.py       # 3-tier evaluation script
├── config/
│   └── dqn_config.yaml   # All hyperparameters
├── worlds/
│   └── pursuit_world.sdf # Gazebo world
├── models/               # Iris + Crazyflie meshes
├── checkpoints/
│   └── ep300.pt          # Trained HSV policy (832KB)
├── runs/                 # TensorBoard logs
├── launch/sim.launch.py
├── Dockerfile            # DQN code container
└── README.md
```

---

## System Requirements

- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic 8.11
- ArduPilot SITL (ArduCopter V4.8.0)
- MAVROS2
- PyTorch 2.5.1 + CUDA
- Python 3.12

---

## Option 1 — Run Directly (Recommended)

### 1. Clone and build

```bash
cd ~/drone_pursuit_ws/src
git clone https://github.com/masumt2808/drone-pursuit-dqn drone_pursuit
cd ~/drone_pursuit_ws
colcon build --packages-select drone_pursuit --symlink-install
source install/setup.bash
```

### 2. Install Python dependencies

```bash
pip3 install torch numpy==1.26.4 opencv-python ultralytics tensorboard pyyaml --break-system-packages
```

### 3. Start simulation (6 terminals)

**Terminal 1 — Gazebo:**
```bash
source /opt/ros/jazzy/setup.bash
gz sim -r ~/drone_pursuit_ws/src/drone_pursuit/worlds/pursuit_world.sdf
```

**Terminal 2 — ArduPilot SITL:**
```bash
cd ~/ardupilot
sim_vehicle.py -v ArduCopter --model=JSON \
  --add-param-file=Tools/autotest/default_params/gazebo-iris.parm \
  --console --map
```
In MAVProxy console type: `output add 127.0.0.1:14551`

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
sleep 6
```

**Terminal 5 — Camera Bridge + Evader:**
```bash
source /opt/ros/jazzy/setup.bash
source ~/drone_pursuit_ws/install/setup.bash
ros2 run ros_gz_bridge parameter_bridge \
  /iris/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image &
sleep 2
ros2 run drone_pursuit evader_node \
  --ros-args -p speed:=0.5 -p start_x:=3.0 \
  -p start_y:=0.0 -p start_z:=3.0 -p use_gazebo:=True
```

**Terminal 6 — Evaluate trained policy:**
```bash
source /opt/ros/jazzy/setup.bash
source ~/drone_pursuit_ws/install/setup.bash
python3 -u ~/drone_pursuit_ws/src/drone_pursuit/drone_pursuit/evaluate.py \
  --checkpoint ~/drone_pursuit_ws/src/drone_pursuit/checkpoints/ep300.pt \
  --difficulty static
```

**Or train from scratch:**
```bash
python3 -u ~/drone_pursuit_ws/src/drone_pursuit/drone_pursuit/train.py
```

---

## Option 2 — Docker (DQN Code Container)

The Dockerfile containerizes the DQN training and evaluation code.
Gazebo, ArduPilot SITL, and MAVROS must run on the host machine first (Terminals 1-5 above).
The container connects to host ROS 2 topics via `--network host`.

### Build

```bash
docker build -t drone-pursuit-dqn .
```

### Evaluate (start Terminals 1-5 on host first, then run this)

```bash
docker run --rm --network host \
  -v ~/drone_pursuit_ws/src/drone_pursuit/checkpoints:/checkpoints:ro \
  drone-pursuit-dqn eval \
  --checkpoint /checkpoints/ep300.pt \
  --difficulty static
```

### Run all difficulties

```bash
docker run --rm --network host \
  -v ~/drone_pursuit_ws/src/drone_pursuit/checkpoints:/checkpoints:ro \
  drone-pursuit-dqn eval \
  --checkpoint /checkpoints/ep300.pt \
  --difficulty all
```

### Train from scratch

```bash
docker run --rm --network host --gpus all \
  -v ~/drone_pursuit_ws/models:/checkpoints \
  -v ~/drone_pursuit_ws/runs:/runs \
  drone-pursuit-dqn train
```

### TensorBoard

```bash
docker run --rm --network host \
  -v ~/drone_pursuit_ws/runs:/runs:ro \
  drone-pursuit-dqn tensorboard
```
Open `http://localhost:6006`

### Interactive shell

```bash
docker run --rm -it --network host drone-pursuit-dqn bash
```

---

## TensorBoard

```bash
tensorboard --logdir ~/drone_pursuit_ws/runs --port 6006
```
Open `http://localhost:6006` — the `runs/` folder in this repo contains all training logs.

---

## Switching Perception Mode

In `config/dqn_config.yaml`:

```yaml
perception:
  mode: hsv    # 10-D state — fully trained (100%/95%/75%)
  # mode: yolo # 15-D state — spatial bbox features, still converging
```

> Do not load an HSV checkpoint when running YOLO mode — state dimensions differ (10 vs 15).

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| State dim | 10-D (HSV) / 15-D (YOLO) |
| Actions | 6 discrete (±0.5m in x, y, z) |
| Network | MLP 10→256→256→6, ReLU |
| Learning rate | 0.0005 |
| Gamma | 0.99 |
| Epsilon decay | 0.9998 per step |
| Batch size | 64 |
| Replay buffer | 50,000 (Prioritized) |
| Target sync | Every 500 steps |
| Intercept threshold | 1.0m |

---

## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature, 518.
- van Hasselt et al. (2016). Deep Reinforcement Learning with Double Q-learning. AAAI.
- Panerati et al. (2021). Learning to Fly — PyBullet Physics for RL. IROS.
- Chen et al. (2024). Online Planning for Multi-UAV Pursuit-Evasion. arXiv:2409.15866.
- Bitcraze AB (2024). Crazyflie Simulation. github.com/bitcraze/crazyflie-simulation.
