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

Checkpoint included at `checkpoints/ep300.pt` (832KB, step 40636).

---

## Repository Structure

```
drone-pursuit-dqn/
├── drone_pursuit/
│   ├── train.py          # DQN training loop
│   ├── env.py            # ROS 2 environment node
│   ├── dqn_agent.py      # Double DQN + Prioritized Experience Replay
│   ├── perception.py     # HSV + YOLOv8n detection
│   ├── evader_node.py    # Random-walk evader node
│   └── evaluate.py       # Evaluation script
├── config/dqn_config.yaml
├── worlds/pursuit_world.sdf
├── models/               # Iris + Crazyflie meshes
├── checkpoints/ep300.pt  # Trained model
├── runs/                 # TensorBoard logs
├── Dockerfile            # Full stack container
├── docker-compose.yml    # 6-service compose file
└── docker/entrypoint.sh  # Container entrypoint
```

---

## Option 1 — Docker (Full Stack)

The Dockerfile builds a complete self-contained image with Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic 8.11, ArduPilot SITL (compiled from source), ardupilot_gazebo plugin (compiled from source), MAVROS2, and PyTorch.

### Build (30-40 minutes first time)

```bash
git clone https://github.com/masumt2808/drone-pursuit-dqn
cd drone-pursuit-dqn
docker build -t drone-pursuit-dqn-full .
```

### Run — 6 separate terminals

**Terminal 1 — Gazebo:**
```bash
xhost +local:docker
docker compose up gazebo
```

**Terminal 2 — ArduPilot SITL** (after Gazebo loads):
```bash
docker compose up ardupilot
```

**Terminal 3 — MAVROS** (after ArduPilot shows `ArduPilot Ready`):
```bash
docker compose up mavros
```

**Terminal 4 — Arm and Takeoff** (after MAVROS shows `Plugin local_position initialized`):
```bash
docker exec dqn_mavros bash -c "
source /opt/ros/jazzy/setup.bash &&
source /drone_ws/install/setup.bash &&
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
  '{base_mode: 0, custom_mode: GUIDED}' &&
sleep 3 &&
ros2 service call /mavros/cmd/arming \
  mavros_msgs/srv/CommandBool '{value: true}' &&
sleep 3 &&
ros2 service call /mavros/cmd/takeoff \
  mavros_msgs/srv/CommandTOL \
  '{min_pitch: 0.0, yaw: 0.0, latitude: 0.0, longitude: 0.0, altitude: 3.0}'"
```

**Terminal 5 — Evader** (after drone takes off):
```bash
docker compose up evader
```

**Terminal 6 — Evaluate** (run from inside mavros container):
```bash
docker exec -it dqn_mavros bash -c "
source /opt/ros/jazzy/setup.bash &&
source /drone_ws/install/setup.bash &&
python3 -u /drone_ws/src/drone_pursuit/drone_pursuit/evaluate.py \
  --checkpoint /drone_ws/src/drone_pursuit/checkpoints/ep300.pt \
  --difficulty static"
```

### Train from scratch

```bash
docker exec -it dqn_mavros bash -c "
source /opt/ros/jazzy/setup.bash &&
source /drone_ws/install/setup.bash &&
python3 -u /drone_ws/src/drone_pursuit/drone_pursuit/train.py"
```

---

## Option 2 — Run Directly on Host

### Requirements
- Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic 8.11
- ArduPilot SITL (ArduCopter V4.8.0), MAVROS2
- PyTorch 2.5.1 + CUDA, Python 3.12

### Setup

```bash
cd ~/drone_pursuit_ws/src
git clone https://github.com/masumt2808/drone-pursuit-dqn drone_pursuit
cd ~/drone_pursuit_ws
colcon build --packages-select drone_pursuit --symlink-install
source install/setup.bash
pip3 install torch numpy==1.26.4 opencv-python ultralytics tensorboard pyyaml --break-system-packages
```

### Run (6 terminals)

**Terminal 1 — Gazebo:**
```bash
gz sim -r ~/drone_pursuit_ws/src/drone_pursuit/worlds/pursuit_world.sdf
```

**Terminal 2 — ArduPilot SITL:**
```bash
cd ~/ardupilot
sim_vehicle.py -v ArduCopter --model=JSON \
  --add-param-file=Tools/autotest/default_params/gazebo-iris.parm \
  --console --map
```
In MAVProxy console: `output add 127.0.0.1:14551`

**Terminal 3 — MAVROS:**
```bash
ros2 run mavros mavros_node --ros-args \
  -p fcu_url:=udp://127.0.0.1:14551@14555 \
  -p tgt_system:=1 -p tgt_component:=1 -p system_id:=255
```

**Terminal 4 — Arm and Takeoff:**
```bash
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
  "{base_mode: 0, custom_mode: 'GUIDED'}"
sleep 2
ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"
sleep 2
ros2 service call /mavros/cmd/takeoff mavros_msgs/srv/CommandTOL \
  "{min_pitch: 0.0, yaw: 0.0, latitude: 0.0, longitude: 0.0, altitude: 3.0}"
sleep 6
```

**Terminal 5 — Evader:**
```bash
source ~/drone_pursuit_ws/install/setup.bash
ros2 run ros_gz_bridge parameter_bridge \
  /iris/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image &
sleep 2
ros2 run drone_pursuit evader_node \
  --ros-args -p speed:=0.5 -p start_x:=3.0 \
  -p start_y:=0.0 -p start_z:=3.0 -p use_gazebo:=True
```

**Terminal 6 — Evaluate:**
```bash
python3 -u ~/drone_pursuit_ws/src/drone_pursuit/drone_pursuit/evaluate.py \
  --checkpoint ~/drone_pursuit_ws/src/drone_pursuit/checkpoints/ep300.pt \
  --difficulty static
```

---

## TensorBoard

```bash
tensorboard --logdir ~/drone_pursuit_ws/runs --port 6006
```
Open `http://localhost:6006`

---

## Perception Mode

```yaml
# config/dqn_config.yaml
perception:
  mode: hsv   # 10-D state (default, fully trained)
  # mode: yolo  # 15-D state (converging)
```

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| State dim | 10-D (HSV) / 15-D (YOLO) |
| Actions | 6 discrete (±0.5m in x, y, z) |
| Network | MLP 10→256→256→6 |
| Learning rate | 0.0005 |
| Epsilon decay | 0.9998 per step |
| Replay buffer | 50,000 (Prioritized) |
| Intercept threshold | 1.0m |

---

## References

- Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature, 518.
- van Hasselt et al. (2016). Deep Reinforcement Learning with Double Q-learning. AAAI.
- Panerati et al. (2021). Learning to Fly — PyBullet Physics for RL. IROS.
- Chen et al. (2024). Online Planning for Multi-UAV Pursuit-Evasion. arXiv:2409.15866.
