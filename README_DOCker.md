## Running with Docker (ROS2 Humble)

### Build
```bash
docker build -f Dockerfile.humble -t drone-pursuit-dqn .
```

### Run
```bash
xhost +local:docker
docker run -it \
  --env DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --network host \
  drone-pursuit-dqn
```

### Terminal 1 — Gazebo
```bash
gz sim -r /drone_pursuit_ws/src/drone_pursuit/worlds/pursuit_world.sdf
```

### Terminal 2 — ArduPilot
```bash
docker exec -it $(docker ps -q) bash
cd /ardupilot
python3 Tools/autotest/sim_vehicle.py -v ArduCopter --model=JSON \
  --add-param-file=Tools/autotest/default_params/gazebo-iris.parm \
  --out 127.0.0.1:14551 --no-console --no-map
```

### Terminal 3 — MAVROS
```bash
docker exec -it $(docker ps -q) bash
source /opt/ros/humble/setup.bash
ros2 run mavros mavros_node --ros-args \
  -p fcu_url:=udp://127.0.0.1:14551@14555 \
  -p tgt_system:=1 -p tgt_component:=1 -p system_id:=255
```

### Terminal 4 — Arm and Takeoff
```bash
docker exec -it $(docker ps -q) bash
source /opt/ros/humble/setup.bash
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
  "{base_mode: 0, custom_mode: 'GUIDED'}"
sleep 2
ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"
sleep 2
ros2 service call /mavros/cmd/takeoff mavros_msgs/srv/CommandTOL \
  "{min_pitch: 0.0, yaw: 0.0, latitude: 0.0, longitude: 0.0, altitude: 3.0}"
```

### Terminal 5 — Camera Bridge and Evader
```bash
docker exec -it $(docker ps -q) bash
source /opt/ros/humble/setup.bash
source /drone_pursuit_ws/install/setup.bash
ros2 run ros_gz_bridge parameter_bridge \
  /iris/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image &
sleep 2
ros2 run drone_pursuit evader_node \
  --ros-args -p speed:=0.5 -p start_x:=3.0 \
  -p start_y:=0.0 -p start_z:=3.0 -p use_gazebo:=True
```

### Terminal 6 — Training
```bash
docker exec -it $(docker ps -q) bash
source /opt/ros/humble/setup.bash
source /drone_pursuit_ws/install/setup.bash
cd /drone_pursuit_ws/src/drone_pursuit
PYTHONUNBUFFERED=1 python3 -u drone_pursuit/train.py

# Resume from checkpoint
PYTHONUNBUFFERED=1 python3 -u drone_pursuit/train.py --checkpoint models/ep150.pt

# TensorBoard
tensorboard --logdir runs/pursuit_dqn --port 6006
```
