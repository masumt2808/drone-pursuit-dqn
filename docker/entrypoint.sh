#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
source /drone_ws/install/setup.bash
export GZ_SIM_RESOURCE_PATH=/usr/local/share/ardupilot_gazebo/models:/drone_ws/src/drone_pursuit/models
export GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/local/lib/ardupilot_gazebo
export PATH=/ardupilot/build/sitl/bin:$PATH

CMD="${1:-help}"
shift 2>/dev/null || true

case "$CMD" in
    sim)
        echo "=== FULL SIMULATION ==="
        rm -f /tmp/.X99-lock
        Xvfb :99 -screen 0 1280x1024x24 &
        export DISPLAY=:99
        sleep 2
        echo "[1/6] Gazebo..."
        gz sim -r -s /drone_ws/src/drone_pursuit/worlds/pursuit_world.sdf &
        sleep 8
        echo "[2/6] ArduPilot SITL..."
        cd /ardupilot
        python3 Tools/autotest/sim_vehicle.py \
            -v ArduCopter --model=JSON --no-rebuild \
            --add-param-file=Tools/autotest/default_params/gazebo-iris.parm \
            --out=127.0.0.1:14551 &
        sleep 20
        echo "[3/6] MAVROS..."
        ros2 run mavros mavros_node --ros-args \
            -p fcu_url:=udp://127.0.0.1:14551@14555 \
            -p tgt_system:=1 -p tgt_component:=1 -p system_id:=255 &
        sleep 10
        echo "[4/6] Arm and Takeoff..."
        ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \
            '{base_mode: 0, custom_mode: GUIDED}'
        sleep 2
        ros2 service call /mavros/cmd/arming \
            mavros_msgs/srv/CommandBool '{value: true}'
        sleep 2
        ros2 service call /mavros/cmd/takeoff \
            mavros_msgs/srv/CommandTOL \
            '{min_pitch: 0.0, yaw: 0.0, latitude: 0.0, longitude: 0.0, altitude: 3.0}'
        sleep 8
        echo "[5/6] Camera bridge and evader..."
        ros2 run ros_gz_bridge parameter_bridge \
            /iris/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image &
        sleep 3
        ros2 run drone_pursuit evader_node --ros-args \
            -p speed:=0.5 -p start_x:=3.0 \
            -p start_y:=0.0 -p start_z:=3.0 -p use_gazebo:=True &
        sleep 3
        echo "[6/6] Running DQN..."
        exec python3 -u \
            /drone_ws/src/drone_pursuit/drone_pursuit/evaluate.py "$@"
        ;;
    train)
        exec python3 -u /drone_ws/src/drone_pursuit/drone_pursuit/train.py "$@"
        ;;
    eval|evaluate)
        exec python3 -u /drone_ws/src/drone_pursuit/drone_pursuit/evaluate.py "$@"
        ;;
    tensorboard|tb)
        exec tensorboard --logdir /runs --host 0.0.0.0 --port 6006
        ;;
    bash|shell)
        exec /bin/bash "$@"
        ;;
    *)
        echo ""
        echo "drone-pursuit-dqn — ENPM690 Spring 2026"
        echo "GitHub: https://github.com/masumt2808/drone-pursuit-dqn"
        echo ""
        echo "Commands:"
        echo "  sim   --checkpoint /checkpoints/ep300.pt --difficulty static"
        echo "  train"
        echo "  train --checkpoint /checkpoints/ep300.pt"
        echo "  eval  --checkpoint /checkpoints/ep300.pt --difficulty static"
        echo "  tensorboard"
        echo "  bash"
        echo ""
        echo "Run full simulation:"
        echo "  xhost +local:docker"
        echo "  docker run -it --rm --network host \\"
        echo "    -e DISPLAY=\$DISPLAY \\"
        echo "    -v /tmp/.X11-unix:/tmp/.X11-unix \\"
        echo "    -v ~/drone_pursuit_ws/models:/checkpoints \\"
        echo "    -v ~/drone_pursuit_ws/runs:/runs \\"
        echo "    drone-pursuit-dqn-full sim \\"
        echo "    --checkpoint /checkpoints/ep300.pt --difficulty static"
        ;;
esac
