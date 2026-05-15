FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV ROS_DISTRO=jazzy
ENV LANG=en_US.UTF-8

# 1. Base packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        locales curl wget gnupg2 lsb-release ca-certificates \
        software-properties-common apt-transport-https \
        git cmake build-essential ninja-build \
        python3-pip python3-dev \
        xvfb libgl1-mesa-dri libgles2 \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        python3-wxgtk4.0 python3-lxml python3-pexpect \
        python3-opencv libxml2-dev libxslt1-dev \
        sudo unzip zip ccache \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# 2. ROS 2 Jazzy + MAVROS + Gazebo bridge
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-desktop \
        ros-jazzy-mavros ros-jazzy-mavros-extras ros-jazzy-mavros-msgs \
        ros-jazzy-cv-bridge ros-jazzy-ros-gz-bridge ros-jazzy-ros-gz-sim \
        ros-jazzy-nav-msgs ros-jazzy-sensor-msgs ros-jazzy-geometry-msgs \
        ros-jazzy-std-msgs ros-jazzy-std-srvs \
        python3-colcon-common-extensions python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# 3. GeographicLib
RUN wget -q https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh \
        -O /tmp/geo.sh && bash /tmp/geo.sh && rm /tmp/geo.sh

# 4. Gazebo Harmonic
RUN wget -q https://packages.osrfoundation.org/gazebo.gpg \
        -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable noble main" \
        > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        gz-harmonic libgz-sim8-dev libgz-plugin2-dev libgz-common5-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. MAVProxy and Python deps
RUN pip3 install --break-system-packages --no-cache-dir \
        MAVProxy==1.8.74 pymavlink future empy==3.3.4 pexpect dronecan \
        "torch>=2.0" "numpy==1.26.4" ultralytics tensorboard pyyaml \
        opencv-python-headless

# 6. ArduPilot SITL
WORKDIR /ardupilot
RUN git clone --depth 1 https://github.com/ArduPilot/ardupilot.git . \
    && git submodule update --init --recursive --depth 1 \
    && Tools/environment_install/install-prereqs-ubuntu.sh -y || true \
    && ./waf configure --board sitl \
    && ./waf copter

# 7. ardupilot_gazebo plugin
WORKDIR /ardupilot_gazebo_build
RUN git clone --depth 1 https://github.com/ArduPilot/ardupilot_gazebo.git . \
    && /bin/bash -c "\
        source /opt/ros/jazzy/setup.bash && \
        cmake -B build -S . \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DCMAKE_INSTALL_PREFIX=/usr/local && \
        cmake --build build -j$(nproc) && \
        cmake --install build"

# 8. Build drone pursuit package
WORKDIR /drone_ws
COPY . /drone_ws/src/drone_pursuit/

RUN /bin/bash -c "\
    source /opt/ros/jazzy/setup.bash && \
    colcon build --packages-select drone_pursuit --symlink-install \
    2>&1 | tail -5"

RUN mkdir -p /root/drone_pursuit_ws/src/drone_pursuit && \
    ln -s /drone_ws/src/drone_pursuit/config \
          /root/drone_pursuit_ws/src/drone_pursuit/config && \
    mkdir -p /checkpoints /runs

# 9. Entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["help"]
