FROM ros:humble

# Add Gazebo Harmonic repo
RUN apt-get update && apt-get install -y wget lsb-release && \
    wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    libopencv-dev \
    ros-humble-mavros \
    ros-humble-mavros-extras \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-sim \
    gz-harmonic \
    libgz-sim8-dev \
    rapidjson-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    cmake \
    git \
    build-essential \
    python3-dev \
    python3-future \
    python3-setuptools \
    x11-apps \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install GeographicLib
RUN /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh

# Install Python dependencies  ← CHANGE 1
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install numpy opencv-python pyyaml tensorboard setuptools==58.2.0 mavproxy && \
    pip3 install "numpy<2"

# Install ArduPilot Gazebo plugin
RUN git clone https://github.com/ArduPilot/ardupilot_gazebo.git --depth 1 /ardupilot_gazebo && \
    cd /ardupilot_gazebo && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
    make -j4 && \
    make install && \
    rm -rf /ardupilot_gazebo

# Create non-root user for ArduPilot SITL
RUN useradd -m -s /bin/bash ardupilot && \
    echo "ardupilot ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install ArduPilot SITL as non-root user
RUN git clone https://github.com/ArduPilot/ardupilot.git \
    --recurse-submodules --depth 1 /ardupilot && \
    chown -R ardupilot:ardupilot /ardupilot

RUN su ardupilot -c "cd /ardupilot && \
    Tools/environment_install/install-prereqs-ubuntu.sh -y"

# Fix git ownership and pre-compile ArduCopter
RUN git config --global --add safe.directory /ardupilot && \
    su ardupilot -c "git config --global --add safe.directory /ardupilot" && \
    cd /ardupilot && \
    /ardupilot/modules/waf/waf-light configure --board sitl && \
    /ardupilot/modules/waf/waf-light build --target bin/arducopter

# Set environment variables 
ENV GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/local/lib/ardupilot_gazebo
ENV GZ_SIM_RESOURCE_PATH=/drone_pursuit_ws/src/drone_pursuit/models
ENV PATH=$PATH:/ardupilot/Tools/autotest
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV QT_X11_NO_MITSHM=1

# Create workspace
WORKDIR /drone_pursuit_ws/src
RUN git clone https://github.com/masumt2808/drone-pursuit-dqn.git drone_pursuit

# Fix hardcoded mesh paths
RUN sed -i 's|/home/swathi/drone_pursuit_ws/src/drone_pursuit/models|/drone_pursuit_ws/src/drone_pursuit/models|g' \
    /drone_pursuit_ws/src/drone_pursuit/models/crazyflie_red/model.sdf && \
    sed -i 's|/home/masum/drone_pursuit_ws/src/drone_pursuit/models|/drone_pursuit_ws/src/drone_pursuit/models|g' \
    /drone_pursuit_ws/src/drone_pursuit/models/crazyflie_red/model.sdf

# Download iris_with_standoffs
RUN git clone https://github.com/ArduPilot/ardupilot_gazebo.git \
    --depth 1 /tmp/ag && \
    cp -r /tmp/ag/models/iris_with_standoffs \
    /drone_pursuit_ws/src/drone_pursuit/models/ && \
    rm -rf /tmp/ag

# Build package
WORKDIR /drone_pursuit_ws
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install"

# Source setup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /drone_pursuit_ws/install/setup.bash" >> ~/.bashrc && \
    echo "export GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/local/lib/ardupilot_gazebo" >> ~/.bashrc && \
    echo "export GZ_SIM_RESOURCE_PATH=/drone_pursuit_ws/src/drone_pursuit/models" >> ~/.bashrc && \
    echo "export PATH=\$PATH:/ardupilot/Tools/autotest" >> ~/.bashrc

CMD ["/bin/bash"]
