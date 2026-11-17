# Entorno de simulación lite6_zed: Ubuntu 24.04 + ROS 2 Jazzy + MoveIt 2 + Gazebo Harmonic.
# Build (desde la raíz del repo): docker compose -f docker/compose.yaml build
FROM osrf/ros:jazzy-desktop-full

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy \
    WS=/ws

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3-vcstool python3-colcon-common-extensions python3-rosdep \
        python3-opencv python3-scipy python3-yaml python3-pytest python3-pyflakes python3-pip \
        ros-jazzy-moveit ros-jazzy-gz-ros2-control ros-jazzy-ros-gz \
        ros-jazzy-rqt-image-view ros-jazzy-tf2-tools \
        mesa-utils x11-apps \
    && pip3 install --no-cache-dir --break-system-packages usd-core "onnxruntime-gpu[cuda,cudnn]" ruckig fastapi "uvicorn[standard]" \
    && rm -rf /var/lib/apt/lists/*

# Dependencias externas (capa cacheable: solo cambia si cambia sim.repos)
WORKDIR ${WS}
COPY docker/sim.repos /tmp/sim.repos
RUN mkdir -p src && vcs import --recursive --shallow src < /tmp/sim.repos

# Paquete propio (en runtime se monta encima desde el host)
COPY src/lite6_handeye src/lite6_handeye

# Sin rosdep: su base de datos vive en raw.githubusercontent.com, bloqueado en algunas redes.
COPY docker/install_deps.py /usr/local/bin/install_deps.py
RUN apt-get update \
    && python3 /usr/local/bin/install_deps.py src \
    && rm -rf /var/lib/apt/lists/*

RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
    && colcon build --symlink-install \
        --packages-up-to xarm_moveit_config xarm_gazebo xarm_api easy_handeye2 lite6_handeye \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && chown -R ubuntu:ubuntu ${WS}

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# uid/gid 1000 = usuario 'ubuntu' de la imagen base (coincide con el usuario del host)
USER ubuntu
ENV HOME=/home/ubuntu
RUN mkdir -p ~/.ros/lite6_handeye \
    && echo 'source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash' >> ~/.bashrc
WORKDIR ${WS}
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
