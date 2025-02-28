# for a container that can run the vision node with SAM in a ros humble + CUDA 12.4 environment

# Use an NVIDIA CUDA image as the base
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="${PATH}:/home/user/.local/bin"

# We love UTF!
ENV LANG C.UTF-8

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Set the nvidia container runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME="/usr/local/cuda"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX 8.9"

# Install ROS Humble
RUN apt update && apt install locales -y && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    LANG=en_US.UTF-8

RUN apt install software-properties-common -y && \
    add-apt-repository universe


RUN apt update && apt install curl -y &&  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt update 

RUN apt upgrade -y

RUN apt install ros-humble-ros-base ros-dev-tools -y

# Install some handy tools. Even Guvcview for webcam support!
RUN set -x \
    && apt-get update \
    && apt-get install -y apt-transport-https ca-certificates \
    && apt-get install -y git vim tmux nano htop sudo curl wget gnupg2 \
    && apt-get install -y bash-completion \
    && apt-get install -y guvcview 

RUN set -x \
    && apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN set -x \
    && apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3-pip

# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
#     && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2


WORKDIR /home/user


RUN git clone https://github.com/facebookresearch/segment-anything-2

WORKDIR /home/user/segment-anything-2

RUN python3 -m pip install -e . -v && \
    python3 -m pip install -e ".[demo]"

WORKDIR /home/user/segment-anything-2/checkpoints

RUN ./download_ckpts.sh

WORKDIR /home/user

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

RUN /bin/bash -c "source /root/.bashrc"