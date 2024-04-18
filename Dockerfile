
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# Parameters
ARG username
ARG password=docker
ARG userid

ENV DEBIAN_FRONTEND=nonintercative

# Create new user with same name as current user
RUN apt update -y && \
        apt upgrade -y && \
        apt install -y git htop vim wget python3 sudo tmux powerline tar unzip locales rsync libgl1-mesa-dev libglib2.0-dev && \
        apt install -y libboost-all-dev libcgal-dev libeigen3-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev && \
        apt clean && \
        locale-gen en_GB.UTF-8


# Set user as default user
RUN useradd -ms /bin/bash -u ${userid} ${username} && \
    echo "${username}:${password}" | chpasswd && \
    chown -R ${username}:${username} /home/${username} && \
    usermod -aG sudo ${username}


WORKDIR /home/${username}
USER ${username}


## Custom devel environment

ARG CONDA_VERSION="py39_24.1.2-0"
ARG CONDA_SHA="2ec135e4ae2154bb41e8df9ecac7ef23a7d6ca59fc1c8071cfe5298505c19140"
ARG CONDA_DIR="/home/${username}/conda"

ENV PATH="$CONDA_DIR/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1

# Install conda
RUN echo "**** install dev packages ****" && \
    echo "**** get Miniconda ****" && \
    mkdir -p "$CONDA_DIR" && \
    wget "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh" -O miniconda.sh && \
    echo "$CONDA_SHA  miniconda.sh" | sha256sum -c && \
    \
    echo "**** install Miniconda ****" && \
    bash miniconda.sh -f -b -p "$CONDA_DIR" && \
    \
    echo "**** setup Miniconda ****" && \
    conda update --all --yes && \
    conda config --set auto_update_conda False && \
    \
    echo "**** cleanup ****" && \
    rm -f miniconda.sh && \
    conda clean --all --force-pkgs-dirs --yes && \
    find "$CONDA_DIR" -follow -type f \( -iname '*.a' -o -iname '*.pyc' -o -iname '*.js.map' \) -delete && \
    \
    echo "**** finalize ****" && \
    mkdir -p "$CONDA_DIR/locks" && \
    chmod 777 "$CONDA_DIR/locks"


RUN conda init bash && \
    conda update -n base -c defaults conda && \
    conda create -n nssm python=3.9 && \
    conda install -n nssm -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia && \
    conda install -n nssm -y conda-forge::matplotlib conda-forge::trimesh conda-forge::scipy=1.10.1 conda-forge::termcolor conda-forge::igl conda-forge::prettytable conda-forge::tqdm conda-forge::scikit-learn xformers::xformers conda-forge::multiprocess conda-forge::tensorboard && \
    conda run -n nssm pip3 install pymeshlab==2023.12.post1 && \
    conda run -n nssm pip3 install mitsuba && \
    conda run -n nssm pip3 install timm && \
    conda run -n nssm pip3 install gdist && \
    conda clean --yes --all



RUN git clone https://github.com/KinglittleQ/torch-batch-svd.git && \
        cd torch-batch-svd && \
        conda run -n nssm pip install .

