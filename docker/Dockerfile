# Qwen3-TTS development image (GPU, CUDA 12.8, Python 3.12)
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
ARG TORCH_VERSION=2.8.0
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Taipei \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages: python, build toolchain, audio libs, dev / profiling tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential cmake ninja-build pkg-config \
        git git-lfs curl wget ca-certificates sudo locales tzdata \
        tmux vim less htop nvtop procps psmisc ripgrep jq tree unzip \
        sox libsox-dev libsox-fmt-all ffmpeg libsndfile1 \
        cuda-nsight-systems-12-8 cuda-nsight-compute-12-8 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user matching host uid/gid so bind-mounted files keep correct ownership.
# The base image ships a "ubuntu" user (uid 1000); remove it to free the ids.
RUN userdel -r ubuntu 2>/dev/null || true \
    && (getent group ${GID} || groupadd -g ${GID} ${USERNAME}) \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Python venv owned by the user (so pip installs inside the container need no root)
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/home/${USERNAME}/.local/bin:/usr/local/cuda/bin:${PATH}
RUN python3.12 -m venv /opt/venv && chown -R ${UID}:${GID} /opt/venv

USER ${USERNAME}
WORKDIR /home/${USERNAME}

RUN pip install --upgrade pip setuptools wheel \
    && pip install torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} \
        --index-url https://download.pytorch.org/whl/cu128 \
    && pip install "${FLASH_ATTN_WHEEL}"

# Install repo dependencies (pinned by pyproject.toml) as an editable package.
# The real source is bind-mounted over REPO_DIR at runtime; compat mode uses a
# plain .pth path entry, so the mounted code is what gets imported.
ARG REPO_DIR=/workspace
COPY --chown=${UID}:${GID} pyproject.toml README.md MANIFEST.in ${REPO_DIR}/
COPY --chown=${UID}:${GID} qwen_tts ${REPO_DIR}/qwen_tts
USER root
RUN chown ${UID}:${GID} $(dirname ${REPO_DIR})  2>/dev/null || true
USER ${USERNAME}
RUN pip install -e ${REPO_DIR} --config-settings editable_mode=compat \
    && pip install "huggingface_hub[cli]" modelscope ipython py-spy

# Claude Code (native installer -> ~/.local/bin/claude)
RUN curl -fsSL https://claude.ai/install.sh | bash

COPY --chown=${UID}:${GID} docker/tmux.conf /home/${USERNAME}/.tmux.conf
RUN mkdir -p /home/${USERNAME}/.cache/huggingface /home/${USERNAME}/.claude \
    && echo 'export PS1="(qwen3-tts) \u@\h:\w\$ "' >> /home/${USERNAME}/.bashrc

WORKDIR ${REPO_DIR}
CMD ["sleep", "infinity"]
