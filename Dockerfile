FROM nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04

# Set environment variable to avoid interactive prompts
ARG DEBIAN_FRONTEND=noninteractive

# Install apt packages and clean up in a single layer
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        unzip psmisc wget bc jq htop curl git git-lfs vim nano ssh gcc tmux && \
    rm -rf /var/lib/apt/lists/* /var/cache/*

# Install uv and clean up
RUN wget -qO /uv-installer.sh https://astral.sh/uv/install.sh && \
    sh /uv-installer.sh && \
    rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Install Visual Studio Code CLI and clean up
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz && \
    tar -xf vscode_cli.tar.gz && \
    rm vscode_cli.tar.gz

# Copy project files
COPY .. /exact
WORKDIR /exact

# Install Python libraries
RUN uv sync --all-extras --frozen

CMD ["sleep", "infinity"]