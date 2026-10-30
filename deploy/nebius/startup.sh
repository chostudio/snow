#!/usr/bin/env bash
# Startup and initialization script for Nebius GPU compute instances.
# Deploys NVIDIA Container Toolkit, runs the Body Pose NIM, and starts
# the Model serving backend.
set -euo pipefail

echo "==> Configuring Nebius GPU Instance for Snow"

# 1. System packages
apt-get update && apt-get install -y curl jq bc git htop

# 2. NVIDIA Container Toolkit verification
if ! command -v nvidia-ctk &>/dev/null; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

# 3. Spend & Idle Guards
cat > /usr/local/bin/snow-idle-stop <<'IDLE'
#!/bin/bash
STATE=/var/lib/snow-idle-count
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1 2>/dev/null || echo 0)
load_busy=$(awk '{print ($1 >= 1.0)}' /proc/loadavg)
typed=$(find /dev/pts -name '[0-9]*' -amin -30 2>/dev/null | wc -l)
if [ "${gpu_util:-0}" -lt 5 ] && [ "$load_busy" -eq 0 ] && [ "$typed" -eq 0 ]; then
  n=$(( $(cat $STATE 2>/dev/null || echo 0) + 1 ))
else
  n=0
fi
echo $n > $STATE
# 30 min idle -> shutdown
[ "$n" -ge 6 ] && { logger "snow-nebius-idle: idle 30 min, shutting down"; echo 0 > $STATE; poweroff; }
IDLE

chmod +x /usr/local/bin/snow-idle-stop
echo "*/5 * * * * root /usr/local/bin/snow-idle-stop" > /etc/cron.d/snow-nebius-guards

# 4. Launch 3D Body Pose NIM Container
if [ -n "${NVIDIA_API_KEY:-}" ]; then
    echo "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
    mkdir -p /opt/nim-cache && chown 1000:1000 /opt/nim-cache
    docker run -d --name body-pose --restart unless-stopped --gpus all --shm-size=8g \
      -e NGC_API_KEY="$NVIDIA_API_KEY" -v /opt/nim-cache:/opt/nim/.cache \
      -p 127.0.0.1:8000:8000 -p 127.0.0.1:8001:8001 \
      nvcr.io/nim/nvidia/body-pose:latest
fi

echo "==> Nebius setup complete"
