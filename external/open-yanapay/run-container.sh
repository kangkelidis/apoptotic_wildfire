#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}/workspace"
FRAMES_DIR="${WORKSPACE_DIR}/results/frames"

docker rm -f evacuation-simulation >/dev/null 2>&1 || true

image="robot-assisted-evacuation"
args=()
for arg in "$@"; do
    if [ "$arg" == "hub" ]; then
        image="alexandroskangkelidis/robot-assisted-evacuation:v1.0"
    else
        args+=("$arg")
    fi
done

if [ "${#args[@]}" -gt 0 ]; then
    docker run --platform linux/amd64 --name evacuation-simulation -it \
        -v "${WORKSPACE_DIR}":/home/workspace \
        "$image" "${args[@]}"
else
    docker run --platform linux/amd64 --name evacuation-simulation -it \
        -v "${WORKSPACE_DIR}":/home/workspace \
        "$image"
fi

if [ -d "${FRAMES_DIR}" ]; then
    find "${FRAMES_DIR}" -type f -exec rm {} +
fi
