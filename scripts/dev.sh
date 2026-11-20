#!/usr/bin/env bash
# Run a command inside the routerdev container with the repo mounted at /src.
#   scripts/dev.sh 'uv run pytest -q'
# Env: ROUTER_CPUS (docker --cpus), ROUTER_NET (default host), ROUTER_GPU=1 (--gpus all).
set -euo pipefail
cd "$(dirname "$0")/.."
image=routerdev:py3.12
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build -q -t "$image" -f docker/dev.Dockerfile docker >/dev/null
fi
mkdir -p "$HOME/.cache/routerdev"
args=(--rm -i -v "$PWD:/src" -v "$HOME/.cache/routerdev:/cache" -w /src
      --user "$(id -u):$(id -g)" -e HOME=/cache --network "${ROUTER_NET:-host}")
[ -n "${ROUTER_CPUS:-}" ] && args+=(--cpus "$ROUTER_CPUS")
[ "${ROUTER_GPU:-0}" = 1 ] && args+=(--gpus all)
exec docker run "${args[@]}" "$image" bash -lc "$*"
