#!/usr/bin/env bash
# Build the Cosimo fine-tuning image (cosimo-fine-tune:latest) for the DGX Spark.
#
# Usage, from anywhere:
#   bash docker/fine-tune/build.sh              # normal build
#   bash docker/fine-tune/build.sh --no-cache   # extra args are passed to docker build
#
# The build context is the repository root (matching docker/app/), because the Dockerfile needs
# pyproject.toml, which holds the single source of truth for the pinned fine-tune stack.
# Expect a long first build: the NGC base image is several GB. Nothing is trained and no model is
# downloaded at build time; no GPU is required to build.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "${REPO_ROOT}"
docker build -t cosimo-fine-tune:latest -f docker/fine-tune/Dockerfile "$@" .

echo "Built cosimo-fine-tune:latest"
echo "Next: bash docker/fine-tune/run.sh python scripts/00_check_env.py"
