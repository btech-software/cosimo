#!/usr/bin/env bash
# Run a command inside the Cosimo fine-tuning container on the DGX Spark.
#
# Usage, from anywhere:
#   bash docker/fine-tune/run.sh                                   # interactive shell
#   bash docker/fine-tune/run.sh python scripts/00_check_env.py    # one-shot command
#   nohup bash docker/fine-tune/run.sh python scripts/04_train_sft.py > sft.log 2>&1 &
#
# The repository is bind-mounted at /workspace/cosimo and the working directory is the harness
# root (/workspace/cosimo/jobs/fine-tune), so every script can be invoked exactly as written in
# the README, edits on the host take effect immediately, and data/ + runs/ land on the host
# filesystem. The host Hugging Face cache is mounted at /workspace/.hf to match HF_HOME in the
# image, so model weights are downloaded once.
#
# --ipc=host and the memlock/stack ulimits are the standard NGC flags: dataloader workers use
# shared memory, and pinned-memory allocation on the 128 GB unified-memory GB10 needs an
# unlimited memlock. HF_TOKEN and WANDB_API_KEY are forwarded only when set on the host.
#
# TTY: -t is added only when stdin and stdout are both terminals. Training and evaluation are
# multi-hour jobs, so they are normally started under nohup, over `ssh host '...'`, or piped into
# a log; `docker run -t` fails outright in all three ("the input device is not a TTY").
#
# FILE OWNERSHIP: the NGC image runs as root and no --user is passed by default, so files the
# container creates under data/, runs/ and the mounted Hugging Face cache are owned by root on the
# host. To take them back:
#   sudo chown -R "$(id -u):$(id -g)" jobs/fine-tune/data jobs/fine-tune/runs "$HOME/.cache/huggingface"
# Alternatively, export COSIMO_RUN_AS_HOST_USER=1 to run the container as your own uid/gid (with
# HOME=/tmp, which NGC images need or pip and huggingface_hub try to write to /). That is opt-in
# and not the default: parts of the NGC stack expect to be root, and a mismatched uid inside the
# container makes some tooling behave oddly. Prefer the chown recipe unless it gets in your way.
#
# NGC BANNER: the base image's ENTRYPOINT prints a licence/driver banner on stdout before running
# the command, so `run.sh python scripts/07_compare.py > table.md` captures the banner too. None of
# the harness scripts need stdout redirection (07_compare.py writes runs/comparisons/*.md itself),
# so the entrypoint is left in place — it also runs NGC's /opt/nvidia/entrypoint.d/ setup. If you
# do need clean stdout, follow NVIDIA's playbook and bypass it for that one command:
#   docker run ... --entrypoint /usr/bin/env cosimo-fine-tune:latest python scripts/07_compare.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_CACHE="${HOME}/.cache/huggingface"
mkdir -p "${HF_CACHE}"

# -t only when both stdin and stdout are terminals; see the TTY note above.
tty_flags=(-i)
if [[ -t 0 && -t 1 ]]; then
    tty_flags+=(-t)
fi

docker_args=(
    --rm
    "${tty_flags[@]}"
    --gpus all
    --ipc=host
    --net=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${REPO_ROOT}:/workspace/cosimo"
    -v "${HF_CACHE}:/workspace/.hf"
    -w /workspace/cosimo/jobs/fine-tune
)

if [[ "${COSIMO_RUN_AS_HOST_USER:-0}" == "1" ]]; then
    docker_args+=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
fi

# Passed by name, not by value: `-e HF_TOKEN=<secret>` would put the token in this
# process's argv, where `ps` exposes it to every other user on the machine. With the
# bare name Docker reads the value from this shell's environment instead.
if [[ -n "${HF_TOKEN:-}" ]]; then
    docker_args+=(-e HF_TOKEN)
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    docker_args+=(-e WANDB_API_KEY)
fi

if [[ $# -eq 0 ]]; then
    set -- /bin/bash
fi

docker run "${docker_args[@]}" cosimo-fine-tune:latest "$@"
