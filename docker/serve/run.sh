#!/usr/bin/env bash
# Serve a merged Cosimo checkpoint locally over an OpenAI-compatible API, for manual testing.
#
# Usage, from anywhere:
#   bash docker/serve/run.sh                                  # serve runs/sft/merged on :8000
#   bash docker/serve/run.sh --run-name dpo                   # serve a different run
#   bash docker/serve/run.sh --port 8001                       # move it off the app's port
#   bash docker/serve/run.sh --model /abs/path/to/checkpoint   # anything on disk
#   bash docker/serve/run.sh -- --gpu-memory-utilization 0.7   # extra args go to vllm serve
#
#   nohup bash docker/serve/run.sh > serve.log 2>&1 &          # leave it running
#
# This is a MANUAL TESTING harness, not a deployment. It binds to 127.0.0.1, has no
# authentication, and serves whatever checkpoint you point it at.
#
# WHY A DIFFERENT IMAGE: the fine-tuning image (cosimo-fine-tune:latest) exists to train, and
# its NGC/unsloth stack is pinned for that. Serving is a different job with a different stack,
# so this uses vLLM's own published image rather than adding an inference server to the training
# one. Nothing is built here -- the image comes straight from Docker Hub.
#
# ARCHITECTURE: the DGX Spark is aarch64, so this needs the `-aarch64` tag. The default x86-64
# `vllm/vllm-openai:latest` manifest will not run here. Override with COSIMO_VLLM_IMAGE if you
# want to pin a version rather than track latest -- recommended once you find a tag that works,
# because `latest` moves under you.
#
# THE FLAGS THAT ARE NOT OPTIONAL:
#
#   --tool-call-parser hermes + --enable-auto-tool-choice
#       The model is trained to emit Hermes-format <tool_call> blocks
#       (jobs/fine-tune/cosimo_ft/tools.py). Without these, vLLM returns that block as plain
#       message content, LangGraph never sees a tool call, and the ReAct loop in
#       cosimo/agents/react_agent/agent.py terminates on its first step with raw JSON as the
#       answer. See "Tool calling and the LangGraph flow" in jobs/fine-tune/README.md.
#
#   --chat-template <the checkpoint's own chat_template.jinja>
#       08_export_merge.py writes the harness template into the checkpoint and verifies the
#       vendor "Your name is Phi, an AI math expert developed by Microsoft." preamble is gone.
#       Passing the file explicitly means a runtime that fails to pick up chat_template.jinja
#       from the model directory cannot silently fall back to a different prompt surface than
#       the one the model was trained and evaluated under.
#
#   --dtype bfloat16
#       Explicit rather than `auto`. Checkpoints exported before the torch_dtype fix in
#       08_export_merge.py carry `"torch_dtype": null` in config.json, which `auto` resolves to
#       float16 -- a much smaller exponent range than the bf16 these weights were trained in.
#
# CONTEXT LENGTH: --max-model-len defaults to 8192, matching model.max_seq_length in
# jobs/fine-tune/configs/base.yaml. The architecture claims 131072 (LongRoPE), but the LoRA was
# only ever trained at 8192 and adapts attention (qkv_proj, o_proj), so long-context behaviour is
# untested and unmeasured. Raise it deliberately, not by default.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS_DIR="${REPO_ROOT}/jobs/fine-tune"

IMAGE="${COSIMO_VLLM_IMAGE:-vllm/vllm-openai:latest-aarch64}"
RUN_NAME="sft"
MODEL=""
PORT="8000"
MAX_MODEL_LEN="8192"
SERVED_NAME="cosimo"
vllm_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)       RUN_NAME="$2"; shift 2 ;;
        --model)          MODEL="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
        --served-name)    SERVED_NAME="$2"; shift 2 ;;
        --)               shift; vllm_args+=("$@"); break ;;
        -h|--help)        sed -n '2,50p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)
            echo "unknown option: $1" >&2
            echo "run with --help for usage; pass vllm's own flags after --" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    MODEL="${HARNESS_DIR}/runs/${RUN_NAME}/merged"
fi

# Fail here rather than inside the container, where the error is a Python traceback about a
# missing repo id and reads like a Hugging Face problem.
if [[ ! -f "${MODEL}/config.json" ]]; then
    echo "no merged checkpoint at ${MODEL}" >&2
    echo "Export one first:" >&2
    echo "  bash docker/fine-tune/run.sh python scripts/08_export_merge.py --run-name ${RUN_NAME}" >&2
    exit 1
fi

# The adapter directory is the other thing that lives under runs/<name>/, and pointing a server
# at it produces a confusing failure much later. Catch the mistake by name.
if [[ -f "${MODEL}/adapter_config.json" ]]; then
    echo "${MODEL} is a LoRA adapter, not a merged checkpoint." >&2
    echo "Serve runs/${RUN_NAME}/merged, or export it with scripts/08_export_merge.py." >&2
    exit 1
fi

CHAT_TEMPLATE="${MODEL}/chat_template.jinja"
if [[ ! -f "${CHAT_TEMPLATE}" ]]; then
    echo "no chat_template.jinja in ${MODEL}." >&2
    echo "Re-export with scripts/08_export_merge.py so the served prompt surface matches the" >&2
    echo "one the model was trained and evaluated under." >&2
    exit 1
fi

echo "image        : ${IMAGE}"
echo "checkpoint   : ${MODEL}"
echo "served as    : ${SERVED_NAME}"
echo "endpoint     : http://127.0.0.1:${PORT}/v1"
echo "max-model-len: ${MAX_MODEL_LEN}"
echo

# --ipc=host and the ulimits are the standard NGC/vLLM flags, matching docker/fine-tune/run.sh:
# the server uses shared memory, and pinned-memory allocation on the 128 GB unified-memory GB10
# needs an unlimited memlock. The Hugging Face cache is mounted because vLLM still reads tokenizer
# assets through it. Port is published on 127.0.0.1 only: this server has no authentication.
docker run --rm -i \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${MODEL}:/model:ro" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -p "127.0.0.1:${PORT}:8000" \
    "${IMAGE}" \
    --model /model \
    --served-model-name "${SERVED_NAME}" \
    --chat-template "/model/chat_template.jinja" \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    "${vllm_args[@]}"
