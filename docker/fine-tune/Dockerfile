# Fine-tuning image for jobs/fine-tune, targeting an NVIDIA DGX Spark
# (GB10 Grace Blackwell, aarch64, compute capability sm_121, 128 GB unified memory, CUDA 13.0).
#
# Build context is the repository root, matching docker/app/:
#   docker build -t cosimo-fine-tune:latest -f docker/fine-tune/Dockerfile .
# or simply: bash docker/fine-tune/build.sh
#
# This is the image Unsloth's official DGX Spark Dockerfile builds on. It already carries the
# aarch64 CUDA 13 torch build tuned for GB10; that torch must survive every step below untouched,
# which is what the torch guard further down enforces.
#
# Triton and xformers are deliberately NOT rebuilt from source. Unsloth's official
# Dockerfile_DGX_Spark still compiles both on this same base "for latest blackwell support", but
# NVIDIA's own DGX Spark fine-tuning playbook uses the Triton and attention kernels shipped in
# nvcr.io/nvidia/pytorch:25.11-py3 unchanged, and those source builds cost roughly an hour each
# and pull a second toolchain into the image. We follow the playbook; scripts/00_check_env.py
# compiles and launches a real @triton.jit kernel on the live device and fails hard if that
# assumption does not hold, which is a cheaper and more honest test than building blind.
FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV CUDA_HOME=/usr/local/cuda-13.0
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV C_INCLUDE_PATH=${CUDA_HOME}/include:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH}

# The [dependency-groups] fine-tune group in pyproject.toml is the single source of truth for the
# pinned stack. It is extracted here with stdlib tomllib and never restated in this Dockerfile.
COPY pyproject.toml /tmp/pyproject.toml

# The guard every stage below runs. It lives in a file rather than being inlined three times,
# because two of the three inlined copies of it were wrong in the same two ways; see the module
# docstring for what those were. It needs no GPU: `docker build` has no CUDA driver.
COPY docker/fine-tune/torch_arch_guard.py /usr/local/bin/torch_arch_guard.py

# Base-image guard, run before anything is installed. The version-comparing guards below only
# prove that the torch we ended with is the torch we started with; they say nothing about whether
# that torch can run on a GB10 at all. This one does.
RUN set -eux \
 && python /usr/local/bin/torch_arch_guard.py --stage "base image"

# Plain pip, deliberately NOT `uv pip install --group`: pip accepts the pre-installed NGC
# pre-release torch as already satisfying the transitive `torch>=...` requirements of
# transformers/trl/peft/accelerate, whereas a fresh uv resolution downloads a PyPI torch and
# destroys the CUDA 13 aarch64 build.
#
# torch guard: the NGC torch version is recorded before the install and compared after. A silently
# swapped torch is the single most damaging failure mode for this image and must break the build.
# The arch list is re-checked with it, so a same-version-different-wheel swap is caught too.
#
# bitsandbytes is pinned to 0.49.2 in the group, NOT to the 0.48.0 used by Unsloth's own DGX Spark
# Dockerfile: unsloth's metadata excludes exactly 0.46.0 and 0.48.0, and `uv lock` proves 0.48.0
# and unsloth cannot co-exist. Do not "fix" this back to 0.48.0. 0.49.2's aarch64 wheel does ship
# sm_121 cubins (plus sm_121 PTX) in libbitsandbytes_cuda130.so, the library selected when
# torch.version.cuda == "13.0", so 4-bit is expected to work; scripts/00_check_env.py confirms it
# on the live device. bf16 LoRA stays the default regardless.
RUN set -eux \
 && python -c 'import pathlib, torch; pathlib.Path("/tmp/torch-version-before").write_text(torch.__version__)' \
 && python -c 'import pathlib, sys, tomllib; groups = tomllib.loads(pathlib.Path("/tmp/pyproject.toml").read_text()).get("dependency-groups", {}); reqs = groups.get("fine-tune") or sys.exit("FATAL: pyproject.toml has no [dependency-groups] fine-tune group"); pathlib.Path("/tmp/fine-tune-requirements.txt").write_text("\n".join(reqs) + "\n")' \
 && cat /tmp/fine-tune-requirements.txt \
 && pip install --no-cache-dir -r /tmp/fine-tune-requirements.txt \
 && python /usr/local/bin/torch_arch_guard.py --stage "the dependency group" \
      --expect-torch-version "$(cat /tmp/torch-version-before)"

# unsloth and unsloth_zoo are installed with --no-deps and are intentionally NOT part of the locked
# dependency group: their declared dependencies (torch, bitsandbytes, transformers bounds) conflict
# with the NGC-tuned stack and cannot be co-resolved by uv. --no-deps is what protects the NGC torch
# and the pins installed above; the group already provides everything unsloth actually needs.
# The torch guard is repeated so a bad unsloth release cannot slip a torch past it either.
#
# pip freeze is written into the image afterwards so an operator can pin or diff a build:
#   docker run --rm cosimo-fine-tune:latest cat /opt/cosimo/pip-freeze.txt > freeze-$(date +%F).txt
# It is the resolved transitive closure, which pyproject.toml's direct pins do not record.
RUN set -eux \
 && pip install --no-deps --no-cache-dir unsloth==2026.8.1 unsloth_zoo==2026.8.1 \
 && python /usr/local/bin/torch_arch_guard.py --stage "the unsloth install" \
      --expect-torch-version "$(cat /tmp/torch-version-before)" \
 && mkdir -p /opt/cosimo \
 && pip freeze > /opt/cosimo/pip-freeze.txt \
 && wc -l /opt/cosimo/pip-freeze.txt

ENV HF_HOME=/workspace/.hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

# Nothing else is copied: the repository is bind-mounted at run time by docker/fine-tune/run.sh.
# Nothing is trained and no model is downloaded at build time.
WORKDIR /workspace/cosimo

CMD ["/bin/bash"]
