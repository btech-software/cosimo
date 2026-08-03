#!/usr/bin/env python3
"""Build-time guard: is the torch in this image a usable DGX Spark (GB10) torch?

Run from docker/fine-tune/Dockerfile at every stage that could disturb torch. It exits non-zero
with an explanatory message, so a bad image breaks the build instead of shipping.

Two things here are easy to get wrong, and both were:

* **Read the arch list without a driver.** ``torch.cuda.get_arch_list()`` returns ``[]`` when
  ``torch.cuda.is_available()`` is False, and there is no GPU inside ``docker build``. A guard
  written against the public wrapper can therefore never pass at build time. The private
  ``torch._C._cuda_getArchFlags()`` reads the arch list compiled into the binary and needs no
  driver, which is the whole point of a build-time guard.

* **GB10 is sm_121, but a torch does not need literal sm_121 cubins to run on it.** CUDA
  guarantees binary compatibility from one minor revision to the next within a major
  architecture, so sm_120 cubins run on an sm_121 device, and ``compute_120`` PTX is the JIT
  fallback. ``nvcr.io/nvidia/pytorch:25.11-py3`` ships exactly that -- ``sm_80 sm_86 sm_90
  sm_100 sm_110 sm_120 compute_120`` -- and runs bf16 matmuls on a GB10. Demanding the literal
  string ``sm_121`` rejects a working image.

Usage:
    python torch_arch_guard.py --stage base
    python torch_arch_guard.py --stage "dependency group" --expect-torch-version 2.10.0a0+...
"""

import argparse
import platform
import sys

import torch

# Anything in here can execute on an sm_121 device: its own cubins, the next-lower minor revision
# of the same major architecture, or PTX that the driver JITs.
GB10_RUNNABLE = ("sm_121", "sm_120", "compute_121", "compute_120")


def arch_flags():
    """The architectures this torch was compiled for. No CUDA driver required."""
    return (torch._C._cuda_getArchFlags() or "").split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, help="build stage name, used in messages")
    parser.add_argument(
        "--expect-torch-version",
        help="fail if torch.__version__ differs; catches a pip install swapping the NGC torch",
    )
    args = parser.parse_args()

    archs = arch_flags()
    machine = platform.machine()
    cuda = torch.version.cuda or ""
    problems = []

    if args.expect_torch_version and args.expect_torch_version != torch.__version__:
        problems.append(
            f"{args.stage} replaced the NGC torch {args.expect_torch_version} with "
            f"{torch.__version__}. The aarch64 CUDA 13 build for a GB10 is gone; do not ship "
            f"this image"
        )

    if not set(archs) & set(GB10_RUNNABLE):
        problems.append(
            f"torch {torch.__version__} carries no architecture a GB10 (sm_121) can run: arch "
            f"list {', '.join(archs) or 'empty'}, none of {', '.join(GB10_RUNNABLE)}. Every "
            f"kernel launch on a DGX Spark will fail"
        )

    # An amd64 manifest pulled through an emulating builder (docker buildx on an x86 host, or a
    # stale multi-arch tag) otherwise builds perfectly green and then dies on the Spark.
    if machine != "aarch64":
        problems.append(
            f"this image is {machine}, not aarch64; an amd64 manifest was pulled (emulating "
            f"builder or stale multi-arch tag)"
        )

    # A CUDA 12 torch would not match the CUDA_HOME the Dockerfile sets.
    if not cuda.startswith("13"):
        problems.append(f"torch reports CUDA {cuda or 'none'}, expected 13.x to match CUDA_HOME")

    if problems:
        sys.exit(
            f"FATAL: torch is not usable on a DGX Spark after {args.stage}:\n  - "
            + "\n  - ".join(problems)
        )

    print(
        f"torch guard ok after {args.stage}: {torch.__version__}, cuda {cuda}, {machine}, "
        f"arch list {', '.join(archs)}"
    )


if __name__ == "__main__":
    main()
