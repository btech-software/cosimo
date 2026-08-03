"""Cosimo fine-tuning harness.

Submodules are deliberately not imported here: ``modeling``, ``generation``,
``benchmarks`` and ``evalrun`` pull in torch/transformers/unsloth, while
``config``, ``chat``, ``data_schema``, ``grading``, ``splits`` and ``report``
must stay importable on a CPU-only laptop with stdlib + pyyaml. Import the
submodule you need explicitly.
"""

__version__ = "0.1.0"
