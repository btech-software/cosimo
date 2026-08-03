#!/usr/bin/env python3
"""Verify the fine-tuning environment before anything expensive is run.

Checks the interpreter, the CUDA device (a DGX Spark GB10 reports compute capability (12, 1) and
the torch build must carry an arch a GB10 can run), that a real Triton kernel compiles
and launches, the installed versions against the pinned ``fine-tune`` dependency group in the
repository ``pyproject.toml``, whether ``unsloth`` imports, whether a bitsandbytes 4-bit linear
actually runs on this GPU, free disk space on both the harness and the ``HF_HOME`` filesystem, and
whether a Hugging Face token is reachable.

Only stdlib is imported at module level and every other import is guarded, so this script still
produces a useful report on a broken or CPU-only environment instead of raising. It writes
``runs/env_check.json``, prints a summary table, and exits 1 on a hard failure: no CUDA, a CUDA
device query that raises, a torch with no GB10-runnable arch, a Triton kernel that will not run,
``import unsloth`` failing, a version mismatch on transformers / trl / unsloth, huggingface-hub
1.x, or a report file that cannot be written.

Example:
    python scripts/00_check_env.py
"""

import argparse
import functools
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # python < 3.11; reported as a python-version failure below
    tomllib = None

HARNESS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HARNESS_ROOT.parents[1]

MIN_PYTHON = (3, 12)  # pyproject.toml requires-python
EXPECTED_CAPABILITY = (12, 1)  # GB10 Grace Blackwell, sm_121

# A GB10 is sm_121, but a torch does not need literal sm_121 cubins to run on it: CUDA guarantees
# binary compatibility from one minor revision to the next within a major architecture, so sm_120
# cubins execute on an sm_121 device, and compute_120 PTX is the JIT fallback. The NGC aarch64
# image ships sm_80/86/90/100/110/120 + compute_120 and no literal sm_121, and runs bf16 matmuls
# on a Spark. At least one of these must appear in torch.cuda.get_arch_list().
GB10_RUNNABLE_ARCHS = ("sm_121", "sm_120", "compute_121", "compute_120")
DEPENDENCY_GROUP = "fine-tune"

# unsloth and unsloth_zoo are installed with --no-deps in docker/fine-tune/Dockerfile and are
# deliberately outside the locked dependency group, so their expected version cannot be read from
# pyproject.toml. Keep this in sync with that Dockerfile; it is the only version stated in code.
UNSLOTH_EXPECTED = "2026.8.1"

# A version mismatch on these is a hard failure: they define the training API surface the harness
# was written against (TRL 0.24.0 SFTConfig/DPOConfig fields, unsloth's supported ranges).
CRITICAL_PACKAGES = ("transformers", "trl", "unsloth")

# huggingface-hub 1.x dropped APIs that transformers 4.56.2 still calls, so a 1.x hub breaks the
# whole stack at import time. The dependency group carries the same bound; this is the explicit
# cross-check, because a stray `pip install -U huggingface_hub` is a common way to lose an image.
HF_HUB_MAX_EXCLUSIVE = "1.0"

MIN_FREE_DISK_GB = (
    50.0  # base weights + tokenized data + adapters + checkpoints + merged export
)
GB = 1024**3

REQUIREMENT_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*"
    r"(?:\[(?P<extras>[^\]]*)\])?\s*"
    r"(?P<spec>[^;]*?)\s*"
    r"(?:;\s*(?P<marker>.*))?\s*$"
)
SPECIFIER_RE = re.compile(r"(==|!=|>=|<=|~=|>|<)\s*([^,\s]+)")
PLAIN_VERSION_RE = re.compile(r"^\d+(?:\.\d+)*$")


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def load_packaging() -> tuple[object | None, object | None]:
    """Return ``(SpecifierSet, Marker)`` from ``packaging`` if importable, else ``(None, None)``.

    ``packaging`` is guaranteed present in the container image (pip depends on it), but this
    script must also run on a bare stdlib-only interpreter, hence the guarded lazy import and the
    numeric fallbacks below.
    """
    try:
        from packaging.markers import Marker
        from packaging.specifiers import SpecifierSet
    except Exception:
        return None, None
    return SpecifierSet, Marker


@functools.lru_cache(maxsize=1)
def load_requirement_class() -> object | None:
    """Return ``packaging.requirements.Requirement`` if importable, else None."""
    try:
        from packaging.requirements import Requirement
    except Exception:
        return None
    return Requirement


def version_tuple(text: str) -> tuple[int, ...]:
    """Best-effort numeric version tuple; trailing non-numeric parts are dropped."""
    parts: list[int] = []
    for chunk in re.split(r"[._-]", text.strip()):
        match = re.match(r"^(\d+)", chunk)
        if match is None:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


def compare_versions(left: str, right: str) -> int:
    """Return -1/0/1 comparing two versions by their numeric components.

    Numeric only: ``4.57.0.dev0`` and ``4.57.0`` compare equal here. Use :func:`version_equal`
    whenever an exact pin is being enforced.
    """
    a, b = version_tuple(left), version_tuple(right)
    width = max(len(a), len(b))
    a += (0,) * (width - len(a))
    b += (0,) * (width - len(b))
    return (a > b) - (a < b)


def version_equal(installed: str, wanted: str) -> bool:
    """Exact-pin equality: a pre-release, post-release or dev build never satisfies ``==``."""
    installed, wanted = installed.strip(), wanted.strip()
    if installed == wanted:
        return True
    if not PLAIN_VERSION_RE.match(installed):
        # 4.57.0.dev0, 0.49.2rc1, 2026.8.1+local — deliberately not the pinned release.
        return False
    return compare_versions(installed, wanted) == 0


def marker_applies(marker: str | None) -> tuple[bool, str | None]:
    """Evaluate an environment marker.

    Returns ``(applies, note)``. ``note`` is non-None whenever the marker could not be evaluated
    properly, in which case the requirement is assumed to apply — silently dropping a pinned
    package because its marker was not understood is exactly the failure this reports.
    """
    if not marker:
        return True, None
    _, Marker = load_packaging()
    if Marker is not None:
        try:
            return bool(Marker(marker).evaluate()), None
        except Exception as exc:
            return (
                True,
                f"marker {marker!r} could not be evaluated ({type(exc).__name__}: {exc})",
            )
    match = re.search(r"sys_platform\s*(==|!=)\s*['\"]([^'\"]+)['\"]", marker)
    if match is None:
        return True, f"marker {marker!r} was not understood (packaging is unavailable)"
    op, value = match.group(1), match.group(2)
    return (sys.platform == value if op == "==" else sys.platform != value), None


def satisfies(installed: str, specifiers: list[tuple[str, str]]) -> bool:
    """Check an installed version against a list of ``(operator, version)`` pairs.

    Uses ``packaging.specifiers.SpecifierSet`` when available, which correctly refuses to let a
    pre-release or dev build satisfy an exact pin (``0.49.2rc1`` does not satisfy ``==0.49.2``).
    The numeric fallback keeps the same property for ``==`` via :func:`version_equal`.
    """
    if not specifiers:
        return True
    SpecifierSet, _ = load_packaging()
    if SpecifierSet is not None:
        try:
            return SpecifierSet(
                ",".join(f"{op}{ver}" for op, ver in specifiers)
            ).contains(installed)
        except Exception:
            pass
    for op, wanted in specifiers:
        order = compare_versions(installed, wanted)
        if op == "==" and not version_equal(installed, wanted):
            return False
        if op == "!=" and version_equal(installed, wanted):
            return False
        if op == ">=" and order < 0:
            return False
        if op == "<=" and order > 0:
            return False
        if op == ">" and order <= 0:
            return False
        if op == "<" and order >= 0:
            return False
        if op == "~=" and (
            order < 0 or version_tuple(installed)[:1] != version_tuple(wanted)[:1]
        ):
            return False
    return True


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def cuda_usable(cuda: dict) -> bool:
    """A device is only usable when it reports available AND every device query succeeded.

    ``is_available()`` returning True while ``get_device_name()`` raises
    ``no kernel image is available for execution on the device`` is the characteristic GB10 /
    sm_121 failure; it must never be reported as a working environment.
    """
    return bool(cuda["available"]) and not cuda["error"]


# --------------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------------


def check_python() -> dict:
    info = {
        "version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "system": platform.system(),
        "required": ".".join(str(p) for p in MIN_PYTHON),
        "ok": sys.version_info[:2] >= MIN_PYTHON,
    }
    return info


def check_cuda() -> dict:
    info = {
        "torch_installed": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "arch_list": None,
        "arch_ok": False,
        "arch_error": None,
        "available": False,
        "device_count": 0,
        "device_name": None,
        "capability": None,
        "capability_ok": False,
        "total_memory_gb": None,
        "free_memory_gb": None,
        "bf16_supported": None,
        "error": None,
    }
    try:
        import torch
    except Exception as exc:
        info["error"] = f"import torch failed: {exc}"
        return info

    info["torch_installed"] = True
    info["torch_version"] = getattr(torch, "__version__", None)
    info["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)

    # The compiled arch list is baked into the binary and needs no GPU, so it catches a wrong
    # base image (or an amd64 manifest pulled through an emulating builder) before any kernel
    # launch does. Kept separate from info["error"], which drives the device-query verdict.
    try:
        info["arch_list"] = [str(arch) for arch in torch.cuda.get_arch_list()]
        info["arch_ok"] = bool(set(info["arch_list"]) & set(GB10_RUNNABLE_ARCHS))
    except Exception as exc:
        info["arch_error"] = f"torch.cuda.get_arch_list() raised: {exc}"

    try:
        info["available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["error"] = f"torch.cuda.is_available() raised: {exc}"
        return info
    if not info["available"]:
        info["error"] = "torch.cuda.is_available() returned False"
        return info

    try:
        info["device_count"] = int(torch.cuda.device_count())
        info["device_name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["capability"] = [int(major), int(minor)]
        info["capability_ok"] = tuple(info["capability"]) == EXPECTED_CAPABILITY
        free, total = torch.cuda.mem_get_info(0)
        info["free_memory_gb"] = round(free / GB, 2)
        info["total_memory_gb"] = round(total / GB, 2)
        info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as exc:
        info["error"] = f"device query failed: {exc}"
    return info


def check_triton(cuda_ok: bool) -> dict:
    """Compile and launch a real Triton kernel.

    Unsloth *is* Triton kernels: importing triton proves nothing, because it JIT-compiles on the
    first launch, which on a training run is hours in. This actually compiles for the live device
    and synchronises, so an unsupported architecture surfaces here instead of mid-epoch.

    Note for maintainers: Unsloth's official ``Dockerfile_DGX_Spark`` builds Triton from source
    "for latest blackwell support" while NVIDIA's DGX Spark playbook uses the NGC-shipped Triton
    unchanged. docker/fine-tune/Dockerfile follows the playbook; this check is what proves that
    choice on the actual hardware.
    """
    info = {
        "version": installed_version("triton"),
        "status": "skipped",
        "usable": False,
        "detail": None,
    }
    if not cuda_ok:
        info["detail"] = (
            "no usable CUDA device; a Triton kernel cannot be compiled or launched"
        )
        return info
    try:
        import torch
        import triton
        import triton.language as tl

        info["version"] = getattr(triton, "__version__", None) or info["version"]

        @triton.jit
        def _add_one(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            tl.store(
                y_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + 1.0, mask=mask
            )

        n_elements = 128
        x = torch.ones(n_elements, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        _add_one[(1,)](x, y, n_elements, BLOCK_SIZE=128)
        torch.cuda.synchronize()
        correct = bool(torch.all(y == 2.0).item())
        info["usable"] = correct
        info["status"] = "ok" if correct else "failed"
        info["detail"] = (
            "a @triton.jit kernel compiled and ran on this GPU"
            if correct
            else "the Triton kernel launched but produced wrong values"
        )
    except (Exception, SystemExit) as exc:
        # SystemExit is caught for the same reason as in check_unsloth: a hard abort during
        # compilation must be reported, not propagated as a traceback.
        info["status"] = "failed"
        info["detail"] = f"{type(exc).__name__}: {exc}"
    return info


def read_pinned_requirements() -> tuple[list[dict], str | None, list[str]]:
    """Parse the [dependency-groups] fine-tune pins from the repository pyproject.toml.

    The group is the single source of truth for the stack; the versions are never restated here.
    Returns ``(requirements, error, skipped)``; ``skipped`` holds entries that are not plain
    requirement strings (``{include-group = "dev"}``) or that could not be parsed, so they are
    reported instead of silently vanishing from the table.
    """
    path = REPO_ROOT / "pyproject.toml"
    if tomllib is None:
        return [], "tomllib is unavailable (python < 3.11)", []
    if not path.is_file():
        return [], f"pyproject.toml not found at {path}", []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], f"failed to parse {path}: {exc}", []

    raw = data.get("dependency-groups", {}).get(DEPENDENCY_GROUP)
    if not raw:
        return [], f"{path} has no [dependency-groups] {DEPENDENCY_GROUP} group", []

    requirements: list[dict] = []
    skipped: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            skipped.append(
                f"{entry!r} is not a requirement string (nested group?); not checked"
            )
            continue
        parsed = parse_requirement(entry)
        if parsed is None:
            skipped.append(
                f"{entry!r} could not be parsed as a requirement; not checked"
            )
            continue
        requirements.append(parsed)
    return requirements, None, skipped


def parse_requirement(entry: str) -> dict | None:
    """Parse one requirement string, or return None if it is not a valid requirement.

    ``packaging.requirements.Requirement`` is authoritative when available. The regex fallback
    additionally verifies that the whole version-specifier part was consumed, so that garbage
    (``"this is not >>> a requirement"``) is reported as unparsable rather than silently turned
    into a check for a package called ``this``.
    """
    Requirement = load_requirement_class()
    if Requirement is not None:
        try:
            requirement = Requirement(entry)
        except Exception:
            return None
        return {
            "requirement": entry,
            "name": requirement.name,
            "specifiers": [
                (spec.operator, spec.version) for spec in requirement.specifier
            ],
            "marker": str(requirement.marker) if requirement.marker else None,
        }

    match = REQUIREMENT_RE.match(entry)
    if match is None:
        return None
    spec = (match.group("spec") or "").strip()
    specifiers = SPECIFIER_RE.findall(spec)
    if re.sub(r"\s+", "", spec) != ",".join(f"{op}{ver}" for op, ver in specifiers):
        return None
    return {
        "requirement": entry,
        "name": match.group("name"),
        "specifiers": specifiers,
        "marker": (match.group("marker") or "").strip() or None,
    }


def check_packages(requirements: list[dict]) -> list[dict]:
    rows = []
    for req in requirements:
        name = req["name"]
        pin = "".join(f"{op}{ver}" for op, ver in req["specifiers"]) or "any"
        found = installed_version(name)
        applies, marker_note = marker_applies(req["marker"])
        if not applies:
            status = "skipped"
        elif found is None:
            status = "missing"
        elif satisfies(found, req["specifiers"]):
            status = "ok"
        else:
            status = "mismatch"
        rows.append(
            {
                "name": name,
                "pin": pin,
                "marker": req["marker"],
                "marker_note": marker_note,
                "installed": found,
                "status": status,
                "critical": name in CRITICAL_PACKAGES,
            }
        )
    return rows


def check_hf_hub() -> dict:
    """huggingface-hub must stay below 1.0 for transformers 4.56.2."""
    version = installed_version("huggingface-hub")
    return {
        "version": version,
        "max_exclusive": HF_HUB_MAX_EXCLUSIVE,
        "ok": bool(version) and satisfies(version, [("<", HF_HUB_MAX_EXCLUSIVE)]),
    }


def check_unsloth() -> dict:
    info = {
        "importable": False,
        "version": installed_version("unsloth"),
        "zoo_version": installed_version("unsloth_zoo"),
        "expected": UNSLOTH_EXPECTED,
        "version_ok": False,
        "error": None,
    }
    try:
        import unsloth
    except (Exception, SystemExit) as exc:
        # SystemExit is included on purpose: unsloth aborts hard at import time on an
        # unsupported device, and this check must report that rather than crash.
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info

    info["importable"] = True
    info["version"] = getattr(unsloth, "__version__", None) or info["version"]
    info["version_ok"] = bool(info["version"]) and version_equal(
        info["version"], UNSLOTH_EXPECTED
    )
    return info


def check_bnb_4bit(cuda_ok: bool) -> dict:
    """Tiny bitsandbytes 4-bit forward pass; decides whether load_in_4bit is usable.

    The pinned bitsandbytes 0.49.2 aarch64 wheel does ship sm_121 cubins (plus sm_121 PTX) in
    ``libbitsandbytes_cuda130.so``, the library selected when ``torch.version.cuda == "13.0"``,
    so 4-bit is expected to work on a GB10. This smoke test is what confirms it on the actual
    device. It stays informational and never fatal: the default path is bf16 LoRA and 4-bit
    QLoRA is only a config toggle.
    """
    info = {
        "version": installed_version("bitsandbytes"),
        "status": "skipped",
        "usable": False,
        "detail": None,
    }
    if not cuda_ok:
        info["detail"] = "no usable CUDA device; 4-bit cannot be tested"
        return info
    try:
        import torch
        import bitsandbytes as bnb

        info["version"] = getattr(bnb, "__version__", None) or info["version"]
        layer = bnb.nn.Linear4bit(
            64,
            64,
            bias=False,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
        ).to("cuda")
        with torch.no_grad():
            out = layer(torch.randn(2, 64, device="cuda", dtype=torch.bfloat16))
        finite = bool(torch.isfinite(out).all().item())
        info["usable"] = finite
        info["status"] = "ok" if finite else "failed"
        info["detail"] = (
            "4-bit nf4 linear ran on this GPU"
            if finite
            else "4-bit nf4 linear produced non-finite values"
        )
    except Exception as exc:
        info["status"] = "failed"
        info["detail"] = f"{type(exc).__name__}: {exc}"
    return info


def measure_filesystem(path: Path) -> dict:
    """Measure the filesystem holding ``path``, walking up to its nearest existing parent."""
    entry: dict = {
        "labels": [],
        "path": str(path),
        "measured_path": None,
        "device": None,
        "total_gb": None,
        "free_gb": None,
        "ok": False,
        "error": None,
    }
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        entry["measured_path"] = str(probe)
        entry["device"] = int(os.stat(probe).st_dev)
        entry["total_gb"] = round(usage.total / GB, 2)
        entry["free_gb"] = round(usage.free / GB, 2)
        entry["ok"] = entry["free_gb"] >= MIN_FREE_DISK_GB
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def check_disk() -> dict:
    """Measure both filesystems that fill up.

    Checkpoints and the merged export land under the harness root, but base weights and the
    benchmark datasets land in HF_HOME, which docker/fine-tune/run.sh maps to a different host
    path that may well be a different filesystem. Entries sharing a device are merged.
    """
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    filesystems: list[dict] = []
    for label, path in (("harness", HARNESS_ROOT), ("hf_home", Path(hf_home))):
        entry = measure_filesystem(path)
        shared = next(
            (
                known
                for known in filesystems
                if entry["device"] is not None and known["device"] == entry["device"]
            ),
            None,
        )
        if shared is not None:
            shared["labels"].append(label)
            continue
        entry["labels"] = [label]
        filesystems.append(entry)
    return {
        "required_gb": MIN_FREE_DISK_GB,
        "filesystems": filesystems,
        "ok": all(entry["ok"] for entry in filesystems),
    }


def check_hf_token() -> dict:
    """Report a reachable token, from the environment or from a stored login.

    ``huggingface-cli login`` writes ``$HF_HOME/token`` (and run.sh mounts the host cache), so an
    operator who logged in on the host must not be told that gated repositories are unreachable.
    """
    hf_home = os.environ.get("HF_HOME")
    source = None
    for variable in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(variable):
            source = variable
            break
    variable = source
    if source is None:
        candidates = [Path(hf_home) / "token"] if hf_home else []
        candidates.append(Path.home() / ".cache" / "huggingface" / "token")
        for candidate in candidates:
            try:
                if (
                    candidate.is_file()
                    and candidate.read_text(encoding="utf-8").strip()
                ):
                    source = str(candidate)
                    break
            except Exception:
                continue
    return {
        "present": source is not None,
        "source": source,
        "variable": variable,  # None when no environment variable carries a token
        "hf_home": hf_home,
    }


# --------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------


def build_report() -> dict:
    requirements, pins_error, pins_skipped = read_pinned_requirements()
    cuda = check_cuda()
    cuda_ok = cuda_usable(cuda)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "harness_root": str(HARNESS_ROOT),
        "repo_root": str(REPO_ROOT),
        "python": check_python(),
        "cuda": cuda,
        "triton": check_triton(cuda_ok),
        "pins_error": pins_error,
        "pins_skipped": pins_skipped,
        "packages": check_packages(requirements),
        "hf_hub": check_hf_hub(),
        "unsloth": check_unsloth(),
        "bnb_4bit": check_bnb_4bit(cuda_ok),
        "disk": check_disk(),
        "hf_token": check_hf_token(),
    }
    report["failures"], report["warnings"] = evaluate(report)
    report["ok"] = not report["failures"]
    return report


def evaluate(report: dict) -> tuple[list[str], list[str]]:
    """Split the findings into hard failures (exit 1) and warnings (exit 0)."""
    failures: list[str] = []
    warnings: list[str] = []

    if not report["python"]["ok"]:
        failures.append(
            f"python {report['python']['version']} is older than the required "
            f"{report['python']['required']}"
        )

    cuda = report["cuda"]
    if not cuda["available"]:
        failures.append(f"no usable CUDA device ({cuda['error']})")
    elif cuda["error"]:
        # is_available() said yes but a device query threw. On a GB10 this is typically
        # "no kernel image is available for execution on the device": a torch build without
        # sm_121 kernels. Reporting this as a working environment is the worst possible outcome.
        failures.append(
            f"the CUDA device reports as available but the device query failed ({cuda['error']}); "
            "the device is not usable"
        )
    else:
        if not cuda["capability_ok"]:
            warnings.append(
                f"compute capability {cuda['capability']} is not the expected "
                f"{list(EXPECTED_CAPABILITY)} of a DGX Spark GB10 (sm_121): kernels, "
                "bitsandbytes support and the tuned defaults in configs/ were chosen for that "
                "device, so treat the pinned stack and the memory budget as unverified here"
            )
        if cuda["bf16_supported"] is False:
            warnings.append(
                "bfloat16 is not supported by this device; the bf16 LoRA default will fail"
            )
        if cuda["total_memory_gb"] and cuda["total_memory_gb"] < 100:
            warnings.append(
                f"only {cuda['total_memory_gb']} GB of device memory; the DGX Spark defaults "
                "assume 128 GB of unified memory"
            )

    if cuda["torch_installed"] and not cuda["arch_ok"]:
        failures.append(
            f"this torch ({cuda['torch_version']}, cuda {cuda['torch_cuda_version']}, "
            f"{report['python']['machine']}) carries no architecture a GB10 can run "
            f"({', '.join(GB10_RUNNABLE_ARCHS)}): arch list "
            f"{cuda['arch_list'] if cuda['arch_list'] is not None else cuda['arch_error']}. "
            "Every kernel launch on a GB10 will fail; the NGC aarch64 CUDA 13 torch has been "
            "replaced or the wrong base image was used"
        )

    triton = report["triton"]
    if cuda_usable(cuda) and not triton["usable"]:
        # Hard, not advisory: Unsloth's fast paths are Triton kernels, so a Triton that cannot
        # compile for this device means no training at all.
        failures.append(
            f"a Triton kernel could not be compiled and launched ({triton['detail']}); "
            "unsloth's kernels cannot run on this device"
        )

    if report["pins_error"]:
        failures.append(f"cannot read the pinned versions: {report['pins_error']}")
    for entry in report["pins_skipped"]:
        warnings.append(f"pinned requirement skipped: {entry}")
    for row in report["packages"]:
        if row["marker_note"]:
            warnings.append(
                f"{row['name']}: {row['marker_note']}; the requirement is assumed to apply"
            )
        if row["status"] in ("missing", "mismatch"):
            message = (
                f"{row['name']} is not installed ({row['pin']} required)"
                if row["installed"] is None
                else f"{row['name']} {row['installed']} is installed, {row['pin']} required"
            )
            (failures if row["critical"] else warnings).append(message)

    hub = report["hf_hub"]
    if hub["version"] is None:
        failures.append(
            f"huggingface-hub is not installed; transformers needs it and requires "
            f"< {hub['max_exclusive']}"
        )
    elif not hub["ok"]:
        failures.append(
            f"huggingface-hub {hub['version']} is installed but transformers 4.56.2 requires "
            f"< {hub['max_exclusive']}; 1.x removed APIs it still calls and breaks the stack at "
            "import time"
        )

    unsloth = report["unsloth"]
    if not unsloth["importable"]:
        failures.append(f"import unsloth failed: {unsloth['error']}")
    elif not unsloth["version_ok"]:
        failures.append(
            f"unsloth {unsloth['version'] or 'of an unknown version'} is installed, "
            f"{UNSLOTH_EXPECTED} required"
        )
    elif unsloth["zoo_version"] and not version_equal(
        unsloth["zoo_version"], UNSLOTH_EXPECTED
    ):
        warnings.append(
            f"unsloth_zoo {UNSLOTH_EXPECTED} expected, {unsloth['zoo_version']} found"
        )

    bnb = report["bnb_4bit"]
    if not bnb["usable"]:
        warnings.append(
            f"bitsandbytes 4-bit is not usable ({bnb['detail']}): keep model.load_in_4bit false "
            "and do not use adamw_8bit"
        )

    for entry in report["disk"]["filesystems"]:
        where = "+".join(entry["labels"])
        if entry["free_gb"] is None:
            warnings.append(
                f"could not measure free disk space for {where} ({entry['path']}): {entry['error']}"
            )
        elif not entry["ok"]:
            warnings.append(
                f"only {entry['free_gb']} GB free on the {where} filesystem "
                f"({entry['measured_path']}); at least {MIN_FREE_DISK_GB} GB is recommended for "
                "weights, checkpoints and merged exports"
            )

    if not report["hf_token"]["present"]:
        warnings.append(
            "no Hugging Face token found in HF_TOKEN, HUGGING_FACE_HUB_TOKEN or a stored login; "
            "gated or private repositories will not be reachable"
        )

    return failures, warnings


def render_table(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        lines.append(
            "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        )
    return "\n".join(lines)


def print_report(report: dict) -> None:
    cuda = report["cuda"]
    disk = report["disk"]
    unsloth = report["unsloth"]
    hub = report["hf_hub"]
    device_ok = cuda_usable(cuda)

    def state(ok: bool, warn_only: bool = False) -> str:
        if ok:
            return "ok"
        return "WARN" if warn_only else "FAIL"

    checks = [("CHECK", "STATUS", "DETAIL")]
    checks.append(
        (
            "python",
            state(report["python"]["ok"]),
            f"{report['python']['version']} ({report['python']['implementation']}, "
            f"{report['python']['machine']}) requires >= {report['python']['required']}",
        )
    )
    checks.append(
        (
            "torch",
            state(cuda["torch_installed"], warn_only=True),
            f"{cuda['torch_version'] or 'not installed'} (cuda {cuda['torch_cuda_version'] or 'n/a'})",
        )
    )
    checks.append(
        (
            "torch arch list",
            state(bool(cuda["arch_ok"])) if cuda["torch_installed"] else "WARN",
            f"GB10-runnable arch {'present' if cuda['arch_ok'] else 'MISSING'} in "
            f"{cuda['arch_list'] if cuda['arch_list'] is not None else cuda['arch_error']}"
            if cuda["torch_installed"]
            else "torch not installed",
        )
    )
    checks.append(
        (
            "cuda device",
            state(device_ok),
            cuda["error"] if cuda["error"] else (cuda["device_name"] or "unavailable"),
        )
    )
    checks.append(
        (
            "capability",
            state(bool(cuda["capability_ok"]), warn_only=True),
            f"{cuda['capability']} (expected {list(EXPECTED_CAPABILITY)}, sm_121)"
            if cuda["capability"]
            else "unknown",
        )
    )
    checks.append(
        (
            "device memory",
            state(cuda["total_memory_gb"] is not None, warn_only=True),
            f"{cuda['free_memory_gb']} GB free of {cuda['total_memory_gb']} GB"
            if cuda["total_memory_gb"] is not None
            else "unknown",
        )
    )
    checks.append(
        (
            "triton kernel",
            state(report["triton"]["usable"], warn_only=not device_ok),
            f"{report['triton']['status']}: {report['triton']['detail']} "
            f"(triton {report['triton']['version'] or 'not installed'})",
        )
    )
    checks.append(
        (
            "unsloth",
            state(unsloth["importable"] and unsloth["version_ok"]),
            f"{unsloth['version'] or 'not installed'} (expected {UNSLOTH_EXPECTED})"
            + (f" - {unsloth['error']}" if unsloth["error"] else ""),
        )
    )
    checks.append(
        (
            "huggingface-hub",
            state(hub["ok"]),
            f"{hub['version'] or 'not installed'} (requires < {hub['max_exclusive']})",
        )
    )
    checks.append(
        (
            "bnb 4-bit",
            state(report["bnb_4bit"]["usable"], warn_only=True),
            f"{report['bnb_4bit']['status']}: {report['bnb_4bit']['detail']}",
        )
    )
    for entry in disk["filesystems"]:
        checks.append(
            (
                "disk " + "+".join(entry["labels"]),
                state(bool(entry["ok"]), warn_only=True),
                f"{entry['free_gb']} GB free at {entry['measured_path']}"
                if entry["free_gb"] is not None
                else str(entry["error"]),
            )
        )
    checks.append(
        (
            "HF token",
            "ok" if report["hf_token"]["present"] else "info",
            f"found via {report['hf_token']['source']}"
            if report["hf_token"]["present"]
            else "not found (public repositories only)",
        )
    )

    print("Cosimo fine-tune environment check")
    print(f"host {report['hostname']}  harness {report['harness_root']}")
    print()
    print(render_table(checks))
    print()

    package_rows = [("PACKAGE", "PIN", "INSTALLED", "STATUS")]
    for row in report["packages"]:
        package_rows.append(
            (
                row["name"] + ("*" if row["critical"] else ""),
                row["pin"],
                row["installed"] or "-",
                row["status"].upper()
                if row["status"] in ("missing", "mismatch")
                else row["status"],
            )
        )
    if report["pins_error"]:
        print(f"pinned versions unavailable: {report['pins_error']}")
    else:
        print(
            f"pins from {REPO_ROOT / 'pyproject.toml'} [dependency-groups] {DEPENDENCY_GROUP}"
        )
        print(render_table(package_rows))
        print("* a mismatch on these is a hard failure")
        if report["pins_skipped"]:
            print(
                f"{len(report['pins_skipped'])} entr(ies) in the group were not checked:"
            )
            for entry in report["pins_skipped"]:
                print(f"  - {entry}")
    print()

    for warning in report["warnings"]:
        print(f"WARN  {warning}")
    for failure in report["failures"]:
        print(f"FAIL  {failure}")
    if not report["warnings"] and not report["failures"]:
        print("No warnings.")
    print()
    print(
        "VERDICT: environment ready"
        if report["ok"]
        else "VERDICT: environment not ready"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        default=str(HARNESS_ROOT / "runs" / "env_check.json"),
        help="where to write the JSON report (default: runs/env_check.json)",
    )
    args = parser.parse_args()

    report = build_report()

    output = Path(args.output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written: str | None = str(output)
    except Exception as exc:
        # Hard, not advisory: every training and evaluation script creates its run directory
        # under the same tree, so an unwritable runs/ kills the very next command.
        written = None
        report["failures"].append(
            f"could not write {output}: {exc}. runs/ must be writable — every training and "
            "evaluation script creates its run directory there"
        )
        report["ok"] = not report["failures"]

    print_report(report)
    if written:
        print(f"Report: {written}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
