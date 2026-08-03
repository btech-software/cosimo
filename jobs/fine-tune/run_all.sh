#!/usr/bin/env bash
# End-to-end orchestration of the Cosimo fine-tuning harness.
#
# This script only chains the commands documented in README.md, in order, and
# prints each one before running it. It implements nothing itself: every flag it
# passes exists on the script it passes it to, so anything this does can also be
# done by hand.
#
# Run it from inside the container (see docker/fine-tune/run.sh):
#   ./run_all.sh                 # data -> baseline -> SFT -> DPO -> compare -> export
#   ./run_all.sh --dry-run       # print the plan, run nothing
#   ./run_all.sh --eval-only     # evaluate existing checkpoints, never train
#
# Guard rails:
#   * an existing runs/baseline is never overwritten; the step is skipped loudly
#     unless --force-baseline is given (the baseline is the reference
#     measurement for every later comparison),
#   * --eval-only never invokes a training or export script,
#   * every command is echoed, so the transcript is a record of what ran.
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HARNESS_ROOT}"

PYTHON="${PYTHON:-python}"
SFT_RUN="sft"
DPO_RUN="dpo"

EVAL_ONLY=0
DRY_RUN=0
FORCE_BASELINE=0
FORCE_TRAIN=0
LIMIT=""
SUITES=""

usage() {
    cat <<'EOF'
Usage: ./run_all.sh [options]

Options:
  --eval-only         Evaluate the baseline and any existing adapters, then
                      compare. Never trains, never exports.
  --dry-run           Print every command that would run, run none of them.
                      (Distinct from 03_train_sft.py --dry-run, which builds the
                      trainer for real and skips only .train(); this script runs
                      that as a pre-flight step before real SFT.)
  --limit N           Pass --limit N to every evaluation, for a smoke run.
  --suites "A B"      Pass --suites A B to every evaluation
                      (default: eval.suites from configs/eval.yaml).
  --force-baseline    Re-measure runs/baseline even though it already exists.
  --force-train       Pass --force to the training scripts, overwriting their
                      run directories.
  -h, --help          Show this message.

Steps (full pipeline):
  00_check_env -> 01_prepare_data -> 02_baseline_eval -> 03_train_sft --dry-run
  -> 03_train_sft -> 05_evaluate sft -> 04_train_dpo -> 05_evaluate dpo
  -> 06_compare -> 07_export_merge
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval-only) EVAL_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --limit) LIMIT="${2:?--limit needs a value}"; shift 2 ;;
        --suites) SUITES="${2:?--suites needs a value}"; shift 2 ;;
        --force-baseline) FORCE_BASELINE=1; shift ;;
        --force-train) FORCE_TRAIN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

step() { printf '\n=== %s ===\n' "$*"; }
note() { printf '    %s\n' "$*"; }

run() {
    printf '+ %s\n' "$*"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        return 0
    fi
    "$@"
}

# Evaluation flags shared by 02_baseline_eval.py and 05_evaluate.py.
eval_args=()
if [[ -n "${SUITES}" ]]; then
    # Word splitting is intended: --suites takes several names.
    # shellcheck disable=SC2206
    eval_args+=(--suites ${SUITES})
fi
if [[ -n "${LIMIT}" ]]; then
    eval_args+=(--limit "${LIMIT}")
fi

train_args=()
if [[ "${FORCE_TRAIN}" -eq 1 ]]; then
    train_args+=(--force)
fi

has_adapter() { [[ -f "runs/$1/adapter/adapter_config.json" ]]; }
has_metrics() { [[ -f "runs/$1/eval/metrics.json" ]]; }

if [[ "${DRY_RUN}" -eq 1 ]]; then
    note "dry run: nothing below is executed; skip decisions reflect the current runs/ and data/."
fi

# --------------------------------------------------------------------------
step "0/8 environment check"
run "${PYTHON}" scripts/00_check_env.py

# --------------------------------------------------------------------------
step "1/8 data preparation"
if [[ -f data/processed/split_manifest.json ]]; then
    note "data/processed/split_manifest.json exists; keeping the existing splits."
    note "Re-run ./scripts/01_prepare_data.py --force to rebuild them (this changes"
    note "the split assignment and invalidates every existing run)."
elif [[ "${EVAL_ONLY}" -eq 1 ]]; then
    note "--eval-only, but data/processed/ has not been prepared: the cosimo"
    note "suites cannot be evaluated. Run ./scripts/01_prepare_data.py first."
    [[ "${DRY_RUN}" -eq 1 ]] || exit 1
else
    run "${PYTHON}" scripts/01_prepare_data.py
fi

# --------------------------------------------------------------------------
step "2/8 baseline evaluation (untuned base model)"
if [[ -d runs/baseline && "${FORCE_BASELINE}" -eq 0 ]]; then
    note "runs/baseline already exists and is the reference measurement for every"
    note "comparison, so it is being kept. Pass --force-baseline to re-measure it."
else
    baseline_args=()
    if [[ "${FORCE_BASELINE}" -eq 1 ]]; then
        baseline_args+=(--force)
    fi
    run "${PYTHON}" scripts/02_baseline_eval.py \
        ${baseline_args[@]+"${baseline_args[@]}"} \
        ${eval_args[@]+"${eval_args[@]}"}
fi

compare_runs=(baseline)

if [[ "${EVAL_ONLY}" -eq 1 ]]; then
    # --------------------------------------------------------------------------
    step "3/8 evaluate existing checkpoints (--eval-only: no training)"
    for name in "${SFT_RUN}" "${DPO_RUN}"; do
        if has_adapter "${name}"; then
            run "${PYTHON}" scripts/05_evaluate.py \
                --run-name "${name}" --adapter "runs/${name}/adapter" \
                ${eval_args[@]+"${eval_args[@]}"}
            compare_runs+=("${name}")
        else
            note "runs/${name}/adapter not found; skipping ${name}."
        fi
    done
else
    # --------------------------------------------------------------------------
    step "3/8 SFT pre-flight (builds everything, trains nothing)"
    run "${PYTHON}" scripts/03_train_sft.py --run-name "${SFT_RUN}" --dry-run

    step "4/8 SFT training"
    run "${PYTHON}" scripts/03_train_sft.py --run-name "${SFT_RUN}" \
        ${train_args[@]+"${train_args[@]}"}

    step "5/8 evaluate the SFT adapter"
    run "${PYTHON}" scripts/05_evaluate.py \
        --run-name "${SFT_RUN}" --adapter "runs/${SFT_RUN}/adapter" \
        ${eval_args[@]+"${eval_args[@]}"}
    compare_runs+=("${SFT_RUN}")

    step "6/8 DPO training on top of the SFT adapter"
    run "${PYTHON}" scripts/04_train_dpo.py --run-name "${DPO_RUN}" \
        --sft-adapter "runs/${SFT_RUN}/adapter" \
        ${train_args[@]+"${train_args[@]}"}

    step "7/8 evaluate the DPO adapter"
    run "${PYTHON}" scripts/05_evaluate.py \
        --run-name "${DPO_RUN}" --adapter "runs/${DPO_RUN}/adapter" \
        ${eval_args[@]+"${eval_args[@]}"}
    compare_runs+=("${DPO_RUN}")
fi

# --------------------------------------------------------------------------
step "8/8 comparison"
comparable=()
for name in "${compare_runs[@]}"; do
    if [[ "${DRY_RUN}" -eq 1 ]] || has_metrics "${name}"; then
        comparable+=("${name}")
    else
        note "runs/${name}/eval/metrics.json not found; leaving ${name} out."
    fi
done
if [[ "${#comparable[@]}" -ge 2 ]]; then
    run "${PYTHON}" scripts/06_compare.py --runs "${comparable[@]}"
else
    note "fewer than two evaluated runs; nothing to compare."
fi

if [[ "${EVAL_ONLY}" -eq 0 ]]; then
    step "export: merge the DPO adapter into bf16 weights"
    run "${PYTHON}" scripts/07_export_merge.py --run-name "${DPO_RUN}" \
        ${train_args[@]+"${train_args[@]}"}
fi

printf '\ndone. Artifacts are under %s/runs/\n' "${HARNESS_ROOT}"
if [[ "${EVAL_ONLY}" -eq 0 ]]; then
    printf 'Before shipping, spot-check open-ended financial questions against the\n'
    printf 'baseline: the exam suites do not measure assistant quality (README, "Known limitations").\n'
fi
