#!/usr/bin/env bash
#
# The v3 corpus operator entry point. Modes, not a firehose (amendment §F).
#
# What this script used to be: six `make` lines under a header that exported
# COSIMO_V3_LIVE=1, TEACHER_TIMEOUT_S=900 and the same model name into both
# teacher lanes. Every one of those was load-bearing and wrong.
#
#   * `COSIMO_V3_LIVE=1` at the top made a live, unbounded render the operator
#     *default*: running the file at all billed the teacher for 2,810 jobs, and
#     there was no way to ask for a slice without editing it.
#   * `TEACHER_REASONING=qwen3.8-flash-next` flattened the two lanes into one.
#     config.py already routes reasoning to DeepSeek and prose to Qwen, and
#     every row records which model answered it -- so a script that overrode
#     both made the corpus's own provenance stamp a record of the script rather
#     than of a decision. It is a bake-off knob and it now has to be asked for.
#   * `TEACHER_TIMEOUT_S=900` existed because a think-on prose lane at a 16,384
#     token budget genuinely needed it. §B removes both, so the 120s default in
#     config.py is the ceiling again -- and a live slice that exceeds it is a
#     brief bug to fix, not a wall clock to raise.
#
# Usage:
#   ./dataset_build.sh                 # smoke: no teacher, no network, no GPU
#   ./dataset_build.sh slice           # fixture render of a small slice
#   ./dataset_build.sh live-slice      # the same slice against the real teacher
#   ./dataset_build.sh prefer-slice    # pairs over what a slice already shipped
#   ./dataset_build.sh full            # everything; refuses without CONFIRM_FULL=1
#
# Knobs (all optional): TYPES, LIMIT, WORK, OUT, QUICK, BAKEOFF, CONFIRM_FULL.

set -euo pipefail

MODE="${1:-smoke}"

# The endpoint. Not the models: those are config.py's two defaults, and the
# stamp on every row is only worth reading while the script leaves them alone.
export TEACHER_BASE_URL="${TEACHER_BASE_URL:-http://192.168.2.198:8888}"
export TEACHER_API_KEY="${TEACHER_API_KEY:-local-no-auth}"

# A bake-off is the one reason to flatten the lanes, and it says so out loud.
if [[ -n "${BAKEOFF:-}" ]]; then
  echo "bake-off: both lanes pinned to ${BAKEOFF} (the row stamps will say so)"
  export TEACHER_REASONING="${BAKEOFF}"
  export TEACHER_PROSE="${BAKEOFF}"
fi

TYPES="${TYPES:-analysis,memo,grounded,abstention,critique}"
LIMIT="${LIMIT:-20}"

# OUT is exported, never passed as --out. The CLI's flag means two different
# things: on `inventory` it is the *plan file* to write, on every other stage it
# is the *corpus root* -- so a script that forwarded one OUT to both would write
# the plan where the tree belongs and then fail to find it. `COSIMO_V3_OUT` has
# exactly one meaning (config.out_dir), and every stage reads it.
[[ -n "${OUT:-}" ]] && export COSIMO_V3_OUT="$OUT"
CORPUS="${COSIMO_V3_OUT:-dataset/shards/v3}"

v3() { uv run --group corpus python -m dataset.pipelines.v3.cli "$@"; }

stage_order() {
  cat <<'TXT'

Stage order for any live path (amendment §F):

  test -> smoke -> inventory -> packs -> render(limit, types) -> verify --quick
                                                            \-> human read
  prefer is not on the happy path until register_ok is boring.
  publish is last and still needs the gold bar.

TXT
}

# ---------------------------------------------------------------- full's gate
#
# `full` is the only mode that can spend the whole budget, so it is the only
# one with preconditions. Each refusal below is a lesson already paid for once.
refuse_unless_ready() {
  local out="$CORPUS" problems=()

  [[ "${CONFIRM_FULL:-}" == "1" ]] || problems+=(
    "CONFIRM_FULL=1 is not set: a full render is a budget decision, not a default"
  )

  local goldbar="${COSIMO_V3_GOLDBAR:-dataset/goldbar/gold_bar_v3.jsonl}"
  [[ -f "$goldbar" ]] || problems+=(
    "no gold bar at ${goldbar}: publish will refuse anyway, and a corpus you
     cannot certify is a corpus you should not pay to generate"
  )

  # The last slice has to be clean on the three things §F names. Read off the
  # shards rather than off a remembered run: a green board in somebody's
  # scrollback is not evidence about the tree on disk.
  if [[ -d "${out}/sft" ]]; then
    local audit
    audit="$(uv run --group corpus python dataset/tools/slice_audit.py --out "$out" || true)"
    echo "$audit"
    grep -q '^slice_audit: ready$' <<<"$audit" || problems+=(
      "the last slice is not clean: see the slice_audit lines above"
    )
  else
    problems+=("no sft/ tree at ${out}: run a slice and read it before scaling")
  fi

  if (( ${#problems[@]} )); then
    echo
    echo "REFUSING full render:"
    printf '  - %s\n' "${problems[@]}"
    stage_order
    exit 1
  fi
}

case "$MODE" in
  smoke)
    # The CI gate: packs only, no teacher, no network, no GPU.
    v3 smoke
    ;;

  packs)
    v3 inventory
    v3 packs ${WORK:+--work-type "$WORK"}
    ;;

  slice)
    # Fixture render. The default, and the one an operator should reach for
    # first: it exercises every stage boundary and bills nobody.
    v3 inventory
    v3 packs ${WORK:+--work-type "$WORK"}
    v3 render --types "$TYPES" --limit "$LIMIT" ${WORK:+--work-type "$WORK"}
    v3 verify --quick
    echo
    echo "slice done. Read the rows before raising LIMIT:"
    echo "  ${CORPUS}/sft/*.jsonl"
    ;;

  live-slice)
    # The same slice, billed. `--live` is per-invocation and deliberately not
    # an export: nothing downstream of this line inherits a licence to spend.
    v3 inventory
    v3 packs ${WORK:+--work-type "$WORK"}
    v3 render --types "$TYPES" --limit "$LIMIT" ${WORK:+--work-type "$WORK"} --live
    v3 verify --quick
    echo
    echo "live slice done. Now read 20 of them by hand -- that is the gate."
    ;;

  eval-slice)
    # §E: the holdout families render too, into eval/ rather than sft/. Without
    # this the "unseen scenario family" measurement has nothing to measure and
    # the harness has to hold out shipped families instead.
    v3 render --types "$TYPES" --limit "$LIMIT" ${WORK:+--work-type "$WORK"} --holdout
    v3 verify --quick
    ;;

  prefer-slice)
    # Verify before prefer. That is the amendment to spec §7, and the order
    # here is the whole statement of it.
    v3 verify --quick
    v3 prefer --limit "$LIMIT"
    ;;

  full)
    refuse_unless_ready
    v3 inventory
    v3 packs
    v3 render --live
    v3 render --live --holdout
    v3 verify
    v3 prefer --live
    v3 verify
    v3 publish
    ;;

  *)
    echo "unknown mode: ${MODE}" >&2
    echo "modes: smoke | packs | slice | live-slice | eval-slice | prefer-slice | full" >&2
    stage_order
    exit 2
    ;;
esac
