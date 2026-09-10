#!/usr/bin/env bash

export TEACHER_BASE_URL=http://192.168.2.198:8888
export TEACHER_API_KEY=local-no-auth
export TEACHER_PROSE=qwen3.8-flash-next
export TEACHER_REASONING=qwen3.8-flash-next
export TEACHER_TIMEOUT_S=900
export COSIMO_V3_LIVE=1

make v3-inventory     # seconds — writes plan.json (2810 jobs)
make v3-packs         # ~1 min  — computes fact packs, CPU only
make v3-render        # ← the long one: every record type
make v3-prefer        # preference pairs, needs shipped rows
make v3-verify        # the 14-axis board
make v3-publish       # will refuse: no gold_bar_v3.jsonl

