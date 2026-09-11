lint:
	uv run ruff check cosimo tests
	uv run ruff format --check cosimo tests

test_unit:
	uv run --group test pytest tests/unit

test_integration:
	uv run --group test pytest tests/integration

test: test_unit test_integration

# v3 corpus control plane (spec §13). The stage targets are the same commands
# the Airflow DAG runs verbatim (ops/airflow/dags/cosimo_v3_corpus.py), so
# "works under make" == "schedulable". All stages -- inventory, packs, smoke,
# render, verify, prefer, publish -- are wired and run for real; the DAG fans
# packs out per work type (--work-type) and render per record type (--types).
# v3-smoke is the CI gate: no teacher, no network, no GPU.
#
# The CLI's flags reach make (amendment §F). They were absent, and their
# absence was the reason `dataset_build.sh` exported COSIMO_V3_LIVE=1 at the
# top: `make v3-render` could only mean "render everything, against whatever
# the environment happens to say", so the only way to ask for a slice was to
# make the environment say live and edit the script. Each variable below is
# empty by default, so an unset one adds no flag at all and every target keeps
# the behaviour it had.
#
#   make v3-render TYPES=analysis,memo LIMIT=20      # a slice, on the fixture
#   make v3-render TYPES=analysis LIMIT=20 LIVE=1    # the same slice, billed
#   make v3-packs WORK=risk.market.var_es            # one computer
#   make v3-render HOLDOUT=1                         # the eval tree (§E)
#   make v3-verify QUICK=1 OUT=/tmp/corpus           # a board over a scratch tree
#
V3 = uv run --group corpus python -m dataset.pipelines.v3.cli
V3_TYPES = $(if $(TYPES),--types $(TYPES))
V3_LIMIT = $(if $(LIMIT),--limit $(LIMIT))
V3_WORK  = $(if $(WORK),--work-type $(WORK))
V3_OUT   = $(if $(OUT),--out $(OUT))
V3_LIVE  = $(if $(LIVE),--live)
V3_QUICK = $(if $(QUICK),--quick)
V3_HOLD  = $(if $(HOLDOUT),--holdout)

v3-inventory:
	$(V3) inventory $(V3_OUT)

v3-packs:
	$(V3) packs $(V3_OUT) $(V3_WORK) $(V3_TYPES) $(V3_LIMIT)

v3-smoke:
	$(V3) smoke $(V3_OUT)

v3-render:
	$(V3) render $(V3_OUT) $(V3_TYPES) $(V3_LIMIT) $(V3_LIVE) $(V3_HOLD)

v3-verify:
	$(V3) verify $(V3_OUT) $(V3_QUICK)

v3-prefer:
	$(V3) prefer $(V3_OUT) $(V3_TYPES) $(V3_LIMIT) $(V3_LIVE) $(V3_HOLD)

v3-publish:
	$(V3) publish $(V3_OUT)

# Is the tree on disk good enough to scale from? The four §F preconditions,
# read off the shards rather than off a remembered run. `dataset_build.sh full`
# refuses on this; run it yourself before raising LIMIT.
v3-slice-audit:
	uv run --group corpus python dataset/tools/slice_audit.py $(V3_OUT)

v3-test:
	uv run --group test pytest dataset/tests/v3 jobs/fine-tune/tests -q
