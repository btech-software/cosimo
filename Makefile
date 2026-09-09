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
v3-inventory:
	uv run --group corpus python -m dataset.pipelines.v3.cli inventory

v3-packs:
	uv run --group corpus python -m dataset.pipelines.v3.cli packs

v3-smoke:
	uv run --group corpus python -m dataset.pipelines.v3.cli smoke

v3-render:
	uv run --group corpus python -m dataset.pipelines.v3.cli render

v3-verify:
	uv run --group corpus python -m dataset.pipelines.v3.cli verify

v3-prefer:
	uv run --group corpus python -m dataset.pipelines.v3.cli prefer

v3-publish:
	uv run --group corpus python -m dataset.pipelines.v3.cli publish

v3-test:
	uv run --group test pytest dataset/tests/v3 jobs/fine-tune/tests -q
