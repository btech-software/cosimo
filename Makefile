lint:
	uv run ruff check cosimo tests
	uv run ruff format --check cosimo tests

test_unit:
	uv run --group test pytest tests/unit

test_integration:
	uv run --group test pytest tests/integration

test: test_unit test_integration
