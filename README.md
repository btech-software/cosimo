# Cosimo I

Reasoning Language Model Post-Trained with Financial Domain Synthetic Dataset

Built on [Agent-Lab](https://github.com/btech-software/agent-lab) (`btech-agent-lab`).

## Quickstart

```bash
uv lock            # required once before docker builds (uv sync --frozen needs it)
uv sync --group dev --group test
cp .env.example .env

make lint
make test_unit     # no-container smoke tests: app boots, examples registered
```

Run locally (needs Postgres, Redis, and Vault reachable per `config-dev.yml`):

```bash
DEVELOPING=1 uv run uvicorn cosimo.main:app --reload
```

Or run the full stack in containers:

```bash
docker compose up --build -d
curl http://localhost:18000/status/liveness
```


## Project layout

- `cosimo/main.py` — the composition root: `create_app()` scans only this
  project's packages, so none of Agent-Lab's built-in agents/tools are exposed.
  Add `"agent_lab.services.agent_types"` / `"agent_lab.interface.mcp"` to
  `scan_packages` to opt back in.
- `cosimo/core/container.py` — add your own dependency-injection providers
  here; reference them from agents via `extra_deps`.
- `cosimo/agents/` — one example echo agent (`@discoverable_agent`) and a
  single-node ReAct workflow agent whose checkpointed state stores a
  chain-of-thought per turn.
- `cosimo/mcp/` — one example MCP tool and prompt, exposed at `/mcp`.


## Prompt templates

Agent prompts stored in agent settings are Jinja2 templates rendered at
runtime, e.g. `Today is {{ CURRENT_TIME }}.` — see
`agents/react_agent/default_execution_system_prompt.txt`.


## Integration tests

```bash
make test_integration   # boots Postgres, Redis, and Vault via testcontainers
```

