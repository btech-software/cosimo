
```bash
cd ./dataset
rm -rf shards progress
PER_TEMPLATE=250 uv run python3 -m pipelines.generate
uv run python3 verification/verify_all.py
uv run  python3 goldbar/validate.py
rm -rf .smoke && uv run python3 scripts/smoke_generate.py
```
