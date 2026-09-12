"""Path bootstrap for the v3 corpus tests.

The corpus is not an installed package: like every other module under
``dataset/`` it imports through the ``sys.path`` idiom (see
``verification/gates.py`` and ``pipelines/generate.py``), which keeps the CLI
runnable from a bare checkout on a Spark box or a CI runner with no install
step. The tests join the same identities -- ``pipelines.core``, not a shadow
copy -- so a green suite verifies the code the generator actually runs.

``REPO`` goes on the path too: v3 legitimately imports the shared contracts in
``cosimo.tools`` (the one sanctioned cross-subsystem edge).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.dirname(os.path.dirname(_HERE))  # dataset/tests/v3 -> dataset
REPO = os.path.dirname(DATASET)

for _p in (DATASET, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The offline suite replays request *hashes*, and the completion budget is part
# of every hashed body -- `client.DEFAULT_MAX_TOKENS` reads COSIMO_V3_MAX_TOKENS
# once, at import time. `.env.example` ships that variable, so a developer who
# has followed the setup instructions and exported it would watch 27 tests fail
# for a reason that has nothing to do with their change. The committed fixtures
# are built against the client's *built-in* default (the makers drop the
# variable for the same reason), so the suite drops it too. This must run before
# anything imports the client, which is what a conftest is for.
os.environ.pop("COSIMO_V3_MAX_TOKENS", None)

# The gold bar is a *curated human artefact* that lives in the repo, and since
# the renderer learned to skip the coordinates it holds (a gold-barred seed is
# not a seed the corpus regenerates), a suite that read the committed file
# would change behaviour every time somebody certified a row: eight tests began
# failing the hour `gold_bar_v3.jsonl` was first written, all of them counting
# rows that were suddenly, correctly, not rendered.
#
# So the suite points the path at a file that does not exist. Tests that are
# *about* the bar set `COSIMO_V3_GOLDBAR` themselves (monkeypatch wins over
# this, being per-test), and every other test renders the plan it declares.
os.environ.setdefault("COSIMO_V3_GOLDBAR", os.path.join(_HERE, "_no_gold_bar.jsonl"))
os.environ["COSIMO_V3_GOLDBAR"] = os.path.join(_HERE, "_no_gold_bar.jsonl")

