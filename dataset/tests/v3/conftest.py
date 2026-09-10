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
