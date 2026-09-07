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
