"""The v3 teacher: one client, two model names, OpenAI-v1 only (spec §5.4).

No vendor SDKs. The ETL must run on a CPU box with nothing but the standard
library (spec §9), and every endpoint we point it at -- OpenAI, vLLM, SGLang,
Azure, a local teacher -- speaks chat completions. Consequence: the transport
is one ``post(body) -> dict`` hop, injected, so the offline test suite, the
replay fixture, and production share one ``Teacher`` with one contract.

Offline discipline (spec §10, PR1): nothing here may reach the network unless
``COSIMO_V3_LIVE=1``. With no opt-in the factory hands back the *fixture*
transport, and a fixture miss raises rather than improvises -- a generation
run that silently fell back to a canned answer would be worse than one that
stops.
"""

from .client import (
    FixtureTransport,
    HttpTransport,
    Teacher,
    TeacherError,
    TeacherResult,
    teacher_from_env,
)
from .routing import Route, route

__all__ = [
    "FixtureTransport",
    "HttpTransport",
    "Route",
    "Teacher",
    "TeacherError",
    "TeacherResult",
    "route",
    "teacher_from_env",
]
