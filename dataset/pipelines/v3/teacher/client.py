"""One OpenAI-v1 chat-completions client, stdlib only (spec §5.4).

The whole production surface is :meth:`Teacher.complete`; everything below it
exists so that the bytes on the wire are a pure function of the call
arguments:

* :class:`HttpTransport` -- the real POST (stdlib ``urllib``, no SDK);
* :class:`FixtureTransport` -- replay of a captured/crafted JSON, keyed by
  :func:`canonical_request`, so an offline run is *identical* call for call
  to the captured run and a miss is an error naming the hash, not a fallback;
* :func:`teacher_from_env` -- the only place that reads the environment,
  enforcing the ``COSIMO_V3_LIVE=1`` opt-in for anything that can bill.

Retries deliberately do not live here. The teacher loop's retry policy is a
*generation* policy (temperature down, repair prompt, dead letter -- spec
§6.2) and it belongs to the renderers; a transport that silently re-sends a
POST would make "what did the teacher actually answer" ambiguous under the
exactly-duplicate-request conditions the fixture harness creates.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .. import config


class TeacherError(RuntimeError):
    """The teacher could not be asked, or answered in a shapeless way."""


def canonical_request(body: dict) -> str:
    """The replay key for a request body: hash of its canonical JSON.

    ``sort_keys`` + fixed separators + ``ensure_ascii=False``: the same call
    from a Spark executor and from a laptop must land on the same entry, so
    the bytes hashed are specified here rather than left to two json writers.
    """
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf8")).hexdigest()


@dataclass(frozen=True)
class TeacherResult:
    """Everything a shard must record about one completion (spec §5.4).

    ``think`` is the vendor's chain-of-thought channel when one was requested
    (``reasoning_content`` in vLLM/SGLang-style payloads); it rides along for
    the audit trail and the ``verification.teacher`` block, never into the
    student's target text.
    """

    model: str
    text: str
    think: str | None
    finish_reason: str
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)

    def to_verification(self) -> dict:
        """The dict written onto ``verification.teacher`` (schema §5)."""
        return {
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "think_present": self.think is not None and self.think != "",
        }


class HttpTransport:
    """POST {base_url}/v1/chat/completions. One call, one POST, no retries."""

    def __init__(self, base_url: str, api_key: str, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = float(timeout_s)

    def post(self, body: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf8"),
            headers={
                "contenttype": "application/json",
                "accept": "application/json",
                "authorization": f"bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with _urlopen(request, timeout=self.timeout_s) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:  # auth/429/5xx: the caller decides
            snippet = _read_snippet(exc)
            raise TeacherError(
                f"teacher endpoint {exc.code} at {self.base_url}: {snippet}"
            ) from exc
        except OSError as exc:  # urllib.error.URLError subclasses OSError
            raise TeacherError(
                f"teacher unreachable at {self.base_url}: {exc}"
            ) from exc
        try:
            decoded = json.loads(payload.decode("utf8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise TeacherError(
                f"teacher replied with non-json bytes at {self.base_url}"
            ) from exc
        if not isinstance(decoded, dict):
            raise TeacherError(f"teacher reply is not a json object at {self.base_url}")
        return decoded


def _urlopen(request, timeout):  # pragma: no cover -- monkeypatched in tests
    """Seam for tests: the only touch of the network stack in the module."""
    return urllib.request.urlopen(request, timeout=timeout)


def _read_snippet(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read()[:300].decode("utf8", errors="replace")
    except OSError:
        return "<unreadable error body>"


class FixtureTransport:
    """Replay a JSON fixture: ``{"entries": {canonical_hash: reply}, "*": ...}``.

    An exact match always wins. The ``"*"`` entry, when present, is the
    wildcard for requests no captured reply covers (the smoke run walks
    hundreds of packs nobody wants pinned to fixtures one by one) -- its text
    is still *data from the file*, never generated at call time, so a fixture
    run remains a fixture run. A miss without a wildcard raises with the hash
    so the operator can extend the file deterministically.
    """

    def __init__(self, path: str | None = None, entries: dict | None = None):
        if entries is not None:
            self._entries = entries
        elif path is not None:
            self._entries = _load_fixture(path)
        else:
            raise ValueError("FixtureTransport needs path or entries")

    def post(self, body: dict) -> dict:
        digest = canonical_request(body)
        reply = self._entries.get(digest) or self._entries.get("*")
        if reply is None:
            raise TeacherError(
                f"fixture miss for request {digest}; add it to "
                "dataset/tests/v3/fixtures/ or run live (COSIMO_V3_LIVE=1)"
            )
        if not isinstance(reply, dict):
            raise TeacherError(f"fixture entry {digest} is not a json object")
        return reply


def _load_fixture(path: str) -> dict:
    with open(path, encoding="utf8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        raise TeacherError(
            f"fixture {path!r} must be a json object with an 'entries' mapping"
        )
    return data["entries"]


def _parse(payload: dict, requested_model: str) -> TeacherResult:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TeacherError(f"teacher reply has no choices: {_brief(payload)}")
    first = choices[0]
    if not isinstance(first, dict):
        raise TeacherError(f"teacher choice is not an object: {_brief(payload)}")
    message = first.get("message") or {}
    if not isinstance(message, dict):
        raise TeacherError(f"teacher message is not an object: {_brief(payload)}")
    text = message.get("content")
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise TeacherError("teacher content is not a string")
    think = message.get("reasoning_content")
    if think is None:
        think = message.get("reasoning")
    if think is not None and not isinstance(think, str):
        raise TeacherError("teacher reasoning channel is not a string")
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    return TeacherResult(
        model=str(payload.get("model") or requested_model),
        text=text,
        think=think,
        finish_reason=str(first.get("finish_reason") or "stop"),
        usage=dict(usage),
        raw=payload,
    )


def _brief(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)[:300]


class Teacher:
    """The single generation entry point (spec §5.4)."""

    def __init__(self, transport):
        self.transport = transport

    def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        think: bool = False,
        extra: dict | None = None,
    ) -> TeacherResult:
        """Ask for one completion. ``temperature`` is passed through untouched:
        the repair loop's "temperature down" (spec §6.2) sets it, the client
        must not second-guess the generation policy."""
        for turn in messages:
            if (
                not isinstance(turn, dict)
                or "role" not in turn
                or "content" not in turn
            ):
                raise TeacherError(f"malformed message turn: {turn!r}")
        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if think:
            body["thinking"] = {"type": "enabled"}
        if extra:
            body.update(extra)
        return _parse(self.transport.post(body), model)


def teacher_from_env(transport=None, *, live: bool | None = None) -> Teacher:
    """Build the teacher the environment asks for; the live path is opt-in.

    * an injected ``transport`` (a fake, a fixture object) wins -- tests never
      need the environment at all;
    * live means env: ``TEACHER_BASE_URL`` + ``TEACHER_API_KEY`` must be set,
      missing ones are named in the error (an ETL operator staring at "401"
      learns less than one staring at "TEACHER_API_KEY not set");
    * offline (the default, and every CI box) replays
      ``COSIMO_V3_TEACHER_FIXTURE`` -- and refuses loudly if even that is
      absent.
    """
    if transport is not None:
        return Teacher(transport)
    if live is None:
        live = config.live_enabled()
    if live:
        missing = [
            name
            for name in (config.TEACHER_BASE_URL_ENV, config.TEACHER_API_KEY_ENV)
            if not os.environ.get(name)
        ]
        if missing:
            raise TeacherError(
                f"live teacher requested but {', '.join(missing)} not set"
            )
        timeout = float(
            os.environ.get(config.TEACHER_TIMEOUT_ENV, config.DEFAULT_TEACHER_TIMEOUT_S)
        )
        return Teacher(
            HttpTransport(
                os.environ[config.TEACHER_BASE_URL_ENV],
                os.environ[config.TEACHER_API_KEY_ENV],
                timeout_s=timeout,
            )
        )
    path = (
        os.environ.get(config.TEACHER_FIXTURE_ENV) or config.default_teacher_fixture()
    )
    if not os.path.isfile(path):
        raise TeacherError(
            f"offline teacher: fixture {path!r} not found; set "
            f"{config.TEACHER_FIXTURE_ENV} or accept live calls with "
            f"{config.LIVE_ENV}=1"
        )
    return Teacher(FixtureTransport(path=path))
