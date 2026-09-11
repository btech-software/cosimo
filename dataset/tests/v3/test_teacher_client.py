"""The teacher client's contracts, offline, on injected fakes (spec §5.4).

Nothing here opens a socket: ``urlopen`` is reached only through the module
level ``_urlopen`` seam, which the HTTP tests replace so the *real* request
construction -- url, headers, body bytes, error translation -- is exercised
while the network stays hypothetical.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from pipelines.v3 import config
from pipelines.v3.teacher.client import (
    build_body,
    DEFAULT_MAX_TOKENS,
    FixtureTransport,
    HttpTransport,
    Teacher,
    TeacherError,
    canonical_request,
    teacher_from_env,
)


def _reply(text="yes", think=None):
    message = {"role": "assistant", "content": text}
    if think is not None:
        message["reasoning_content"] = think
    return {
        "model": "m-1",
        "choices": [{"finish_reason": "stop", "message": message}],
        "usage": {"total_tokens": 7},
    }


class RecordingTransport:
    def __init__(self, reply=None):
        self.calls = []
        self.reply = reply if reply is not None else _reply()

    def post(self, body):
        self.calls.append(body)
        return self.reply


def test_complete_builds_the_openai_v1_body_exactly():
    transport = RecordingTransport()
    result = Teacher(transport).complete(
        [{"role": "user", "content": "hi"}], model="m-1", temperature=0.3, max_tokens=64
    )
    # The think state is always stated, in both dialects, even when nobody
    # asked for thinking -- see `test_both_thinking_dialects_are_always_sent`
    # for what omitting it cost.
    assert transport.calls[0] == {
        "model": "m-1",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.3,
        "max_tokens": 64,
        "thinking": {"type": "disabled"},
        "chat_template_kwargs": {"thinking": False},
    }
    assert (result.text, result.model, result.finish_reason) == ("yes", "m-1", "stop")
    assert result.usage == {"total_tokens": 7}


def test_think_adds_the_extension_and_extra_merges_top_level():
    transport = RecordingTransport(_reply("x", think="because chains"))
    result = Teacher(transport).complete(
        [{"role": "user", "content": "q"}],
        model="m",
        think=True,
        extra={"seed": 5},
    )
    body = transport.calls[0]
    assert body["thinking"] == {"type": "enabled"}
    assert body["seed"] == 5
    assert result.think == "because chains"
    assert result.to_verification()["think_present"] is True


def test_a_bodyless_turn_is_refused_before_the_wire_is_touched():
    transport = RecordingTransport()
    with pytest.raises(TeacherError, match="malformed message turn"):
        Teacher(transport).complete([{"role": "user"}], model="m")
    assert transport.calls == []


@pytest.mark.parametrize(
    "payload, needle",
    [
        ({"choices": []}, "no choices"),
        ({"choices": ["x"]}, "not an object"),
        ({"choices": [{"message": "x"}]}, "not an object"),
        ({"choices": [{"message": {"content": 3}}]}, "not a string"),
        ({}, "no choices"),
    ],
)
def test_shapeless_replies_are_named_not_trusted(payload, needle):
    class Shape(RecordingTransport):
        def post(self, body):
            self.calls.append(body)
            return payload

    with pytest.raises(TeacherError, match=needle):
        Teacher(Shape()).complete([{"role": "user", "content": "q"}], model="m")


def test_missing_finish_and_usage_degrade_to_honest_defaults():
    transport = RecordingTransport({"choices": [{"message": {"content": "ok"}}]})
    result = Teacher(transport).complete(
        [{"role": "user", "content": "q"}], model="asked"
    )
    assert result.finish_reason == "stop"
    assert result.usage == {}
    assert result.model == "asked", "reply must not invent the model that answered"


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


def test_http_transport_posts_json_with_authorisation(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["headers"] = request.headers
        seen["data"] = request.data
        seen["timeout"] = timeout
        return FakeResponse(json.dumps(_reply("hello")).encode("utf8"))

    monkeypatch.setattr("pipelines.v3.teacher.client._urlopen", fake_urlopen)
    transport = HttpTransport("https://teacher.example/api/", "sekret", timeout_s=7)
    result = Teacher(transport).complete([{"role": "user", "content": "hi"}], model="m")
    assert seen["url"] == "https://teacher.example/api/v1/chat/completions"
    assert seen["method"] == "POST"
    assert seen["headers"]["Authorization"] == "bearer sekret"
    # `Content-Type`, hyphenated, as urllib normalises it. This assertion used
    # to read `Contenttype` -- it pinned the misspelling rather than the
    # contract, so it passed against a header no HTTP server would honour, and
    # a strict endpoint answered 400 on the first live call. Asserted through
    # `urllib`'s own lookup so the spelling that reaches the wire is the
    # spelling under test.
    assert seen["headers"]["Content-type"] == "application/json"
    assert "Contenttype" not in seen["headers"], "the misspelling is back"
    # The deadline is derived (budget / TEACHER_TOKENS_PER_SECOND, floored at
    # timeout_s), so assert the contract rather than a literal that moves
    # whenever a lane's budget does.
    budget = json.loads(seen["data"].decode("utf8"))["max_tokens"]
    assert seen["timeout"] == max(7, budget / config.TEACHER_TOKENS_PER_SECOND)
    assert json.loads(seen["data"].decode("utf8"))["model"] == "m"
    assert result.text == "hello"


def test_http_errors_become_teachererrors_naming_the_endpoint(monkeypatch):
    def http_401(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            __import__("io").BytesIO(b"bad key"),
        )

    def unreachable(request, timeout):
        raise urllib.error.URLError("connection refused")

    client_module = __import__("pipelines.v3.teacher.client", fromlist=["_"])
    monkeypatch.setattr(client_module, "_urlopen", http_401)
    with pytest.raises(TeacherError, match="401"):
        Teacher(HttpTransport("https://t.example", "k")).complete(
            [{"role": "user", "content": "q"}], model="m"
        )
    monkeypatch.setattr(client_module, "_urlopen", unreachable)
    with pytest.raises(TeacherError, match="unreachable"):
        Teacher(HttpTransport("https://t.example", "k")).complete(
            [{"role": "user", "content": "q"}], model="m"
        )


def test_the_deadline_follows_the_work_not_a_ratio(monkeypatch):
    """A wall clock against a variable token budget is the wrong shape.

    The deadline used to be ``timeout_s * (budget / DEFAULT_MAX_TOKENS)``,
    which was right while every call asked for 16384 and wrong the moment the
    lanes carried their own budgets: it scaled the allowance *down* toward the
    flat 120s for exactly the calls that needed patience. Three of four live
    probe calls at a 20,000-token ceiling timed out while the fourth returned
    5,080 tokens in 144s -- the endpoint was never the problem, the arithmetic
    was.

    So the deadline is now the work divided by a declared throughput, floored
    at ``timeout_s``. Never *less* than the floor: queueing and prefill do not
    shrink with the budget.
    """
    seen = {}

    def fake_urlopen(request, timeout):
        seen[json.loads(request.data)["max_tokens"]] = timeout
        return FakeResponse(json.dumps(_reply()).encode())

    monkeypatch.setattr("pipelines.v3.teacher.client._urlopen", fake_urlopen)
    teacher = Teacher(HttpTransport("https://t.example", "k", timeout_s=100))
    messages = [{"role": "user", "content": "q"}]
    rate = config.TEACHER_TOKENS_PER_SECOND
    small, large = 800, 12800
    for budget in (small, large):
        teacher.complete(messages, model="m", max_tokens=budget)

    # Below the floor the configured allowance wins: a 800-token call at 20
    # tok/s wants 40s, and nobody gains from a 40s deadline.
    assert seen[small] == 100, "timeout_s is a floor, never scaled down"
    # Above it, the deadline is the work. 12,800 tokens at 20 tok/s is 640s --
    # which the old formula would have cut to 100s, timing out a call the
    # teacher was going to answer.
    assert seen[large] == large / rate
    assert seen[large] > seen[small], "more tokens, more patience"


def test_non_json_reply_is_refused(monkeypatch):
    monkeypatch.setattr(
        "pipelines.v3.teacher.client._urlopen",
        lambda request, timeout: FakeResponse(b"<html>not json</html>"),
    )
    with pytest.raises(TeacherError, match="non-json"):
        Teacher(HttpTransport("https://t.example", "k")).complete(
            [{"role": "user", "content": "q"}], model="m"
        )


def test_fixture_transport_hits_exact_wildcards_and_misses_loudly():
    messages = [{"role": "user", "content": "hi"}]
    # The budget is the client's own default, not a literal: this test is
    # about exact-beats-wildcard, and pinning a number here made it fail for
    # the unrelated reason that the default moved (1024 -> 16384 when the
    # teacher became a reasoning model).
    # Built through `build_body`, not by hand: the replay key is a hash of the
    # *whole* body, so a hand-written dict silently stops matching the moment a
    # field is added -- and the test then passes on the wildcard while claiming
    # to prove exact-beats-wildcard. That is precisely what happened when the
    # thinking dialects were added.
    body = build_body(
        messages, model="m", temperature=0.7, max_tokens=DEFAULT_MAX_TOKENS
    )
    entries = {
        canonical_request(body): {"choices": [{"message": {"content": "matched"}}]},
        "*": {"choices": [{"message": {"content": "wildcard"}}]},
    }
    transport = FixtureTransport(entries=entries)
    assert Teacher(transport).complete(messages, model="m").text == "matched"
    assert Teacher(transport).complete(
        [{"role": "user", "content": "?"}], model="m"
    ).text == ("wildcard")
    strict = FixtureTransport(entries=dict(entries))
    del strict._entries["*"]
    with pytest.raises(
        TeacherError, match="fixture miss for request [0-9a-f]{64}"
    ) as err:
        Teacher(strict).complete([{"role": "user", "content": "?"}], model="m")
    assert canonical_request(body) not in str(err.value)
    assert canonical_request(
        build_body(
            [{"role": "user", "content": "?"}],
            model="m",
            temperature=0.7,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
    ) in str(err.value), (
        "a miss must name the hash the operator will need to extend the fixture"
    )


def test_fixture_file_shape_is_validated(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"nope": 1}')
    with pytest.raises(TeacherError, match="entries"):
        FixtureTransport(path=str(bad))
    good = tmp_path / "good.json"
    good.write_text('{"entries": {}}')
    FixtureTransport(path=str(good))  # loads, empty, no raise


def test_from_env_offline_uses_the_fixture_never_the_network(monkeypatch, tmp_path):
    fixture = tmp_path / "fx.json"
    fixture.write_text(
        '{"entries": {"*": {"choices": [{"message": {"content": "replay"}}]}}}'
    )
    monkeypatch.setenv(config.TEACHER_FIXTURE_ENV, str(fixture))
    monkeypatch.setenv(config.LIVE_ENV, "0")
    teacher = teacher_from_env(live=False)
    assert isinstance(teacher.transport, FixtureTransport)
    assert (
        teacher.complete([{"role": "user", "content": "q"}], model="m").text == "replay"
    )


def test_from_env_live_demands_and_honours_the_env_quadruple(monkeypatch):
    for name in (config.TEACHER_BASE_URL_ENV, config.TEACHER_API_KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(TeacherError, match="TEACHER_BASE_URL"):
        teacher_from_env(live=True)
    monkeypatch.setenv(config.TEACHER_BASE_URL_ENV, "https://t.example/")
    monkeypatch.setenv(config.TEACHER_API_KEY_ENV, "k")
    monkeypatch.setenv(config.TEACHER_TIMEOUT_ENV, "9")
    teacher = teacher_from_env(live=True)
    assert isinstance(teacher.transport, HttpTransport)
    assert teacher.transport.base_url == "https://t.example"
    assert teacher.transport.timeout_s == 9.0
    monkeypatch.delenv(config.TEACHER_TIMEOUT_ENV, raising=False)
    assert teacher_from_env(live=True).transport.timeout_s == float(
        config.DEFAULT_TEACHER_TIMEOUT_S
    )


def test_injected_transport_short_circuits_the_environment(monkeypatch):
    monkeypatch.setenv(config.LIVE_ENV, "1")  # env must not win over injection
    monkeypatch.setenv(config.TEACHER_BASE_URL_ENV, "https://should-not-be-used")
    transport = RecordingTransport()
    assert isinstance(teacher_from_env(transport=transport), Teacher)
    teacher_from_env(transport=transport).complete(
        [{"role": "user", "content": "q"}], model="m"
    )
    assert transport.calls[0]["model"] == "m"


def test_opt_in_absent_and_no_fixture_is_a_hard_stop(monkeypatch):
    monkeypatch.setenv(config.LIVE_ENV, "1")
    monkeypatch.setenv(config.TEACHER_BASE_URL_ENV, "https://t.example")
    monkeypatch.setenv(config.TEACHER_API_KEY_ENV, "k")
    monkeypatch.setenv(config.LIVE_ENV, "")  # not "1" => offline
    monkeypatch.setenv(config.TEACHER_FIXTURE_ENV, "/nonexistent/echo.json")
    with pytest.raises(TeacherError, match="not found"):
        teacher_from_env()


def test_both_thinking_dialects_are_always_sent(monkeypatch):
    """The think flag must reach the model, whichever field the stack reads.

    This is the regression guard for the most expensive defect in the project.
    `build_body` sent only `thinking: {"type": "enabled"}` -- DeepSeek's *cloud*
    dialect -- and sent nothing at all for think=False. A vLLM/SparkInfer serve
    reads the toggle out of the model's chat template via
    `chat_template_kwargs`, ignores the cloud field, and defaults to
    `thinking: true` when the kwarg is absent.

    So "think off" was never off. Every prose row paid for a full chain of
    thought, and the resulting 3,212 reasoning tokens were read as "this
    teacher reasons unconditionally" rather than "the client is talking to the
    wrong field". Measured on deepseek-v4-flash-0731, same brief: 3,780
    completion tokens with the kwarg omitted against 358 with it set false --
    ten times the bill for a shorter answer.

    Both dialects, both states, never omitted. An unread field is inert; an
    absent one hands the decision to somebody else's default.
    """
    seen = {}

    def fake_urlopen(request, timeout):
        seen.update(json.loads(request.data))
        return FakeResponse(json.dumps(_reply()).encode())

    monkeypatch.setattr("pipelines.v3.teacher.client._urlopen", fake_urlopen)
    teacher = Teacher(HttpTransport("https://t.example", "k"))
    messages = [{"role": "user", "content": "q"}]

    teacher.complete(messages, model="m", think=True)
    assert seen["thinking"] == {"type": "enabled"}
    assert seen["chat_template_kwargs"] == {"thinking": True}

    teacher.complete(messages, model="m", think=False)
    assert seen["thinking"] == {"type": "disabled"}, "off must be stated, not omitted"
    assert seen["chat_template_kwargs"] == {"thinking": False}


def test_extra_still_wins_over_the_thinking_defaults(monkeypatch):
    """A caller probing a stack's dialect must be able to override both."""
    seen = {}

    def fake_urlopen(request, timeout):
        seen.update(json.loads(request.data))
        return FakeResponse(json.dumps(_reply()).encode())

    monkeypatch.setattr("pipelines.v3.teacher.client._urlopen", fake_urlopen)
    Teacher(HttpTransport("https://t.example", "k")).complete(
        [{"role": "user", "content": "q"}],
        model="m",
        think=False,
        extra={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "low"}},
    )
    assert seen["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": "low"}
