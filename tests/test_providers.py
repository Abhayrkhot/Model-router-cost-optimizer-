import gzip
import json

import anthropic
import httpx
import httpx2
import pytest
import respx

from modelrouter.gateway.config import Deployment
from modelrouter.gateway.providers.anthropic_provider import AnthropicProvider, to_messages_api
from modelrouter.gateway.providers.base import Completion, ProviderError, ProviderRequest
from modelrouter.gateway.providers.openai_compat import OpenAICompatProvider
from modelrouter.gateway.providers.replay import ReplayProvider, ReplayStore
from modelrouter.gateway.schemas import ChatMessage


def msgs(*pairs):
    return [ChatMessage(role=r, content=c) for r, c in pairs]


def claude(upstream, handler, **dep):
    d = Deployment(
        name="c",
        provider="anthropic",
        upstream_model=upstream,
        price_input_per_mtok=1.0,
        price_output_per_mtok=5.0,
        **dep,
    )
    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url="http://anthropic.test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicProvider(d, client=client)


def message_json(text="hello", stop="end_turn", model="claude-haiku-5-5"):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }


def test_messages_mapping_merges_turns_and_extracts_system():
    system, turns = to_messages_api(
        msgs(("system", "be brief"), ("user", "a"), ("user", "b"), ("assistant", "c"))
    )
    assert system == "be brief"
    assert turns == [{"role": "user", "content": "a\n\nb"}, {"role": "assistant", "content": "c"}]
    with pytest.raises(ProviderError):
        to_messages_api(msgs(("assistant", "x")))


async def test_claude_completion_and_cost():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["path"] = request.url.path
        return httpx2.Response(200, json=message_json())

    p = claude("claude-haiku-5-5", handler)
    c = await p.complete(ProviderRequest(msgs(("user", "hi")), 64, temperature=0.3))
    assert c.text == "hello"
    assert (c.input_tokens, c.output_tokens) == (12, 3)
    assert c.cost_usd == pytest.approx((12 * 1.0 + 3 * 5.0) / 1e6)
    assert seen["path"] == "/v1/messages"
    assert "temperature" not in seen["body"]  # sampling params are not forwarded
    assert seen["body"]["max_tokens"] == 64


async def test_claude_server_side_fallback_on_supported_models():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta", "")
        return httpx2.Response(200, json=message_json(model="claude-sonnet-5-5"))

    await claude("claude-sonnet-5-5", handler).complete(ProviderRequest(msgs(("user", "x")), 8))
    assert seen["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in seen["beta"]


async def test_claude_refusal_is_a_fallback_error():
    p = claude(
        "claude-haiku-5-5", lambda r: httpx2.Response(200, json=message_json(stop="refusal"))
    )
    with pytest.raises(ProviderError) as err:
        await p.complete(ProviderRequest(msgs(("user", "x")), 8))
    assert err.value.kind == "refusal"
    assert err.value.fallback
    assert not err.value.retryable


@pytest.mark.parametrize(
    ("status", "etype", "kind"),
    [
        (429, "rate_limit_error", "rate_limited"),
        (529, "overloaded_error", "overloaded"),
        (500, "api_error", "server"),
        (400, "invalid_request_error", "client"),
    ],
)
async def test_claude_error_classification(status, etype, kind):
    body = {"type": "error", "error": {"type": etype, "message": "nope"}}
    p = claude("claude-haiku-5-5", lambda r: httpx2.Response(status, json=body))
    with pytest.raises(ProviderError) as err:
        await p.complete(ProviderRequest(msgs(("user", "x")), 8))
    assert err.value.kind == kind


@respx.mock
async def test_openai_compat_complete_and_errors():
    d = Deployment(
        name="local",
        provider="openai_compat",
        upstream_model="qwen",
        base_url="http://llm.test/v1",
        price_input_per_mtok=0.0,
    )
    route = respx.post("http://llm.test/v1/chat/completions")
    route.mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "qwen",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )
    )
    p = OpenAICompatProvider(d)
    c = await p.complete(ProviderRequest(msgs(("user", "x")), 8))
    assert (c.text, c.input_tokens, c.output_tokens) == ("ok", 4, 1)
    route.mock(return_value=httpx.Response(503, text="busy"))
    with pytest.raises(ProviderError) as err:
        await p.complete(ProviderRequest(msgs(("user", "x")), 8))
    assert err.value.kind == "overloaded"
    await p.aclose()


async def test_replay_provider_serves_recorded_answers(tmp_path):
    path = tmp_path / "fixture.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        row = {
            "sample_id": "s1",
            "family": "gsm8k",
            "turns": ["TASK", "2+2?"],
            "models": {"m": {"text": "4", "cost": 0.002, "quality": 1.0}},
        }
        fh.write(json.dumps(row) + "\n")
    store = ReplayStore.load(path)
    d = Deployment(name="m-dep", provider="replay", upstream_model="m")
    p = ReplayProvider(d, store, seed=0)
    c = await p.complete(ProviderRequest(msgs(("user", "TASK"), ("user", "2+2?")), 8))
    assert (c.text, c.cost_usd, c.extra["recorded_quality"]) == ("4", 0.002, 1.0)
    chunks = [
        x async for x in p.stream(ProviderRequest(msgs(("user", "TASK"), ("user", "2+2?")), 8))
    ]
    assert isinstance(chunks[-1], Completion)
    with pytest.raises(ProviderError) as err:
        await p.complete(ProviderRequest(msgs(("user", "unknown")), 8))
    assert err.value.kind == "client"
    p.failure_rate = 1.0
    with pytest.raises(ProviderError) as err:
        await p.complete(ProviderRequest(msgs(("user", "TASK"), ("user", "2+2?")), 8))
    assert err.value.kind == "overloaded"
