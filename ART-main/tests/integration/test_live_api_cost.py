import json
import os
from pathlib import Path
import urllib.request
from uuid import uuid4

import pytest

from art import Model
from art.metrics import track_api_cost

pytestmark = pytest.mark.live_api_cost

_LIVE_ENV = "ART_RUN_LIVE_API_COST_TESTS"


def _require_live_test_env(*required_vars: str) -> None:
    if os.environ.get(_LIVE_ENV) != "1":
        pytest.skip(f"Set {_LIVE_ENV}=1 to run live API cost tests.")
    missing = [name for name in required_vars if not os.environ.get(name)]
    if missing:
        pytest.skip(f"Missing required env vars: {', '.join(missing)}")


def _post_json(url: str, *, headers: dict[str, str], payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _cacheable_prefix(word_count: int = 1500) -> str:
    return " ".join(f"cache-token-{index % 16}" for index in range(word_count))


def _history_rows(history_path: Path) -> list[dict]:
    return [json.loads(line) for line in history_path.read_text().splitlines() if line]


def _openai_completion(*, api_key: str, prompt_cache_key: str, prefix: str) -> dict:
    return _post_json(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload={
            "model": "gpt-4.1",
            "messages": [
                {"role": "system", "content": prefix},
                {"role": "user", "content": "Reply with OK."},
            ],
            "temperature": 0,
            "max_completion_tokens": 4,
            "prompt_cache_key": prompt_cache_key,
        },
    )


def _anthropic_message(*, api_key: str, prefix: str) -> dict:
    return _post_json(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        payload={
            "model": "claude-sonnet-4-6",
            "max_tokens": 8,
            "temperature": 0,
            "system": [
                {
                    "type": "text",
                    "text": prefix,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {"role": "user", "content": "Reply with OK."},
            ],
        },
    )


class TestLiveApiCost:
    @pytest.mark.asyncio
    async def test_openai_gpt_4_1_cached_prompt_cost(self, tmp_path: Path) -> None:
        _require_live_test_env("OPENAI_API_KEY")

        api_key = os.environ["OPENAI_API_KEY"]
        prefix = _cacheable_prefix()
        prompt_cache_key = f"art-live-api-cost-{uuid4()}"

        # Warm the cache first so the tracked request can validate cached pricing.
        _openai_completion(
            api_key=api_key,
            prompt_cache_key=prompt_cache_key,
            prefix=prefix,
        )

        model = Model(
            name="live-openai-api-cost",
            project="live-api-cost",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        @track_api_cost(
            source="llm_judge/openai_cached_prompt",
            provider="openai",
            model_name="openai/gpt-4.1",
        )
        def _judge() -> dict:
            return _openai_completion(
                api_key=api_key,
                prompt_cache_key=prompt_cache_key,
                prefix=prefix,
            )

        token = model.activate_metrics_context("eval")
        try:
            response = _judge()
        finally:
            token.var.reset(token)

        await model.log(trajectories=None, split="val", step=1, metrics={})

        usage = response["usage"]
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        assert cached_tokens > 0

        expected_cost = (
            ((usage["prompt_tokens"] - cached_tokens) * 2.0)
            + (cached_tokens * 0.5)
            + (usage["completion_tokens"] * 8.0)
        ) / 1_000_000

        history_path = (
            tmp_path
            / "live-api-cost"
            / "models"
            / "live-openai-api-cost"
            / "history.jsonl"
        )
        row = _history_rows(history_path)[0]
        assert row["costs/eval/llm_judge/openai_cached_prompt"] == pytest.approx(
            expected_cost
        )

    @pytest.mark.asyncio
    async def test_anthropic_claude_sonnet_4_6_prompt_cache_cost(
        self,
        tmp_path: Path,
    ) -> None:
        _require_live_test_env("ANTHROPIC_API_KEY")

        api_key = os.environ["ANTHROPIC_API_KEY"]
        prefix = _cacheable_prefix()

        model = Model(
            name="live-anthropic-api-cost",
            project="live-api-cost",
            base_path=str(tmp_path),
            report_metrics=[],
        )

        @track_api_cost(
            source="llm_judge/anthropic_prompt_cache",
            provider="anthropic",
            model_name="anthropic/claude-sonnet-4-6",
        )
        def _judge() -> dict:
            return _anthropic_message(api_key=api_key, prefix=prefix)

        token = model.activate_metrics_context("eval")
        try:
            first_response = _judge()
        finally:
            token.var.reset(token)
        await model.log(trajectories=None, split="val", step=1, metrics={})

        token = model.activate_metrics_context("eval")
        try:
            second_response = _judge()
        finally:
            token.var.reset(token)
        await model.log(trajectories=None, split="val", step=2, metrics={})

        first_usage = first_response["usage"]
        second_usage = second_response["usage"]
        assert first_usage.get("cache_creation_input_tokens", 0) > 0
        assert second_usage.get("cache_read_input_tokens", 0) > 0

        first_expected_cost = (
            (first_usage["input_tokens"] * 3.0)
            + (first_usage.get("cache_creation_input_tokens", 0) * 3.75)
            + (first_usage["output_tokens"] * 15.0)
        ) / 1_000_000
        second_expected_cost = (
            (second_usage["input_tokens"] * 3.0)
            + (second_usage.get("cache_read_input_tokens", 0) * 0.30)
            + (second_usage["output_tokens"] * 15.0)
        ) / 1_000_000

        history_path = (
            tmp_path
            / "live-api-cost"
            / "models"
            / "live-anthropic-api-cost"
            / "history.jsonl"
        )
        first_row, second_row = _history_rows(history_path)

        assert first_row[
            "costs/eval/llm_judge/anthropic_prompt_cache"
        ] == pytest.approx(first_expected_cost)
        assert second_row[
            "costs/eval/llm_judge/anthropic_prompt_cache"
        ] == pytest.approx(second_expected_cost)
