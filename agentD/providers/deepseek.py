from langchain_openai import ChatOpenAI
from agentD.runtime_config import load_runtime
import json
from agentD.call_budget import current_budget


class DeepSeekCallError(RuntimeError):
    pass


class DeepSeekChatOpenAI(ChatOpenAI):
    """Keep DeepSeek's documented token-limit field on the pinned SDK stack."""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload

    def _stream(self, *args, **kwargs):
        if current_budget():
            raise ValueError("Bounded test requires non-streaming calls")
        yield from super()._stream(*args, **kwargs)

    async def _astream(self, *args, **kwargs):
        if current_budget():
            raise ValueError("Bounded test requires non-streaming calls")
        async for chunk in super()._astream(*args, **kwargs):
            yield chunk

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        budget = current_budget()
        if budget is None:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        if payload.get("model") != "deepseek-flash" or payload.get("stream"):
            raise ValueError("Bounded test requires non-streaming deepseek-flash")
        token = budget.reserve("chat", len(json.dumps(payload, ensure_ascii=False).encode()), payload["max_tokens"])
        try:
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as error:
            budget.finish(token, success=False)
            raise DeepSeekCallError(f"DeepSeek chat failed ({type(error).__name__}, HTTP {getattr(error, 'status_code', None)})") from None
        budget.finish(token, (result.llm_output or {}).get("token_usage"))
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        budget = current_budget()
        if budget is None:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        if payload.get("model") != "deepseek-flash" or payload.get("stream"):
            raise ValueError("Bounded test requires non-streaming deepseek-flash")
        token = budget.reserve("chat", len(json.dumps(payload, ensure_ascii=False).encode()), payload["max_tokens"])
        try:
            result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as error:
            budget.finish(token, success=False)
            raise DeepSeekCallError(f"DeepSeek chat failed ({type(error).__name__}, HTTP {getattr(error, 'status_code', None)})") from None
        budget.finish(token, (result.llm_output or {}).get("token_usage"))
        return result


def get_chat_model(model=None, temperature=0.0, **kwargs):
    cfg = load_runtime()
    options = dict(
        temperature=temperature,
        max_tokens=4096,
        timeout=60,
        max_retries=2,
        extra_body={"thinking": {"type": "disabled"}},
    )
    options.update(kwargs)
    budget = current_budget()
    if budget:
        options.update(max_retries=0, streaming=False)
        if not budget.monetary_only:
            options["max_tokens"] = 2048
    return DeepSeekChatOpenAI(
        model=model or cfg.model,
        api_key=cfg.require_key(),
        base_url=cfg.api_base,
        **options,
    )
