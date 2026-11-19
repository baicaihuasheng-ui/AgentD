import json
import pytest


def test_money_only_has_no_count_or_deadline_limits(tmp_path, monkeypatch):
    import agentD.call_budget as module
    path = tmp_path / "money.json"
    budget = module.CallBudget.create(path, max_rmb=5, monetary_only=True)
    monkeypatch.setattr(module.time, "time", lambda: 10**12)
    budget.check_deadline()
    for kind, count in (("chat", 21), ("search", 7), ("fetch", 4)):
        for _ in range(count):
            token = budget.reserve(kind, input_bytes=40000, output_tokens=4096)
            budget.finish(token, {"prompt_tokens": 1, "completion_tokens": 1})
    data = json.loads(path.read_text())
    assert data["deadline"] is None
    assert data["limits"] == {"chat": None, "search": None, "fetch": None}
    assert data["max_rmb"] == 5


def test_money_only_still_blocks_excess_cost(tmp_path):
    from agentD.call_budget import CallBudget, BudgetExceeded
    budget = CallBudget.create(tmp_path / "money.json", max_rmb=0.004, monetary_only=True)
    with pytest.raises(BudgetExceeded, match="RMB"):
        budget.reserve("search")


def test_money_only_uses_normal_output_default(tmp_path, monkeypatch):
    from agentD.call_budget import CallBudget
    from agentD.providers.deepseek import get_chat_model
    path = tmp_path / "money.json"
    CallBudget.create(path, max_rmb=5, monetary_only=True)
    monkeypatch.setenv("AGENTD_BUDGET_PATH", str(path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "mock")
    model = get_chat_model()
    assert model.max_tokens == 4096


def test_budget_counts_and_usage_are_shared(tmp_path):
    from agentD.call_budget import CallBudget, BudgetExceeded
    path = tmp_path / "usage.json"
    first = CallBudget.create(path, chat_limit=1)
    token = first.reserve("chat", input_bytes=100, output_tokens=2048)
    second = CallBudget(path)
    with pytest.raises(BudgetExceeded):
        second.reserve("chat", input_bytes=100, output_tokens=2048)
    first.finish(token, {"prompt_tokens": 100, "completion_tokens": 2})
    data = json.loads(path.read_text())
    assert data["counts"]["chat"] == 1
    assert data["spent_rmb"] > 0 and data["reserved_rmb"] == 0
    assert data["events"][0]["usage"]["prompt_tokens"] == 100


def test_budget_blocks_fee_and_deadline_before_requests(tmp_path):
    from agentD.call_budget import CallBudget, BudgetExceeded
    fee = CallBudget.create(tmp_path / "fee.json", max_rmb=0.001)
    with pytest.raises(BudgetExceeded):
        fee.reserve("search")
    expired = CallBudget.create(tmp_path / "time.json", duration_seconds=-1)
    with pytest.raises(BudgetExceeded):
        expired.reserve("fetch")


def test_chat_budget_records_usage_and_disables_retry(tmp_path, monkeypatch):
    import httpx
    from agentD.call_budget import CallBudget
    from agentD.providers.deepseek import get_chat_model
    path = tmp_path / "usage.json"
    CallBudget.create(path)
    monkeypatch.setenv("AGENTD_BUDGET_PATH", str(path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "mock-secret-not-real")
    calls = []
    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0, "model": "deepseek-flash",
             "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}})
    model = get_chat_model(http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    assert model.invoke("short prompt").content == "OK"
    assert model.max_retries == 0 and calls[0]["max_tokens"] == 2048
    state = json.loads(path.read_text())
    assert state["counts"]["chat"] == 1
    assert state["events"][0]["usage"]["total_tokens"] == 13


def test_budget_search_failure_is_one_attempt(tmp_path, monkeypatch):
    from agentD.call_budget import CallBudget
    from agentD.providers.deepseek_search import DeepSeekSearch, ServiceError
    from types import SimpleNamespace
    path = tmp_path / "usage.json"
    CallBudget.create(path)
    monkeypatch.setenv("AGENTD_BUDGET_PATH", str(path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "mock")
    calls = []
    service = DeepSeekSearch(session=SimpleNamespace(post=lambda *a, **k: (calls.append(1) or SimpleNamespace(status_code=429))))
    with pytest.raises(ServiceError):
        service.search("BCL2")
    assert calls == [1]


def test_budget_limit_is_atomic_between_processes(tmp_path):
    import subprocess
    import sys
    from agentD.call_budget import CallBudget
    path = tmp_path / "shared.json"
    CallBudget.create(path, chat_limit=1)
    code = """import sys
from agentD.call_budget import CallBudget, BudgetExceeded
try:
    CallBudget(sys.argv[1]).reserve('chat', input_bytes=100, output_tokens=2048)
except BudgetExceeded:
    sys.exit(7)
"""
    processes = [subprocess.Popen([sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
    try:
        for process in processes:
            process.communicate(timeout=60)
        assert sorted(process.returncode for process in processes) == [0, 7]
        assert json.loads(path.read_text())["counts"]["chat"] == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
