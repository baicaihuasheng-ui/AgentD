"""Process-shared request limits and conservative RMB reservations."""
import json
import os
from pathlib import Path
import time
import uuid
from filelock import FileLock
from agentD.affinity import atomic_json


class BudgetExceeded(RuntimeError):
    pass


class CallBudget:
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def create(cls, path, max_rmb=1.0, chat_limit=20, duration_seconds=1800, monetary_only=False):
        path = Path(path)
        if path.exists():
            raise FileExistsError("Budget already exists; use a new run ID")
        state = {"max_rmb": max_rmb, "deadline": None if monetary_only else time.time() + duration_seconds,
                 "monetary_only": monetary_only,
                 "limits": {"chat": None, "search": None, "fetch": None} if monetary_only else {"chat": chat_limit, "search": 6, "fetch": 3},
                 "counts": {"chat": 0, "search": 0, "fetch": 0}, "spent_rmb": 0.0,
                 "reserved_rmb": 0.0, "events": [],
                 "prices": {"input_per_million": 2.0, "output_per_million": 8.0, "search": 0.005, "fetch": 0.005}}
        atomic_json(path, state)
        return cls(path)

    @property
    def monetary_only(self):
        return json.loads(self.path.read_text()).get("monetary_only", False)

    def check_deadline(self):
        deadline = json.loads(self.path.read_text())["deadline"]
        if deadline is not None and time.time() >= deadline:
            raise BudgetExceeded("Run deadline exceeded")

    def reserve(self, kind, input_bytes=0, output_tokens=0):
        with FileLock(str(self.path) + ".lock"):
            state = json.loads(self.path.read_text())
            if state["deadline"] is not None and time.time() >= state["deadline"]:
                raise BudgetExceeded("Run deadline exceeded")
            if state["limits"][kind] is not None and state["counts"][kind] >= state["limits"][kind]:
                raise BudgetExceeded(f"{kind} request limit exceeded")
            if kind == "chat":
                if not state.get("monetary_only", False) and (input_bytes > 32768 or output_tokens > 2048):
                    raise BudgetExceeded("Chat payload/output exceeds this test's limits")
                cost = ((input_bytes * 2 + 512) * 2 + output_tokens * 8) / 1_000_000
            else:
                cost = state["prices"][kind]
            if state["spent_rmb"] + state["reserved_rmb"] + cost > state["max_rmb"]:
                raise BudgetExceeded("Insufficient remaining RMB budget")
            token = uuid.uuid4().hex
            state["counts"][kind] += 1
            state["reserved_rmb"] += cost
            state["events"].append({"id": token, "kind": kind, "reserved_rmb": cost, "status": "reserved"})
            atomic_json(self.path, state)
            return token

    def finish(self, token, usage=None, success=True):
        with FileLock(str(self.path) + ".lock"):
            state = json.loads(self.path.read_text())
            event = next(e for e in state["events"] if e["id"] == token)
            if event["status"] != "reserved":
                raise ValueError("Request already settled")
            cost = event["reserved_rmb"]
            if event["kind"] == "chat" and usage and "prompt_tokens" in usage and "completion_tokens" in usage:
                cost = (usage["prompt_tokens"] * 2 + usage["completion_tokens"] * 8) / 1_000_000
            state["reserved_rmb"] = max(0, state["reserved_rmb"] - event["reserved_rmb"])
            state["spent_rmb"] += cost
            event.update(status="succeeded" if success else "failed", charged_upper_rmb=cost)
            if usage:
                event["usage"] = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens") if key in usage}
            atomic_json(self.path, state)


def current_budget():
    path = os.getenv("AGENTD_BUDGET_PATH")
    return CallBudget(path) if path else None
