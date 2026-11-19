import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_monitor_tracks_only_bound_run_and_reports_substeps(tmp_path, capsys):
    import run_agentd
    own = tmp_path / "runs/own"
    other = tmp_path / "runs/hf-modules"
    other.mkdir(parents=True)
    (other / "status.json").write_text(json.dumps({"current_step": "wrong"}))

    async def exercise():
        stop = asyncio.Event()
        task = asyncio.create_task(run_agentd.monitor_progress_background(stop, own, interval=0.001))
        await asyncio.sleep(0.01)
        own.mkdir()
        status = own / "status.json"
        status.write_text("{")
        await asyncio.sleep(0.01)
        status.write_text(json.dumps({"current_step": "prediction", "steps": [{"details": {"substep": "waiting"}}]}))
        await asyncio.sleep(0.01)
        status.write_text(json.dumps({"current_step": "prediction", "steps": [{"details": {"substep": "completed"}}]}))
        stop.set()
        await task

    asyncio.run(exercise())
    output = capsys.readouterr().out
    assert "等待初始化" in output
    assert "substep=waiting" in output and "substep=completed" in output
    assert "wrong" not in output and "hf-modules" not in output
    assert output.count("substep=waiting") == 1


@pytest.mark.parametrize("run_id", [None, "resume_explicit"])
@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_pipeline_binds_id_and_cleans_monitor(tmp_path, monkeypatch, capsys, run_id, failure):
    import run_agentd
    monkeypatch.setattr(run_agentd, "__file__", str(tmp_path / "run_agentd.py"))
    monkeypatch.setattr(run_agentd, "StdioTransport", lambda **kwargs: object())
    calls = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def list_tools(self):
            return []

        async def call_tool(self, name, parameters):
            calls.append(parameters)
            path = tmp_path / "runs" / parameters["run_id"]
            path.mkdir(parents=True, exist_ok=True)
            (path / "status.json").write_text(json.dumps({"current_step": "pipeline_complete", "steps": []}))
            if failure:
                raise failure("original failure")
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps({"ok": True}))])

    monkeypatch.setattr(run_agentd, "Client", Client)
    config = {**run_agentd.DEFAULT_CONFIG}
    if run_id:
        config["run_id"] = run_id

    async def exercise():
        if failure:
            with pytest.raises(failure, match="original failure"):
                await run_agentd.run_pipeline_mode(config)
        else:
            assert await run_agentd.run_pipeline_mode(config)
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(exercise())
    actual_id = calls[0]["run_id"]
    if run_id:
        assert actual_id == run_id
    else:
        assert actual_id and "run_id" not in config
    output = capsys.readouterr().out
    assert str((tmp_path / "runs" / actual_id).resolve()) in output
    assert "pipeline_complete" in output


def test_monitor_reports_io_errors(tmp_path, capsys):
    import run_agentd
    root = tmp_path / "run"
    root.mkdir()
    (root / "status.json").mkdir()

    async def exercise():
        stop = asyncio.Event()
        stop.set()
        await run_agentd.monitor_progress_background(stop, root)

    asyncio.run(exercise())
    assert "进度读取失败" in capsys.readouterr().out


def test_concurrent_monitors_keep_their_own_status(tmp_path, capsys, monkeypatch):
    import run_agentd
    reads = {"one": [], "two": []}
    read_text = Path.read_text

    def record_read(path, *args, **kwargs):
        if path.name == "status.json" and path.parent.name in reads:
            reads[asyncio.current_task().get_name()].append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_read)

    async def exercise():
        stops = [asyncio.Event(), asyncio.Event()]
        roots = [tmp_path / "one", tmp_path / "two"]
        tasks = [asyncio.create_task(run_agentd.monitor_progress_background(stop, root, interval=0.001), name=root.name)
                 for stop, root in zip(stops, roots)]
        await asyncio.sleep(0.01)
        for index, root in enumerate(roots):
            root.mkdir()
            (root / "status.json").write_text(json.dumps({"current_step": f"own_step_{index}"}))
        for stop in stops:
            stop.set()
        await asyncio.gather(*tasks)

    asyncio.run(exercise())
    output = capsys.readouterr().out
    assert output.count("own_step_0") == 1
    assert output.count("own_step_1") == 1
    for name, paths in reads.items():
        assert paths and all(path == tmp_path / name / "status.json" for path in paths)
