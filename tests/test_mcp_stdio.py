import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_mcp_stdio.py'
spec = importlib.util.spec_from_file_location('stdio_verifier', SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

def test_old_stdio_reproduces_stdout_pollution():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        verifier.verify(raw=True)

def test_real_stdio_isolates_noise_and_delivers_final_responses():
    assert verifier.verify()['status'] == 'passed'


def test_server_exits_when_client_disconnects_during_tool():
    assert verifier.verify(disconnect=True)['status'] == 'passed'

def test_client_transport_reaps_sync_blocked_server(tmp_path):
    import asyncio
    import os
    import sys
    import psutil
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    code = verifier.SERVER.replace(
        'async def slow() -> dict:\n        import asyncio\n        await asyncio.sleep(3600)',
        'def slow() -> dict:\n        import time\n        time.sleep(300)',
    )
    assert code != verifier.SERVER
    async def exercise():
        transport = StdioTransport(
            command=sys.executable, args=['-u', '-c', code, str(verifier.ROOT), 'fixed'],
            keep_alive=False, log_file=tmp_path/'server.log',
            env={'PYTHONNOUSERSITE':'1', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1'},
        )
        before = {p.pid for p in psutil.Process().children()}
        async with Client(transport, timeout=15) as client:
            server_pids = {p.pid for p in psutil.Process().children()} - before
            assert server_pids
            with pytest.raises(Exception, match='(?i)timed out|timeout'):
                await client.call_tool('slow', {}, timeout=1)
        assert not [p.pid for p in psutil.Process().children() if p.pid in server_pids and p.is_running()]
    asyncio.run(exercise())
