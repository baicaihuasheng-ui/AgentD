"""Sampling subprocess stdin compatibility, including real Windows MCP stdio."""
import contextlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize('platform', ['win32', 'linux'])
def test_direct_sampling_only_redirects_windows_stdin(platform, monkeypatch):
    from agentD.tools import generation

    command = ['local-reinvent', '-l', 'sample.log', 'sample.toml']
    monkeypatch.setattr(generation, 'reinvent_command', lambda *args: command)
    run = Mock()
    monkeypatch.setattr(generation.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'platform', platform)
    result = generation.run_reinvent.invoke({'config_file': 'sample.toml'})
    expected = {'check': True}
    if platform == 'win32':
        expected['stdin'] = subprocess.DEVNULL
    run.assert_called_once_with(command, **expected)
    assert result == 'REINVENT execution completed successfully.'


@pytest.mark.parametrize('platform', ['win32', 'linux'])
def test_pooling_only_redirects_windows_stdin(platform, tmp_path, monkeypatch):
    import mcp_agent
    from agentD.tools import generation

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('AGENTD_REINVENT_PATH', str(tmp_path))
    monkeypatch.setenv('AGENTD_MODEL', 'deepseek-flash')
    template = tmp_path / 'configs/toml/sampling.toml'
    template.parent.mkdir(parents=True)
    template.write_text('run_type="sampling"')
    monkeypatch.setattr(mcp_agent, '_load_api_keys_fallback', lambda: None)
    monkeypatch.setattr(generation, 'save_smi_for_mol2mol', Mock())

    def config(model_type):
        path = tmp_path / f'{model_type}.toml'
        path.write_text('num_smiles = 157')
        return str(path)

    monkeypatch.setattr(generation, 'update_reinvent_config', config)
    monkeypatch.setattr(mcp_agent, 'reinvent_command', lambda cfg, log: ['local-reinvent', '-l', log, cfg])
    run = Mock()
    monkeypatch.setattr(mcp_agent.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'platform', platform)
    result = mcp_agent._run_reinvent_pooling(run_dir=tmp_path, initial_smiles='CCO', num_smiles=2)
    expected = {'check': True, 'cwd': str(tmp_path)}
    if platform == 'win32':
        expected['stdin'] = subprocess.DEVNULL
    assert run.call_count == 2
    for call in run.call_args_list:
        assert call.kwargs == expected
        assert call.args[0][0] == 'local-reinvent'
    for path in result['configs'].values():
        assert Path(path).read_text() == 'num_smiles = 2'
    assert result['ok'] is True


SERVER = r'''
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import requests, httpx
def blocked(*args, **kwargs):
    raise AssertionError("Network forbidden in sampling stdio test")
requests.sessions.Session.request = blocked
httpx.HTTPTransport.handle_request = blocked
httpx.AsyncHTTPTransport.handle_async_request = blocked
from agentD.mcp_stdio import run_stdio
def factory():
    from fastmcp import FastMCP
    server = FastMCP("offline-windows-sampling-test")
    @server.tool
    def sampling_smoke() -> dict:
        import mcp_agent
        from agentD.tools import generation
        work = Path(sys.argv[2])
        os.chdir(work)
        os.environ['AGENTD_REINVENT_PATH'] = str(work)
        os.environ['AGENTD_MODEL'] = 'deepseek-flash'
        template = work / 'configs/toml/sampling.toml'
        template.parent.mkdir(parents=True, exist_ok=True)
        template.write_text('run_type="sampling"')
        def config(model_type):
            path = work / (model_type + '.toml')
            path.write_text('num_smiles = 157')
            return str(path)
        def command(*args):
            return [sys.executable, '-S', '-u', '-c', "print('sampling-child-ok', flush=True)"]
        mcp_agent._load_api_keys_fallback = lambda: None
        generation.update_reinvent_config = config
        generation.reinvent_command = command
        mcp_agent.reinvent_command = command
        result = mcp_agent._run_reinvent_pooling(run_dir=work, initial_smiles='CCO', num_smiles=2)
        direct = generation.run_reinvent.invoke({'config_file': str(work / 'Mol2Mol.toml')})
        assert result['ok'] is True
        assert all(value == 'REINVENT execution completed successfully.' for value in result['results'].values())
        assert direct == 'REINVENT execution completed successfully.'
        assert all(Path(path).read_text() == 'num_smiles = 2' for path in result['configs'].values())
        return {'ok': True, 'child_runs': 3, 'num_smiles': 2}
    return server
run_stdio(factory)
'''


@pytest.mark.skipif(sys.platform != 'win32', reason='Actual Windows stdin regression')
def test_real_windows_mcp_sampling_children_complete_and_server_responds(tmp_path):
    import psutil

    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('DEEPSEEK_', 'OPENAI_', 'ANTHROPIC_', 'AGENTD_'))}
    env.update(PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONIOENCODING='utf-8')
    started = time.monotonic()
    deadline = started + 30
    lines = queue.Queue()
    with (tmp_path / 'mcp-server.log').open('wb') as log:
        proc = subprocess.Popen([sys.executable, '-u', '-c', SERVER, str(root), str(tmp_path)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env)

        def read_stdout():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()

        def send(message):
            proc.stdin.write((json.dumps(message) + '\n').encode())
            proc.stdin.flush()

        def receive(identifier):
            while True:
                try:
                    line = lines.get(timeout=max(0.01, deadline - time.monotonic()))
                except queue.Empty:
                    pytest.fail('MCP sampling test exceeded 30-second deadline')
                assert line is not None, 'MCP test server exited before replying'
                message = json.loads(line)
                assert message.get('jsonrpc') == '2.0'
                if message.get('id') == identifier:
                    return message

        try:
            send({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
                'protocolVersion': '2025-06-18', 'capabilities': {},
                'clientInfo': {'name': 'offline-sampling-test', 'version': '1'}}})
            assert 'result' in receive(1)
            send({'jsonrpc': '2.0', 'method': 'notifications/initialized'})
            send({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
                  'params': {'name': 'sampling_smoke', 'arguments': {}}})
            result = receive(2)['result']
            assert not result.get('isError'), result
            payload = json.loads(result['content'][0]['text'])
            assert payload == {'ok': True, 'child_runs': 3, 'num_smiles': 2}
            send({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/list', 'params': {}})
            assert receive(3)['result']['tools'][0]['name'] == 'sampling_smoke'
            proc.stdin.close()
            proc.wait(timeout=max(0.01, deadline - time.monotonic()))
            assert proc.returncode == 0
            print(f'Windows MCP sampling smoke: 3 children completed; elapsed={time.monotonic()-started:.2f}s')
        finally:
            # Only the test server and its own descendants are eligible for cleanup.
            if proc.poll() is None:
                with contextlib.suppress(psutil.NoSuchProcess):
                    children = psutil.Process(proc.pid).children(recursive=True)
                    for child in reversed(children):
                        with contextlib.suppress(psutil.NoSuchProcess):
                            child.kill()
                    proc.kill()
                    proc.wait(timeout=5)
                    psutil.wait_procs(children, timeout=3)
            reader.join(timeout=2)
            for stream in (proc.stdin, proc.stdout):
                if stream and not stream.closed:
                    stream.close()
    assert (tmp_path / 'mcp-server.log').read_text(errors='replace').count('sampling-child-ok') == 3
