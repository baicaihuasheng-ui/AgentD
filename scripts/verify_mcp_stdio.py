import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = r'''
import os, sys, threading, subprocess
sys.path.insert(0, sys.argv[1])
def factory():
    print("startup-noise", flush=True)
    from fastmcp import FastMCP
    m = FastMCP("offline-stdio-check")
    @m.tool
    def noisy(fail: bool = False) -> dict:
        print("tool-entered", file=sys.stderr, flush=True)
        print("Observation: \x1b[36m中文-noise", end="", flush=True)
        sys.stdout.buffer.write(b"binary-noise\\n"); sys.stdout.buffer.flush()
        os.write(1, b"native-noise\\n")
        subprocess.run([sys.executable, "-S", "-u", "-c", "print('child-noise', flush=True)"], check=True, timeout=5, stdin=subprocess.DEVNULL)
        threads = [threading.Thread(target=lambda: print("thread-noise", flush=True)) for _ in range(3)]
        for t in threads: t.start()
        for t in threads: t.join()
        print("large-noise:" + "x" * 140000, flush=True)
        if fail: raise ValueError("offline-expected-error")
        return {"ok": True, "message": "中文响应"}
    @m.tool
    def business_failure() -> dict:
        return {"ok": False, "error": "offline-business-failure"}
    @m.tool
    async def slow() -> dict:
        import asyncio
        await asyncio.sleep(3600)
        return {"ok": True}
    return m
if sys.argv[2] == 'raw':
    factory().run(show_banner=False)
else:
    from agentD.mcp_stdio import run_stdio
    run_stdio(factory)
'''

def verify(raw=False, timeout=40, disconnect=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DEEPSEEK_', 'OPENAI_', 'ANTHROPIC_', 'AGENTD_'))}
    env.update(PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONIOENCODING='utf-8')
    proc = subprocess.Popen([sys.executable, '-u', '-c', SERVER, str(ROOT), 'raw' if raw else 'fixed'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    lines = queue.Queue()
    stderr = bytearray()
    frames = []
    def read_stdout():
        for line in proc.stdout: lines.put(line)
        lines.put(None)
    def read_stderr():
        while True:
            block = proc.stderr.read1(8192)
            if not block: break
            stderr.extend(block)
    readers = [threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)]
    for t in readers: t.start()
    deadline = time.monotonic() + timeout
    def send(value):
        proc.stdin.write((json.dumps(value) + '\n').encode()); proc.stdin.flush()
    def receive(identifier):
        from mcp.types import JSONRPCMessage
        while True:
            try:
                line = lines.get(timeout=max(.01, deadline-time.monotonic()))
            except queue.Empty:
                raise AssertionError('response timeout: ' + repr(frames) + ' STDERR ' + stderr.decode('utf-8', errors='replace')[-3000:])
            assert line is not None, 'server exited before response: ' + stderr.decode('utf-8', errors='replace')
            JSONRPCMessage.model_validate_json(line)
            value = json.loads(line)
            frames.append(value)
            if value.get('id') == identifier: return value
    try:
        send({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-06-18','capabilities':{},'clientInfo':{'name':'offline-check','version':'1'}}})
        receive(1)
        send({'jsonrpc':'2.0','method':'notifications/initialized'})
        send({'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}})
        assert len(receive(2)['result']['tools']) == 3
        for identifier, fail in ((3,False),(4,True),(5,False)):
            send({'jsonrpc':'2.0','id':identifier,'method':'tools/call','params':{'name':'noisy','arguments':{'fail':fail}}})
            result = receive(identifier)['result']
            assert bool(result.get('isError')) == fail, (result, stderr.decode('utf-8', errors='replace')[-2000:])
            if not fail: assert '中文响应' in json.dumps(result, ensure_ascii=False)
        send({'jsonrpc':'2.0','id':6,'method':'tools/call','params':{'name':'business_failure','arguments':{}}})
        assert 'offline-business-failure' in json.dumps(receive(6))
        send({'jsonrpc':'2.0','id':7,'method':'tools/call','params':{'name':'slow','arguments':{}}})
        if not disconnect:
            send({'jsonrpc':'2.0','method':'notifications/cancelled','params':{'requestId':7,'reason':'offline-test'}})
            send({'jsonrpc':'2.0','id':8,'method':'tools/call','params':{'name':'business_failure','arguments':{}}})
            receive(8)
        proc.stdin.close()
        proc.wait(timeout=max(.01, deadline-time.monotonic()))
        for t in readers: t.join(timeout=3)
        while not lines.empty():
            line = lines.get_nowait()
            if line is not None:
                from mcp.types import JSONRPCMessage
                JSONRPCMessage.model_validate_json(line)
        assert proc.returncode == 0
        diagnostic = stderr.decode('utf-8', errors='replace')
        for text in ('startup-noise','Observation:', 'binary-noise', 'native-noise', 'child-noise', 'thread-noise', 'large-noise:', 'offline-expected-error'):
            assert text in diagnostic, text
        return {'status':'passed', 'platform':sys.platform, 'frames':len(frames), 'stderr_bytes':len(stderr), 'live_api_requests':0}
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait(timeout=5)
        for t in readers: t.join(timeout=3)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream and not stream.closed: stream.close()

if __name__ == '__main__':
    if '--raw' in sys.argv:
        print(json.dumps(verify(raw=True), ensure_ascii=False))
    else:
        from pydantic import ValidationError
        from importlib.metadata import version
        result = verify()
        result['disconnect'] = verify(disconnect=True)['status']
        try:
            verify(raw=True)
        except ValidationError:
            result['negative_control'] = 'pollution reproduced'
        else:
            raise AssertionError('Negative control unexpectedly accepted polluted stdout')
        result['packages'] = {name: version(name) for name in ('fastmcp', 'mcp', 'anyio')}
        print(json.dumps(result, ensure_ascii=False))
