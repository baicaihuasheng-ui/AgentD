"""Isolate MCP's wire from Python, native and inherited subprocess stdout.

Compatibility adapter for FastMCP 2.14.x / MCP 1.30.x. Private FastMCP
lifecycle access is kept here; application tools never own the wire handle.
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
from collections.abc import Callable


def run_stdio(factory: Callable):
    """Run a FastMCP factory with a private, non-inherited protocol descriptor."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = sys.stdout
    protocol_fd = os.dup(1)
    os.set_inheritable(protocol_fd, False)
    wire = None
    windows_stdout = None
    if os.name == 'nt':
        import ctypes
        import msvcrt
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        kernel32.SetStdHandle.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        windows_stdout = kernel32.GetStdHandle(-11 & 0xffffffff)
    try:
        # Capture native output as well as Python output before importing tools.
        os.dup2(2, 1)
        if windows_stdout is not None:
            if not kernel32.SetStdHandle(-11 & 0xffffffff, msvcrt.get_osfhandle(1)):
                raise ctypes.WinError(ctypes.get_last_error())
        sys.stdout = sys.stderr
        wire = os.fdopen(os.dup(protocol_fd), 'w', encoding='utf-8', newline='\n', buffering=1)
        os.set_inheritable(wire.fileno(), False)
        import anyio
        from mcp.server.stdio import stdio_server
        from mcp.server.lowlevel.server import NotificationOptions

        server = factory()
        if ('stdout' not in inspect.signature(stdio_server).parameters
                or not callable(getattr(server, '_lifespan_manager', None))
                or not callable(getattr(getattr(server, '_mcp_server', None), 'run', None))):
            raise RuntimeError('Unsupported MCP/FastMCP stdio interfaces; restore the previous patch')

        async def serve():
            async with server._lifespan_manager():
                async with stdio_server(stdout=anyio.wrap_file(wire)) as (read_stream, write_stream):
                    low_level = server._mcp_server
                    # This SDK can retain in-flight requests after input EOF.
                    # Forward without filtering, and cancel the session on disconnect.
                    sender, receiver = anyio.create_memory_object_stream(0)
                    async with sender, receiver:
                        async with anyio.create_task_group() as tasks:
                            async def forward_input():
                                try:
                                    async for message in read_stream:
                                        await sender.send(message)
                                finally:
                                    tasks.cancel_scope.cancel()
                            tasks.start_soon(forward_input)
                            await low_level.run(
                                receiver, write_stream,
                                low_level.create_initialization_options(
                                    notification_options=NotificationOptions(tools_changed=True)),
                            )
                            tasks.cancel_scope.cancel()
        anyio.run(serve)
    except BaseException:
        logging.getLogger(__name__).exception('MCP stdio service stopped with an error')
        raise
    finally:
        try:
            sys.stdout.flush()
        finally:
            sys.stdout = saved_stdout
            os.dup2(protocol_fd, 1)
            if windows_stdout is not None:
                kernel32.SetStdHandle(-11 & 0xffffffff, msvcrt.get_osfhandle(1))
            if wire is not None:
                wire.close()
            os.close(protocol_fd)
