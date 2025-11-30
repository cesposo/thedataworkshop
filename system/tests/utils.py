"""Test helpers."""

import itertools
from typing import Callable, Dict, Any


class DummyCommunicator:
    """In-memory communicator used in tests to avoid opening sockets."""

    _counter = itertools.count(8100)
    _registry: Dict[str, "DummyCommunicator"] = {}

    def __init__(self, host: str = "localhost", port: int = 0):
        self.host = host or "localhost"
        # Allocate a pseudo-port so addresses look realistic
        self.port = port if port not in (None, 0) else next(self._counter)
        self._functions: Dict[str, Callable[..., Any]] = {}
        self._address = f"http://{self.host}:{self.port}"
        self._registry[self._address] = self

    @property
    def address(self) -> str:
        return self._address

    def register_function(self, func: Callable[..., Any], name: str) -> None:
        self._functions[name] = func

    def start_server(self) -> None:
        # No-op for in-memory communicator
        return None

    def stop_server(self) -> None:
        self._registry.pop(self._address, None)

    def send(self, destination: str, message: Dict[str, Any]):
        dest = self._registry.get(destination)
        if dest is None:
            return None
        method = message.get("method")
        params = message.get("params", [])
        func = dest._functions.get(method)
        if func is None:
            return None
        return func(*params)

    def receive(self):
        return None

    def broadcast(self, message):
        return None
