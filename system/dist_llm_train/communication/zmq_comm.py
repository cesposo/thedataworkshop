"""ZeroMQ-based communicator with MessagePack serialization.

This communicator replaces XML-RPC to provide efficient binary serialization
for large gradient tensors, dramatically reducing network overhead.
"""

import zmq
import msgpack
import threading
import logging
from typing import Any, Dict, Callable, Optional
import time

logger = logging.getLogger("dist_llm_train.zmq_comm")


class ZMQCommunicator:
    """
    High-performance communicator using ZeroMQ with MessagePack serialization.

    Benefits over XML-RPC:
    - Binary serialization (msgpack) vs text-based XML
    - Native support for numpy arrays and binary data
    - Lower latency and higher throughput
    - Better handling of large messages
    """

    def __init__(self, host: str, port: int, mode: str = 'router'):
        """
        Initialize ZMQ communicator.

        Args:
            host: Host address to bind/connect
            port: Port number
            mode: 'router' for server (controller), 'dealer' for client (worker)
        """
        self.host = host
        self.port = port
        self.mode = mode
        self.address = f"tcp://{host}:{port}"

        # ZMQ context and socket
        self.context = zmq.Context()

        if mode == 'router':
            # Server mode - binds and listens
            self.socket = self.context.socket(zmq.ROUTER)
            self.socket.bind(self.address)
            logger.info(f"ZMQ ROUTER bound to {self.address}")

            # Function registry for RPC-style calls
            self.functions: Dict[str, Callable] = {}

            # Start server thread
            self.running = True
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()

        elif mode == 'dealer':
            # Client mode - connects to server
            self.socket = self.context.socket(zmq.DEALER)
            self.socket.connect(self.address)
            logger.info(f"ZMQ DEALER connected to {self.address}")

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'router' or 'dealer'")

    def register_function(self, func: Callable, name: str = None):
        """
        Register a function for RPC calls (server mode only).

        Args:
            func: Function to register
            name: Name to register under (defaults to func.__name__)
        """
        if self.mode != 'router':
            raise RuntimeError("Can only register functions in router (server) mode")

        name = name or func.__name__
        self.functions[name] = func
        logger.debug(f"Registered function: {name}")

    def _server_loop(self):
        """Server loop for handling incoming requests (router mode)."""
        while self.running:
            try:
                # Receive message with identity
                # ROUTER socket prepends sender identity
                identity = self.socket.recv()
                message_bytes = self.socket.recv()

                # Deserialize request
                request = msgpack.unpackb(message_bytes, raw=False)

                # Process request
                method = request.get('method')
                params = request.get('params', [])

                # Call registered function
                if method in self.functions:
                    try:
                        result = self.functions[method](*params)
                        response = {'result': result, 'error': None}
                    except Exception as e:
                        logger.error(f"Error executing {method}: {e}", exc_info=True)
                        response = {'result': None, 'error': str(e)}
                else:
                    logger.warning(f"Unknown method: {method}")
                    response = {'result': None, 'error': f"Unknown method: {method}"}

                # Send response
                response_bytes = msgpack.packb(response, use_bin_type=True)
                self.socket.send(identity, zmq.SNDMORE)
                self.socket.send(response_bytes)

            except zmq.ZMQError as e:
                if self.running:
                    logger.error(f"ZMQ error in server loop: {e}")
            except Exception as e:
                logger.error(f"Unexpected error in server loop: {e}", exc_info=True)

    def send(self, address: str, message: Dict[str, Any], timeout: int = 30000) -> Optional[Any]:
        """
        Send a message and wait for response (client mode).

        Args:
            address: Target address (ignored in ZMQ DEALER mode, uses connected address)
            message: Message dictionary with 'method' and 'params'
            timeout: Timeout in milliseconds

        Returns:
            Response result or None if error
        """
        if self.mode != 'dealer':
            raise RuntimeError("send() is only for dealer (client) mode")

        try:
            # Serialize and send request
            request_bytes = msgpack.packb(message, use_bin_type=True)
            self.socket.send(request_bytes)

            # Wait for response with timeout
            if self.socket.poll(timeout, zmq.POLLIN):
                response_bytes = self.socket.recv()
                response = msgpack.unpackb(response_bytes, raw=False)

                if response.get('error'):
                    logger.error(f"RPC error: {response['error']}")
                    return None

                return response.get('result')
            else:
                logger.error(f"Request timeout after {timeout}ms")
                return None

        except Exception as e:
            logger.error(f"Error sending message: {e}", exc_info=True)
            return None

    def start(self):
        """Start the communicator (compatibility with RPC interface)."""
        if self.mode == 'router' and not self.server_thread.is_alive():
            self.running = True
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()
        logger.info(f"ZMQ communicator started in {self.mode} mode")

    def stop(self):
        """Stop the communicator and clean up resources."""
        logger.info("Stopping ZMQ communicator")
        self.running = False

        if hasattr(self, 'server_thread'):
            self.server_thread.join(timeout=2)

        if self.socket:
            self.socket.close()

        if self.context:
            self.context.term()

        logger.info("ZMQ communicator stopped")


class HybridCommunicator:
    """
    Hybrid communicator that falls back to XML-RPC if ZeroMQ is unavailable.

    Attempts to use ZMQ for efficiency, but can fall back to XML-RPC for
    compatibility with existing systems.
    """

    def __init__(self, host: str, port: int, mode: str = 'router', prefer_zmq: bool = True):
        """
        Initialize hybrid communicator.

        Args:
            host: Host address
            port: Port number
            mode: 'router' (server) or 'dealer' (client)
            prefer_zmq: Try ZMQ first before falling back to RPC
        """
        self.host = host
        self.port = port
        self.mode = mode
        self.backend = None

        if prefer_zmq:
            try:
                # Try ZeroMQ first
                from dist_llm_train.communication.zmq_comm import ZMQCommunicator
                self.backend = ZMQCommunicator(host, port, mode)
                logger.info("Using ZeroMQ communicator")
            except Exception as e:
                logger.warning(f"ZeroMQ unavailable ({e}), falling back to XML-RPC")
                self._init_rpc()
        else:
            self._init_rpc()

    def _init_rpc(self):
        """Initialize XML-RPC backend."""
        from dist_llm_train.communication.rpc import RPCCommunicator
        self.backend = RPCCommunicator(self.host, self.port)
        logger.info("Using XML-RPC communicator")

    def register_function(self, func: Callable, name: str = None):
        """Register function for RPC calls."""
        self.backend.register_function(func, name)

    def send(self, address: str, message: Dict[str, Any], timeout: int = 30000) -> Optional[Any]:
        """Send message and wait for response."""
        return self.backend.send(address, message, timeout)

    def start(self):
        """Start the communicator."""
        self.backend.start()

    def stop(self):
        """Stop the communicator."""
        self.backend.stop()
