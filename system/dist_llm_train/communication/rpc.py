from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import ServerProxy
import threading

from .communicator import Communicator

class RPCCommunicator(Communicator):
    """
    An implementation of the Communicator that uses XML-RPC for communication.
    """

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = SimpleXMLRPCServer((self.host, self.port), allow_none=True)
        if self.port == 0:
            self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True

    def start_server(self):
        """Starts the RPC server in a separate thread."""
        print(f"Starting RPC server on {self.host}:{self.port}")
        self.server_thread.start()

    def stop_server(self):
        """Stops the RPC server."""
        print(f"Stopping RPC server on {self.host}:{self.port}")
        self.server.shutdown()
        # Ensure the underlying socket is closed to avoid ResourceWarning
        try:
            self.server.server_close()
        except Exception:
            pass
        self.server_thread.join()

    def register_function(self, function, name):
        """Registers a function with the RPC server."""
        self.server.register_function(function, name)

    def send(self, destination, message):
        """
        Sends a message to a destination using an RPC call.

        This is a simplified approach where the message is the method name
        and the destination is the server address.
        """
        # This is a placeholder implementation. A real implementation would need
        # a more robust way to handle messages and destinations.
        with ServerProxy(destination) as proxy:
            method = getattr(proxy, message['method'])
            return method(*message['params'])

    def receive(self):
        """Not implemented for this RPC communicator."""
        # TODO: Implement a proper receive mechanism if needed.
        pass

    def broadcast(self, message):
        """Not implemented for this RPC communicator."""
        # TODO: Implement broadcast functionality.
        pass
"""XML-RPC communication layer.

Provides a thin wrapper around `SimpleXMLRPCServer` for the controller and a
`ServerProxy` client for sending messages. Using `port=0` binds a free port and
the actual port is made available via `self.server.server_address[1]`.
"""
