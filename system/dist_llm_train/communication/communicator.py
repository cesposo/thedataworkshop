from abc import ABC, abstractmethod

class Communicator(ABC):
    """
    An abstract base class for communication between nodes in the distributed system.
    """

    @abstractmethod
    def send(self, destination, message):
        """
        Sends a message to a destination.

        Args:
            destination: The address or identifier of the recipient.
            message: The message to send.
        """
        pass

    @abstractmethod
    def receive(self):
        """
        Receives a message.
        """
        pass

    @abstractmethod
    def broadcast(self, message):
        """
        Broadcasts a message to all nodes.

        Args:
            message: The message to broadcast.
        """
        pass
