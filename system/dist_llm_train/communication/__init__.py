"""Communication module - supports both XML-RPC and ZeroMQ."""

from dist_llm_train.communication.rpc import RPCCommunicator

try:
    from dist_llm_train.communication.zmq_comm import ZMQCommunicator, HybridCommunicator
    __all__ = ['RPCCommunicator', 'ZMQCommunicator', 'HybridCommunicator']
except ImportError:
    # ZeroMQ dependencies not available
    __all__ = ['RPCCommunicator']
