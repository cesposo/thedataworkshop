"""Gradient synchronization for distributed training."""

from .parameter_server import ParameterServer
from .gradient_aggregator import GradientAggregator

__all__ = ['ParameterServer', 'GradientAggregator']
