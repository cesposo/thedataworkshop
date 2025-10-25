from abc import ABC, abstractmethod
from typing import List, Dict

from dist_llm_train.task.training_task import TrainingTask

class BaseScheduler(ABC):
    """
    An abstract base class for scheduling algorithms.
    """

    @abstractmethod
    def __init__(self, tasks: List[TrainingTask], workers: List[Dict]):
        pass

    @abstractmethod
    def schedule(self) -> Dict[str, str]:
        """
        Performs the scheduling algorithm.

        Returns:
            A dictionary representing the stable matching, where keys are
            worker IDs and values are the assigned task IDs.
        """
        pass
