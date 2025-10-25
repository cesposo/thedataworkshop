"""Distributed data loading utilities."""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional


class DistributedDataLoader:
    """
    A data loader that handles distributed data loading across workers.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 1,
        worker_id: int = 0,
        shuffle: bool = True,
        drop_last: bool = True
    ):
        """
        Initialize a distributed data loader.

        Args:
            dataset: The dataset to load from
            batch_size: Batch size per worker
            num_workers: Total number of workers in distributed setup
            worker_id: ID of this worker
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.worker_id = worker_id

        # Calculate data split for this worker
        total_size = len(dataset)
        per_worker = total_size // num_workers

        # Each worker gets a contiguous chunk of data
        self.start_idx = worker_id * per_worker
        self.end_idx = self.start_idx + per_worker if worker_id < num_workers - 1 else total_size

        # Create indices for this worker's portion
        indices = list(range(self.start_idx, self.end_idx))

        # Create sampler
        if shuffle:
            sampler = torch.utils.data.SubsetRandomSampler(indices)
        else:
            sampler = torch.utils.data.SequentialSampler(
                torch.utils.data.Subset(dataset, indices)
            )

        # Create DataLoader
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=0,  # Single-threaded for simplicity
            pin_memory=torch.cuda.is_available()
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def get_num_samples(self) -> int:
        """Returns the number of samples this worker is responsible for."""
        return self.end_idx - self.start_idx
