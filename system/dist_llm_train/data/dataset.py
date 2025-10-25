"""Dataset implementations for distributed training."""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, List, Tuple


class SimpleTextDataset(Dataset):
    """
    A simple text dataset for language modeling.

    This creates sequences from a list of token IDs suitable for
    next-token prediction training.
    """

    def __init__(
        self,
        token_ids: List[int],
        seq_length: int = 128,
        vocab_size: Optional[int] = None
    ):
        """
        Initialize the dataset.

        Args:
            token_ids: List of token IDs
            seq_length: Length of each sequence
            vocab_size: Size of vocabulary (for validation)
        """
        self.token_ids = token_ids
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Create sequences
        self.sequences = []
        for i in range(0, len(token_ids) - seq_length - 1, seq_length):
            input_seq = token_ids[i:i + seq_length]
            target_seq = token_ids[i + 1:i + seq_length + 1]

            if len(input_seq) == seq_length and len(target_seq) == seq_length:
                self.sequences.append((input_seq, target_seq))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training example.

        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        input_seq, target_seq = self.sequences[idx]
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )


def create_dummy_dataset(
    num_samples: int = 1000,
    seq_length: int = 128,
    vocab_size: int = 10000
) -> SimpleTextDataset:
    """
    Create a dummy dataset for testing.

    This generates random token sequences that follow a simple pattern
    to make convergence testing easier.

    Args:
        num_samples: Number of samples to generate
        seq_length: Length of each sequence
        vocab_size: Size of vocabulary

    Returns:
        SimpleTextDataset instance
    """
    # Generate random token IDs with some patterns
    # Use a smaller effective vocab to make learning easier
    effective_vocab = min(vocab_size, 100)

    # Create sequences with some repetitive patterns for easier learning
    total_length = num_samples * (seq_length + 1)
    token_ids = []

    # Create patterns like: 1, 2, 3, 4, 1, 2, 3, 4, ...
    pattern_length = 10
    for i in range(total_length):
        token_ids.append((i % pattern_length) % effective_vocab)

    # Add some randomness (20% of tokens)
    num_random = int(total_length * 0.2)
    random_indices = np.random.choice(total_length, num_random, replace=False)
    for idx in random_indices:
        token_ids[idx] = np.random.randint(0, effective_vocab)

    return SimpleTextDataset(
        token_ids=token_ids,
        seq_length=seq_length,
        vocab_size=vocab_size
    )


class DistributedDatasetSplit:
    """
    Splits a dataset for distributed training across multiple workers.
    """

    def __init__(self, dataset: Dataset, num_workers: int, worker_id: int):
        """
        Initialize a dataset split for a specific worker.

        Args:
            dataset: The full dataset
            num_workers: Total number of workers
            worker_id: ID of this worker (0-indexed)
        """
        self.dataset = dataset
        self.num_workers = num_workers
        self.worker_id = worker_id

        # Calculate split indices
        total_size = len(dataset)
        per_worker = total_size // num_workers
        self.start_idx = worker_id * per_worker
        self.end_idx = self.start_idx + per_worker if worker_id < num_workers - 1 else total_size

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError("Index out of range")
        return self.dataset[self.start_idx + idx]
