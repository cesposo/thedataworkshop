"""Model shard representation and management."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List


class ModelShard(nn.Module):
    """
    Represents a shard (portion) of a larger model for distributed training.

    This can represent:
    - A subset of transformer layers (pipeline parallelism)
    - A partition of model weights (tensor parallelism)
    - A complete small model (for testing)
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        shard_id: int,
        total_shards: int,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None
    ):
        """
        Initialize a model shard.

        Args:
            layers: The layers that make up this shard
            shard_id: The ID of this shard (0-indexed)
            total_shards: Total number of shards in the model
            input_size: Expected input size (for validation)
            output_size: Expected output size (for validation)
        """
        super().__init__()
        self.layers = layers
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through this model shard.

        Args:
            x: Input tensor

        Returns:
            Output tensor after passing through shard layers
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def get_num_parameters(self) -> int:
        """Returns the total number of parameters in this shard."""
        return sum(p.numel() for p in self.parameters())

    def get_memory_footprint(self) -> float:
        """
        Returns the approximate memory footprint in GB.

        Includes parameters and typical activation memory.
        """
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters())
        # Rough estimate: activations are typically 2-3x parameter memory during training
        total_memory = param_memory * 3
        return total_memory / (1024 ** 3)  # Convert to GB

    def state_dict_serializable(self) -> Dict[str, Any]:
        """
        Returns a serializable representation of the model state.

        This can be sent over RPC for checkpointing or migration.
        """
        return {
            'shard_id': self.shard_id,
            'total_shards': self.total_shards,
            'state_dict': {k: v.cpu().numpy().tolist() for k, v in self.state_dict().items()},
            'num_parameters': self.get_num_parameters()
        }


class SimpleTransformerShard(ModelShard):
    """
    A simple transformer-based model shard for testing.

    This creates a small transformer model suitable for distributed training experiments.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        shard_id: int = 0,
        total_shards: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize a simple transformer shard.

        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers in this shard
            num_heads: Number of attention heads
            shard_id: ID of this shard
            total_shards: Total number of shards
            dropout: Dropout probability
        """
        # Create embedding layer (only for first shard)
        layers = nn.ModuleList()

        if shard_id == 0:
            layers.append(nn.Embedding(vocab_size, hidden_size))
            layers.append(nn.Dropout(dropout))

        # Add transformer layers
        for _ in range(num_layers):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True
                )
            )

        # Add output layer (only for last shard)
        if shard_id == total_shards - 1:
            layers.append(nn.Linear(hidden_size, vocab_size))

        super().__init__(
            layers=layers,
            shard_id=shard_id,
            total_shards=total_shards,
            input_size=hidden_size,
            output_size=vocab_size if shard_id == total_shards - 1 else hidden_size
        )

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class SimpleLSTMModel(nn.Module):
    """
    A simple LSTM model for testing basic distributed training.

    This is easier to work with than transformers for initial testing.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize a simple LSTM language model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None):
        """
        Forward pass through the LSTM model.

        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden: Optional hidden state tuple (h, c)

        Returns:
            Tuple of (logits, hidden_state)
        """
        # Embed input
        embedded = self.dropout(self.embedding(x))

        # LSTM forward
        if hidden is not None:
            output, hidden = self.lstm(embedded, hidden)
        else:
            output, hidden = self.lstm(embedded)

        # Decode to logits
        output = self.dropout(output)
        logits = self.fc(output)

        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h, c)
