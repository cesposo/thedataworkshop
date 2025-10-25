"""Model loading and sharding utilities."""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
from .model_shard import ModelShard, SimpleTransformerShard, SimpleLSTMModel


class ModelLoader:
    """
    Handles loading and sharding of models for distributed training.
    """

    @staticmethod
    def create_simple_lstm(
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ) -> SimpleLSTMModel:
        """
        Create a simple LSTM model for testing.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate

        Returns:
            SimpleLSTMModel instance
        """
        return SimpleLSTMModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    @staticmethod
    def create_transformer_shard(
        shard_id: int,
        total_shards: int,
        vocab_size: int = 10000,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ) -> SimpleTransformerShard:
        """
        Create a transformer shard for distributed training.

        Args:
            shard_id: ID of this shard
            total_shards: Total number of shards
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension
            num_layers: Number of layers in this shard
            num_heads: Number of attention heads
            dropout: Dropout rate

        Returns:
            SimpleTransformerShard instance
        """
        return SimpleTransformerShard(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            shard_id=shard_id,
            total_shards=total_shards,
            dropout=dropout
        )

    @staticmethod
    def shard_model_by_layers(
        model: nn.Module,
        num_shards: int
    ) -> List[ModelShard]:
        """
        Shard a model by dividing its layers across shards.

        Args:
            model: The model to shard
            num_shards: Number of shards to create

        Returns:
            List of ModelShard instances
        """
        # Get all layers from the model
        layers = []
        for module in model.children():
            if isinstance(module, nn.ModuleList):
                layers.extend(module)
            else:
                layers.append(module)

        # Divide layers across shards
        layers_per_shard = len(layers) // num_shards
        remainder = len(layers) % num_shards

        shards = []
        start_idx = 0

        for shard_id in range(num_shards):
            # Distribute remainder layers to first shards
            num_layers = layers_per_shard + (1 if shard_id < remainder else 0)
            end_idx = start_idx + num_layers

            shard_layers = nn.ModuleList(layers[start_idx:end_idx])
            shard = ModelShard(
                layers=shard_layers,
                shard_id=shard_id,
                total_shards=num_shards
            )
            shards.append(shard)

            start_idx = end_idx

        return shards

    @staticmethod
    def get_model_config(model_name: str) -> Dict[str, Any]:
        """
        Get configuration for predefined models.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model configuration
        """
        configs = {
            'tiny-lstm': {
                'type': 'lstm',
                'vocab_size': 5000,
                'embedding_dim': 64,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.2
            },
            'small-lstm': {
                'type': 'lstm',
                'vocab_size': 10000,
                'embedding_dim': 128,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.2
            },
            'tiny-transformer': {
                'type': 'transformer',
                'vocab_size': 5000,
                'hidden_size': 128,
                'num_layers': 2,
                'num_heads': 4,
                'dropout': 0.1
            },
            'small-transformer': {
                'type': 'transformer',
                'vocab_size': 10000,
                'hidden_size': 256,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': 0.1
            }
        }

        return configs.get(model_name, configs['tiny-lstm'])

    @staticmethod
    def load_model_from_config(config: Dict[str, Any]) -> nn.Module:
        """
        Load a model from configuration dictionary.

        Args:
            config: Model configuration

        Returns:
            nn.Module instance
        """
        model_type = config.get('type', 'lstm')

        if model_type == 'lstm':
            return ModelLoader.create_simple_lstm(
                vocab_size=config.get('vocab_size', 10000),
                embedding_dim=config.get('embedding_dim', 128),
                hidden_dim=config.get('hidden_dim', 256),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.2)
            )
        elif model_type == 'transformer':
            return ModelLoader.create_transformer_shard(
                shard_id=0,
                total_shards=1,
                vocab_size=config.get('vocab_size', 10000),
                hidden_size=config.get('hidden_size', 256),
                num_layers=config.get('num_layers', 4),
                num_heads=config.get('num_heads', 8),
                dropout=config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
