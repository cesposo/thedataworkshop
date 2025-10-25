"""Tests for ML training functionality."""

import unittest
import torch

from dist_llm_train.models.model_loader import ModelLoader
from dist_llm_train.models.model_shard import SimpleLSTMModel
from dist_llm_train.data.dataset import create_dummy_dataset
from dist_llm_train.data.data_loader import DistributedDataLoader
from dist_llm_train.task.training_task import TrainingTask


class TestMLTraining(unittest.TestCase):

    def test_model_creation(self):
        """Test that we can create a simple LSTM model."""
        model = ModelLoader.create_simple_lstm(
            vocab_size=1000,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2
        )

        self.assertIsInstance(model, SimpleLSTMModel)
        self.assertEqual(model.vocab_size, 1000)
        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 128)

    def test_dataset_creation(self):
        """Test that we can create a dataset."""
        dataset = create_dummy_dataset(
            num_samples=100,
            seq_length=32,
            vocab_size=1000
        )

        self.assertGreater(len(dataset), 0)

        # Test getting an item
        inputs, targets = dataset[0]
        self.assertEqual(inputs.shape[0], 32)
        self.assertEqual(targets.shape[0], 32)

    def test_data_loader(self):
        """Test the distributed data loader."""
        dataset = create_dummy_dataset(
            num_samples=100,
            seq_length=32,
            vocab_size=1000
        )

        data_loader = DistributedDataLoader(
            dataset=dataset,
            batch_size=8,
            num_workers=2,
            worker_id=0
        )

        # Test iteration
        batch_count = 0
        for inputs, targets in data_loader:
            batch_count += 1
            self.assertEqual(inputs.shape[0], 8)  # batch size
            self.assertEqual(inputs.shape[1], 32)  # seq length
            if batch_count >= 2:  # Just test first 2 batches
                break

        self.assertGreater(batch_count, 0)

    def test_model_forward_pass(self):
        """Test that we can do a forward pass through the model."""
        model = ModelLoader.create_simple_lstm(
            vocab_size=100,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )

        device = torch.device('cpu')
        model.to(device)
        model.eval()

        # Create dummy input
        batch_size = 4
        seq_length = 16
        inputs = torch.randint(0, 100, (batch_size, seq_length))

        # Forward pass
        with torch.no_grad():
            hidden = model.init_hidden(batch_size, device)
            logits, _ = model(inputs, hidden)

        # Check output shape
        self.assertEqual(logits.shape, (batch_size, seq_length, 100))

    def test_training_task_with_configs(self):
        """Test creating a training task with model and training configs."""
        model_config = ModelLoader.get_model_config('tiny-lstm')
        training_config = {
            'learning_rate': 0.001,
            'batch_size': 8,
            'num_epochs': 1
        }

        task = TrainingTask(
            task_id="test-task",
            model_name="tiny-lstm",
            model_layer=0,
            model_shard_size=0.5,
            data_size=0.1,
            required_flops=1000,
            model_config=model_config,
            training_config=training_config
        )

        self.assertEqual(task.id, "test-task")
        self.assertEqual(task.model_config['type'], 'lstm')
        self.assertEqual(task.training_config['learning_rate'], 0.001)

    def test_single_training_step(self):
        """Test a single training step."""
        # Create model
        model = ModelLoader.create_simple_lstm(
            vocab_size=100,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )

        device = torch.device('cpu')
        model.to(device)
        model.train()

        # Create optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Create batch
        batch_size = 4
        seq_length = 16
        inputs = torch.randint(0, 100, (batch_size, seq_length))
        targets = torch.randint(0, 100, (batch_size, seq_length))

        # Training step
        optimizer.zero_grad()
        hidden = model.init_hidden(batch_size, device)
        logits, _ = model(inputs, hidden)

        loss = criterion(logits.view(-1, 100), targets.view(-1))
        loss.backward()
        optimizer.step()

        # Check that loss is a valid number
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)


if __name__ == '__main__':
    unittest.main()
