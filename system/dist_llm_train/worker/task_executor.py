import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any
from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.models.model_loader import ModelLoader
from dist_llm_train.data.dataset import create_dummy_dataset
from dist_llm_train.data.data_loader import DistributedDataLoader


class TaskExecutor:
    """
    Executes a training task on a worker node.
    """

    def __init__(self, worker_id: str, communicator, controller_address: str):
        self.worker_id = worker_id
        self.communicator = communicator
        self.controller_address = controller_address
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.CrossEntropyLoss()

    def execute_task(self, task: TrainingTask):
        """
        Executes a given training task with real PyTorch training.

        Args:
            task: The training task to execute
        """
        print(f"[Executor {self.worker_id}] Starting execution of task {task.id}")
        print(f"[Executor {self.worker_id}] Device: {self.device}")

        try:
            # Determine if this is a real ML task or simulation
            if hasattr(task, 'model_config') and task.model_config:
                self._execute_ml_task(task)
            else:
                # Fallback to simulation for backward compatibility
                self._execute_simulated_task(task)

            print(f"[Executor {self.worker_id}] Task {task.id} completed successfully.")

        except Exception as e:
            print(f"[Executor {self.worker_id}] Task {task.id} failed with error: {e}")
            import traceback
            traceback.print_exc()

        # Notify the controller that the task is complete
        try:
            self.communicator.send(self.controller_address, {
                'method': 'task_completed',
                'params': [self.worker_id, task.id]
            })
        except Exception as e:
            print(f"[Executor {self.worker_id}] Failed to send task completion notification: {e}")

    def _execute_simulated_task(self, task: TrainingTask):
        """
        Executes a simulated training task (for backward compatibility).
        """
        print(f"[Executor {self.worker_id}] Running in SIMULATION mode")
        for i in range(5):
            print(f"[Executor {self.worker_id}] Task {task.id}: Training step {i+1}/5")
            time.sleep(1)

    def _execute_ml_task(self, task: TrainingTask):
        """
        Executes a real ML training task with PyTorch.
        """
        print(f"[Executor {self.worker_id}] Running in ML TRAINING mode")

        # Load model
        model_config = task.model_config
        self.model = ModelLoader.load_model_from_config(model_config)
        self.model.to(self.device)

        # Setup optimizer
        lr = task.training_config.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Load data
        num_samples = task.training_config.get('num_samples', 100)
        seq_length = task.training_config.get('seq_length', 64)
        vocab_size = model_config.get('vocab_size', 10000)
        batch_size = task.training_config.get('batch_size', 8)
        num_epochs = task.training_config.get('num_epochs', 1)

        dataset = create_dummy_dataset(
            num_samples=num_samples,
            seq_length=seq_length,
            vocab_size=vocab_size
        )

        # Create data loader (single worker for now)
        data_loader = DistributedDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            worker_id=0,
            shuffle=True
        )

        # Training loop
        print(f"[Executor {self.worker_id}] Starting training: {num_epochs} epochs, {len(data_loader)} batches/epoch")

        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch_idx, (inputs, targets) in enumerate(data_loader):
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()

                if hasattr(self.model, 'init_hidden'):
                    # LSTM model
                    hidden = self.model.init_hidden(inputs.size(0), self.device)
                    logits, _ = self.model(inputs, hidden)
                else:
                    # Other models
                    logits = self.model(inputs)

                # Reshape for loss calculation
                # logits: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
                # targets: (batch, seq_len) -> (batch * seq_len)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Track loss
                batch_loss = loss.item()
                epoch_loss += batch_loss
                total_loss += batch_loss
                epoch_batches += 1
                num_batches += 1

                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    print(f"[Executor {self.worker_id}] Epoch {epoch+1}/{num_epochs}, "
                          f"Batch {batch_idx+1}/{len(data_loader)}, Loss: {batch_loss:.4f}")

            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            print(f"[Executor {self.worker_id}] Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.4f}")

        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"[Executor {self.worker_id}] Training completed. Avg Loss: {avg_total_loss:.4f}")

        # Store final metrics
        task.metrics = {
            'final_loss': avg_total_loss,
            'num_batches': num_batches,
            'num_epochs': num_epochs
        }

    def get_model_state(self) -> Optional[Dict[str, Any]]:
        """
        Returns the current model state for checkpointing or synchronization.
        """
        if self.model is None:
            return None

        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }
