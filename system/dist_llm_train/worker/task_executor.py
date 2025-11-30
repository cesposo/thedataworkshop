import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, Callable
from dist_llm_train.task.training_task import TrainingTask
from dist_llm_train.models.model_loader import ModelLoader
from dist_llm_train.data.dataset import create_dummy_dataset
from dist_llm_train.data.data_loader import DistributedDataLoader
from dist_llm_train.scheduler.training_history import TrainingHistoryStore, TrainingRun
from dist_llm_train.compression.compressor import GradientCompressor


class TaskExecutor:
    """
    Executes a training task on a worker node.
    """

    def __init__(self, worker_id: str, communicator, controller_address: str,
                 status_callback: Optional[Callable[[TrainingTask, bool], None]] = None,
                 history_db_path: str = 'training_history.db'):
        self.worker_id = worker_id
        self.communicator = communicator
        self.controller_address = controller_address
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.CrossEntropyLoss()
        self.logger = logging.getLogger(f"dist_llm_train.executor.{worker_id}")
        self._status_callback = status_callback
        self.compressor: Optional[GradientCompressor] = None  # Initialized based on task config

        # Training history for learned preferences
        self.history_store: Optional[TrainingHistoryStore] = None
        try:
            self.history_store = TrainingHistoryStore(history_db_path)
            self.logger.info(f"Training history store initialized at {history_db_path}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize training history store: {e}")

    def execute_task(self, task: TrainingTask):
        """
        Executes a given training task with real PyTorch training.

        Args:
            task: The training task to execute
        """
        self.logger.info(f"Starting execution of task {task.id}")
        self.logger.info(f"Device: {self.device}")
        success = False
        start_time = time.time()

        try:
            # Determine if this is a real ML task or simulation
            if hasattr(task, 'model_config') and task.model_config:
                self._execute_ml_task(task)
            else:
                # Fallback to simulation for backward compatibility
                self._execute_simulated_task(task)

            self.logger.info(f"Task {task.id} completed successfully.")
            success = True
        except Exception as e:
            self.logger.error(f"Task {task.id} failed with error: {e}")
            import traceback
            traceback.print_exc()
            # Notify failure and return
            try:
                self.communicator.send(self.controller_address, {
                    'method': 'task_failed',
                    'params': [self.worker_id, task.id, str(e)]
                })
            except Exception as send_e:
                self.logger.error(f"Failed to send task failure notification: {send_e}")
        else:
            # Notify the controller that the task is complete
            try:
                self.communicator.send(self.controller_address, {
                    'method': 'task_completed',
                    'params': [self.worker_id, task.id]
                })
            except Exception as e:
                self.logger.error(f"Failed to send task completion notification: {e}")
        finally:
            # Record training run to history
            if success:
                completion_time = time.time() - start_time
                self._record_training_run(task, completion_time)

            if self._status_callback:
                try:
                    self._status_callback(task, success)
                except Exception as cb_err:
                    self.logger.warning(f"Status callback failed for task {task.id}: {cb_err}")

    def _execute_simulated_task(self, task: TrainingTask):
        """
        Executes a simulated training task (for backward compatibility).
        """
        self.logger.info("Running in SIMULATION mode")
        start = time.perf_counter()
        for i in range(5):
            self.logger.info(f"Task {task.id}: Training step {i+1}/5")
            time.sleep(1)
            # Send simple telemetry: step time
            try:
                elapsed = time.perf_counter() - start
                self.communicator.send(self.controller_address, {
                    'method': 'report_telemetry',
                    'params': [self.worker_id, {'mode': 'sim', 'step': i + 1, 'elapsed_s': elapsed}]
                })
            except Exception:
                pass

    def _execute_ml_task(self, task: TrainingTask):
        """
        Executes a real ML training task with PyTorch.
        """
        self.logger.info("Running in ML TRAINING mode")

        # Load model
        model_config = task.model_config
        self.model = ModelLoader.load_model_from_config(model_config)
        self.model.to(self.device)

        # Setup optimizer placeholder (updates will happen on ParameterServer)
        lr = task.training_config.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Load data
        num_samples = task.training_config.get('num_samples', 100)
        seq_length = task.training_config.get('seq_length', 64)
        vocab_size = model_config.get('vocab_size', 10000)
        batch_size = task.training_config.get('batch_size', 8)
        num_epochs = task.training_config.get('num_epochs', 1)
        num_sync_workers = int(task.training_config.get('num_workers', 1))
        checkpoint_dir = task.training_config.get('checkpoint_dir', 'checkpoints')

        dataset = create_dummy_dataset(
            num_samples=num_samples,
            seq_length=seq_length,
            vocab_size=vocab_size
        )

        # Create data loader using distributed worker settings if provided
        num_data_workers = int(task.training_config.get('num_workers', 1) or 1)
        num_data_workers = max(1, num_data_workers)
        worker_rank = int(task.training_config.get('worker_rank', 0) or 0)
        if worker_rank < 0:
            worker_rank = 0
        if worker_rank >= num_data_workers:
            worker_rank = num_data_workers - 1
        data_loader = DistributedDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_data_workers,
            worker_id=worker_rank,
            shuffle=True
        )

        # Configure sync and initialize PS params
        try:
            # Configure coordinator for synchronous steps (optionally with window)
            sync_window = int(task.training_config.get('sync_window', num_sync_workers))
            max_wait_s = float(task.training_config.get('sync_max_wait_s', 0)) or None
            params = [num_sync_workers]
            # XML-RPC positional args; add optional ones if provided
            if sync_window is not None:
                params.append(sync_window)
                if max_wait_s is not None:
                    params.append(max_wait_s)
            self.communicator.send(self.controller_address, {
                'method': 'configure_training_sync',
                'params': params
            })
            # Set aggregation rule if provided
            aggr_rule = task.training_config.get('aggregation_rule')
            trim_ratio = float(task.training_config.get('trim_ratio', 0.0))
            compression = task.training_config.get('compression')

            # Initialize gradient compressor
            if compression and compression != 'none':
                compression_ratio = float(task.training_config.get('compression_ratio', 0.01))
                self.compressor = GradientCompressor(method=compression, compression_ratio=compression_ratio)
                self.logger.info(f"Gradient compression enabled: method={compression}, ratio={compression_ratio}")
            else:
                self.compressor = None
                self.logger.info("Gradient compression disabled")

            if aggr_rule or compression:
                self.communicator.send(self.controller_address, {
                    'method': 'ps_set_aggregation',
                    'params': [aggr_rule or 'mean', trim_ratio, compression]
                })
            # Initialize parameter server if empty with current model state
            init_state = {k: v.detach().cpu().tolist() for k, v in self.model.state_dict().items()}
            self.communicator.send(self.controller_address, {
                'method': 'ps_initialize_if_empty',
                'params': [init_state]
            })
            # Pull current parameters
            resp = self.communicator.send(self.controller_address, {
                'method': 'ps_get_parameters',
                'params': []
            })
            if resp and 'parameters' in resp:
                state = {k: torch.tensor(v) for k, v in resp['parameters'].items()}
                self.model.load_state_dict(state, strict=False)
        except Exception as e:
            self.logger.error(f"Sync/PS initialization failed: {e}")

        # Training loop
        self.logger.info(f"Starting training: {num_epochs} epochs, {len(data_loader)} batches/epoch")

        total_loss = 0.0
        num_batches = 0
        worker_step = 0  # Track worker's local step for async training

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch_idx, (inputs, targets) in enumerate(data_loader):
                batch_start = time.perf_counter()
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

                # Collect gradients by parameter name
                grad_tensors = {}
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        grad_tensors[name] = p.grad.detach()

                # Compress gradients if enabled
                if self.compressor is not None:
                    grads, metadata = self.compressor.compress(grad_tensors)
                    # Log compression stats periodically
                    if (batch_idx + 1) % 50 == 0:
                        stats = self.compressor.get_compression_stats(grad_tensors, grads)
                        self.logger.info(
                            f"Gradient compression: {stats['compression_ratio']:.2f}x, "
                            f"bandwidth reduction: {stats['bandwidth_reduction']:.1f}%"
                        )
                else:
                    # No compression - convert to lists for RPC
                    grads = {k: v.cpu().tolist() for k, v in grad_tensors.items()}
                    metadata = None

                # Synchronous step with parameter server
                try:
                    params_to_send = [self.worker_id, grads, lr, worker_step]
                    if metadata is not None:
                        params_to_send.append(metadata)

                    resp = self.communicator.send(self.controller_address, {
                        'method': 'ps_sync_step',
                        'params': params_to_send
                    })
                    if resp and 'parameters' in resp:
                        new_state = {k: torch.tensor(v) for k, v in resp['parameters'].items()}
                        self.model.load_state_dict(new_state, strict=False)
                except Exception as e:
                    self.logger.error(f"Sync step failed: {e}")

                # Track loss
                batch_loss = loss.item()
                epoch_loss += batch_loss
                total_loss += batch_loss
                epoch_batches += 1
                num_batches += 1
                worker_step += 1  # Increment worker step for async training

                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(data_loader)}, Loss: {batch_loss:.4f}"
                    )

                # Telemetry: tokens/sec and step time
                try:
                    step_time = max(1e-6, time.perf_counter() - batch_start)
                    tokens = int(inputs.size(0) * inputs.size(1)) if inputs.dim() >= 2 else int(inputs.size(0))
                    tps = tokens / step_time
                    self.communicator.send(self.controller_address, {
                        'method': 'report_telemetry',
                        'params': [self.worker_id, {
                            'mode': 'ml', 'epoch': epoch + 1, 'batch': batch_idx + 1,
                            'tokens': tokens, 'step_time_s': step_time, 'tokens_per_sec': tps
                        }]
                    })
                except Exception:
                    pass

            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            self.logger.info(f"Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.4f}")

            # Checkpoint
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(checkpoint_dir, f"{task.id}_epoch{epoch+1}.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch + 1
                }, ckpt_path)
            except Exception as e:
                self.logger.warning(f"Checkpoint save failed: {e}")

        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0
        self.logger.info(f"Training completed. Avg Loss: {avg_total_loss:.4f}")

        # Store final metrics
        task.metrics = {
            'final_loss': avg_total_loss,
            'num_batches': num_batches,
            'num_epochs': num_epochs
        }

        # Report metrics to controller
        try:
            self.communicator.send(self.controller_address, {
                'method': 'report_metrics',
                'params': [self.worker_id, task.id, task.metrics]
            })
        except Exception as e:
            self.logger.warning(f"Failed to report metrics: {e}")

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

    def _record_training_run(self, task: TrainingTask, completion_time_s: float):
        """
        Record training run to history store for learned preference modeling.

        Args:
            task: Completed training task
            completion_time_s: Total time to complete the task in seconds
        """
        if not self.history_store:
            return

        try:
            # Extract task features
            task_features = {
                'model_shard_size': task.model_shard_size,
                'data_size': task.data_size,
                'required_flops': task.required_flops,
                'total_memory_req': task.get_total_memory_req(),
                'priority': getattr(task, 'priority', 0)
            }

            # Extract worker features (from self attributes - these are placeholders)
            # In real usage, these would come from the actual worker node
            worker_features = {
                'flops_per_second': 0.0,  # Placeholder - would need worker info
                'memory': 0.0,  # Placeholder
                'network_bandwidth': 0.0,  # Placeholder
                'gpu_count': 1 if torch.cuda.is_available() else 0
            }

            # Calculate throughput if available
            throughput = None
            if task.metrics and 'num_batches' in task.metrics:
                num_batches = task.metrics['num_batches']
                if num_batches > 0 and completion_time_s > 0:
                    # Rough estimate: batches per second
                    throughput = num_batches / completion_time_s

            # Create training run record
            run = TrainingRun(
                task_id=task.id,
                worker_id=self.worker_id,
                task_features=task_features,
                worker_features=worker_features,
                completion_time_s=completion_time_s,
                throughput_tokens_per_sec=throughput,
                success=True,
                timestamp=time.time()
            )

            # Record to database
            self.history_store.record_run(run)
            self.logger.debug(f"Recorded training run for task {task.id} to history store")

        except Exception as e:
            self.logger.warning(f"Failed to record training run: {e}")
