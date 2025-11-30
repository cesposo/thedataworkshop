"""
Performance prediction models for learning node preferences.

This module implements data-driven preference construction, treating nodes
as input-output systems where performance can be predicted from features.
"""

import numpy as np
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("dist_llm_train.scheduler.preference_learning")

# Try to import sklearn, but don't fail if not available
try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. PerformancePredictor will use simple baseline.")


@dataclass
class TrainingRun:
    """Historical record of a task execution."""
    task_id: str
    worker_id: str
    task_features: Dict[str, float]
    worker_features: Dict[str, float]
    completion_time_s: float
    throughput_tokens_per_sec: Optional[float] = None
    success: bool = True
    timestamp: float = 0.0


class PerformancePredictor:
    """
    Learns to predict task performance metrics from node/task features.

    This implements the "nodes as input-output systems" concept from the paper,
    where we learn an embedding that characterizes how nodes behave under load.
    """

    def __init__(self, model_type: str = 'linear', random_state: int = 42):
        """
        Initialize performance predictor.

        Args:
            model_type: One of 'linear', 'ridge', 'random_forest', or 'baseline'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_fitted = False

        if not SKLEARN_AVAILABLE and model_type != 'baseline':
            logger.warning(f"scikit-learn not available, falling back to baseline model")
            self.model_type = 'baseline'

        self._init_model()

    def _init_model(self):
        """Initialize the prediction model based on type."""
        if self.model_type == 'baseline':
            # Simple baseline: use heuristic scoring
            self.model = None
            self.scaler = None
        elif SKLEARN_AVAILABLE:
            if self.model_type == 'linear':
                self.model = LinearRegression()
            elif self.model_type == 'ridge':
                self.model = Ridge(alpha=1.0, random_state=self.random_state)
            elif self.model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=self.random_state
                )
            else:
                logger.warning(f"Unknown model type '{self.model_type}', using linear")
                self.model = LinearRegression()

            self.scaler = StandardScaler()

    def extract_features(self, task_features: Dict, worker_features: Dict) -> np.ndarray:
        """
        Extract feature vector from task-worker pair.

        Args:
            task_features: Dictionary of task attributes
            worker_features: Dictionary of worker attributes

        Returns:
            Feature vector as numpy array
        """
        features = []

        # Task features
        features.append(task_features.get('model_shard_size', 0.0))
        features.append(task_features.get('data_size', 0.0))
        features.append(task_features.get('required_flops', 0.0))
        features.append(task_features.get('total_memory_req', 0.0))
        features.append(task_features.get('priority', 0.0))

        # Worker features
        features.append(worker_features.get('flops_per_second', 0.0))
        features.append(worker_features.get('memory', 0.0))
        features.append(worker_features.get('network_bandwidth', 0.0))
        features.append(worker_features.get('gpu_count', 0.0))

        # Interaction features (ratios)
        worker_memory = worker_features.get('memory', 1.0)
        task_memory = task_features.get('total_memory_req', 0.0)
        memory_headroom_ratio = max(0.0, (worker_memory - task_memory) / worker_memory) if worker_memory > 0 else 0.0
        features.append(memory_headroom_ratio)

        worker_flops = worker_features.get('flops_per_second', 1.0)
        task_flops = task_features.get('required_flops', 0.0)
        compute_ratio = task_flops / worker_flops if worker_flops > 0 else 0.0
        features.append(compute_ratio)

        # Telemetry features (if available)
        features.append(worker_features.get('avg_tokens_per_sec', 0.0))
        features.append(worker_features.get('avg_step_time_s', 0.0))

        return np.array(features)

    def fit(self, training_data: List[TrainingRun]) -> Dict[str, float]:
        """
        Train predictor from historical runs.

        Args:
            training_data: List of TrainingRun records

        Returns:
            Dictionary of training metrics (r2_score, mse, etc.)
        """
        if len(training_data) == 0:
            logger.warning("No training data provided")
            return {'r2': 0.0, 'mse': 0.0, 'samples': 0}

        # Extract features and targets
        X = []
        y = []

        for run in training_data:
            if not run.success:
                continue  # Skip failed runs

            features = self.extract_features(run.task_features, run.worker_features)
            X.append(features)

            # Target: use throughput if available, otherwise inverse of completion time
            if run.throughput_tokens_per_sec is not None and run.throughput_tokens_per_sec > 0:
                y.append(run.throughput_tokens_per_sec)
            else:
                # Use inverse completion time as proxy for throughput
                y.append(1.0 / max(run.completion_time_s, 0.1))

        if len(X) == 0:
            logger.warning("No successful runs in training data")
            return {'r2': 0.0, 'mse': 0.0, 'samples': 0}

        X = np.array(X)
        y = np.array(y)

        # Train model
        if self.model_type == 'baseline':
            # Baseline doesn't need training
            self.is_fitted = True
            return {'r2': 0.0, 'mse': 0.0, 'samples': len(X), 'model': 'baseline'}

        # Split for validation
        if len(X) < 10:
            # Too few samples, use all for training
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Fit model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        logger.info(f"Trained {self.model_type} model: R²={r2:.3f}, MSE={mse:.3f}, samples={len(X)}")

        return {
            'r2': float(r2),
            'mse': float(mse),
            'samples': len(X),
            'model': self.model_type
        }

    def predict_throughput(self, task_features: Dict, worker_features: Dict) -> float:
        """
        Predict tokens/sec for task-worker assignment.

        Args:
            task_features: Dictionary of task attributes
            worker_features: Dictionary of worker attributes

        Returns:
            Predicted throughput (tokens/sec or proxy score)
        """
        if not self.is_fitted:
            logger.warning("Predictor not fitted, using baseline heuristic")
            return self._baseline_score(task_features, worker_features)

        if self.model_type == 'baseline':
            return self._baseline_score(task_features, worker_features)

        # Extract and scale features
        features = self.extract_features(task_features, worker_features)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        return float(max(0.0, prediction))  # Ensure non-negative

    def predict_completion_time(self, task_features: Dict, worker_features: Dict) -> float:
        """
        Predict time to complete task on worker.

        Args:
            task_features: Dictionary of task attributes
            worker_features: Dictionary of worker attributes

        Returns:
            Predicted completion time in seconds
        """
        throughput = self.predict_throughput(task_features, worker_features)
        if throughput > 0:
            # Inverse of throughput as rough completion time estimate
            return 1.0 / throughput
        else:
            return 1000.0  # Large penalty for zero throughput

    def _baseline_score(self, task_features: Dict, worker_features: Dict) -> float:
        """
        Simple heuristic score when no model is trained.

        This is based on capability matching (FLOPS, memory headroom, bandwidth).
        """
        # Check memory fit
        worker_memory = worker_features.get('memory', 0.0)
        task_memory = task_features.get('total_memory_req', 0.0)

        if worker_memory < task_memory:
            return 0.0  # Can't fit

        # Compute capability score
        flops = worker_features.get('flops_per_second', 0.0)
        bandwidth = worker_features.get('network_bandwidth', 0.0)
        headroom = max(0.0, worker_memory - task_memory)

        score = flops * 1.0 + bandwidth * 0.1 + headroom * 0.01
        return score

    def save(self, path: str) -> bool:
        """
        Save trained model to file.

        Args:
            path: File path to save to

        Returns:
            True if successful
        """
        try:
            state = {
                'model_type': self.model_type,
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names
            }
            with open(path, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"Saved model to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    @staticmethod
    def load(path: str) -> 'PerformancePredictor':
        """
        Load trained model from file.

        Args:
            path: File path to load from

        Returns:
            Loaded PerformancePredictor instance
        """
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)

            predictor = PerformancePredictor(model_type=state['model_type'])
            predictor.model = state['model']
            predictor.scaler = state['scaler']
            predictor.is_fitted = state['is_fitted']
            predictor.feature_names = state.get('feature_names', [])

            logger.info(f"Loaded model from {path}")
            return predictor
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
