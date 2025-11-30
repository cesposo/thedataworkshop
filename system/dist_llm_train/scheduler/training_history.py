"""
Training history storage and retrieval.

This module persists historical training run data for use in
preference learning and performance prediction.
"""

import sqlite3
import json
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from .preference_learning import TrainingRun

logger = logging.getLogger("dist_llm_train.scheduler.training_history")


class TrainingHistoryStore:
    """
    Persists and retrieves training run records using SQLite.

    This implements the data collection infrastructure needed for
    the "nodes as input-output systems" learning approach.
    """

    def __init__(self, db_path: str = 'training_history.db'):
        """
        Initialize training history store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            logger.info(f"Connected to training history database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _create_tables(self):
        """Create database schema if it doesn't exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            task_features TEXT NOT NULL,
            worker_features TEXT NOT NULL,
            completion_time_s REAL NOT NULL,
            throughput_tokens_per_sec REAL,
            success INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_task_id ON training_runs(task_id);
        CREATE INDEX IF NOT EXISTS idx_worker_id ON training_runs(worker_id);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON training_runs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_success ON training_runs(success);
        """

        try:
            self.conn.executescript(schema)
            self.conn.commit()
            logger.debug("Training runs table schema created/verified")
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise

    def record_run(self, run: TrainingRun) -> bool:
        """
        Store a completed training run.

        Args:
            run: TrainingRun record to store

        Returns:
            True if successful
        """
        try:
            query = """
            INSERT INTO training_runs
            (task_id, worker_id, task_features, worker_features,
             completion_time_s, throughput_tokens_per_sec, success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """

            self.conn.execute(query, (
                run.task_id,
                run.worker_id,
                json.dumps(run.task_features),
                json.dumps(run.worker_features),
                run.completion_time_s,
                run.throughput_tokens_per_sec,
                1 if run.success else 0,
                run.timestamp
            ))
            self.conn.commit()

            logger.debug(f"Recorded training run: task={run.task_id}, worker={run.worker_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to record training run: {e}")
            return False

    def get_recent_runs(self, limit: int = 1000, success_only: bool = True) -> List[TrainingRun]:
        """
        Retrieve recent training runs.

        Args:
            limit: Maximum number of runs to retrieve
            success_only: If True, only return successful runs

        Returns:
            List of TrainingRun records
        """
        try:
            query = """
            SELECT * FROM training_runs
            WHERE success = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """

            success_filter = 1 if success_only else -1  # -1 gets all
            if not success_only:
                query = """
                SELECT * FROM training_runs
                ORDER BY timestamp DESC
                LIMIT ?
                """
                cursor = self.conn.execute(query, (limit,))
            else:
                cursor = self.conn.execute(query, (success_filter, limit))

            rows = cursor.fetchall()

            runs = []
            for row in rows:
                run = TrainingRun(
                    task_id=row['task_id'],
                    worker_id=row['worker_id'],
                    task_features=json.loads(row['task_features']),
                    worker_features=json.loads(row['worker_features']),
                    completion_time_s=row['completion_time_s'],
                    throughput_tokens_per_sec=row['throughput_tokens_per_sec'],
                    success=bool(row['success']),
                    timestamp=row['timestamp']
                )
                runs.append(run)

            logger.debug(f"Retrieved {len(runs)} training runs")
            return runs

        except Exception as e:
            logger.error(f"Failed to retrieve training runs: {e}")
            return []

    def get_runs_by_worker(self, worker_id: str, limit: int = 100) -> List[TrainingRun]:
        """
        Get training runs for a specific worker.

        Args:
            worker_id: Worker ID to filter by
            limit: Maximum number of runs

        Returns:
            List of TrainingRun records
        """
        try:
            query = """
            SELECT * FROM training_runs
            WHERE worker_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """

            cursor = self.conn.execute(query, (worker_id, limit))
            rows = cursor.fetchall()

            runs = []
            for row in rows:
                run = TrainingRun(
                    task_id=row['task_id'],
                    worker_id=row['worker_id'],
                    task_features=json.loads(row['task_features']),
                    worker_features=json.loads(row['worker_features']),
                    completion_time_s=row['completion_time_s'],
                    throughput_tokens_per_sec=row['throughput_tokens_per_sec'],
                    success=bool(row['success']),
                    timestamp=row['timestamp']
                )
                runs.append(run)

            return runs

        except Exception as e:
            logger.error(f"Failed to retrieve runs for worker {worker_id}: {e}")
            return []

    def get_runs_in_time_range(self,
                                start_time: float,
                                end_time: float,
                                limit: int = 1000) -> List[TrainingRun]:
        """
        Get training runs within a time range.

        Args:
            start_time: Start timestamp (Unix time)
            end_time: End timestamp (Unix time)
            limit: Maximum number of runs

        Returns:
            List of TrainingRun records
        """
        try:
            query = """
            SELECT * FROM training_runs
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """

            cursor = self.conn.execute(query, (start_time, end_time, limit))
            rows = cursor.fetchall()

            runs = []
            for row in rows:
                run = TrainingRun(
                    task_id=row['task_id'],
                    worker_id=row['worker_id'],
                    task_features=json.loads(row['task_features']),
                    worker_features=json.loads(row['worker_features']),
                    completion_time_s=row['completion_time_s'],
                    throughput_tokens_per_sec=row['throughput_tokens_per_sec'],
                    success=bool(row['success']),
                    timestamp=row['timestamp']
                )
                runs.append(run)

            return runs

        except Exception as e:
            logger.error(f"Failed to retrieve runs in time range: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics about stored runs.

        Returns:
            Dictionary of statistics
        """
        try:
            query = """
            SELECT
                COUNT(*) as total_runs,
                SUM(success) as successful_runs,
                AVG(completion_time_s) as avg_completion_time,
                AVG(throughput_tokens_per_sec) as avg_throughput,
                MIN(timestamp) as first_run_time,
                MAX(timestamp) as last_run_time
            FROM training_runs
            """

            cursor = self.conn.execute(query)
            row = cursor.fetchone()

            stats = {
                'total_runs': row['total_runs'] or 0,
                'successful_runs': row['successful_runs'] or 0,
                'failed_runs': (row['total_runs'] or 0) - (row['successful_runs'] or 0),
                'avg_completion_time_s': row['avg_completion_time'] or 0.0,
                'avg_throughput': row['avg_throughput'] or 0.0,
                'first_run_time': row['first_run_time'] or 0.0,
                'last_run_time': row['last_run_time'] or 0.0
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def cleanup_old_runs(self, days: int = 30) -> int:
        """
        Delete runs older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of runs deleted
        """
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)

            query = "DELETE FROM training_runs WHERE timestamp < ?"
            cursor = self.conn.execute(query, (cutoff_time,))
            self.conn.commit()

            deleted = cursor.rowcount
            logger.info(f"Deleted {deleted} runs older than {days} days")
            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup old runs: {e}")
            return 0

    def export_to_csv(self, output_path: str, limit: int = 10000) -> bool:
        """
        Export training runs to CSV file.

        Args:
            output_path: Path to output CSV file
            limit: Maximum number of runs to export

        Returns:
            True if successful
        """
        try:
            import csv

            runs = self.get_recent_runs(limit=limit, success_only=False)

            with open(output_path, 'w', newline='') as csvfile:
                if not runs:
                    logger.warning("No runs to export")
                    return False

                # Flatten the first run to get all possible field names
                fieldnames = ['task_id', 'worker_id', 'completion_time_s',
                            'throughput_tokens_per_sec', 'success', 'timestamp']

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for run in runs:
                    row = {
                        'task_id': run.task_id,
                        'worker_id': run.worker_id,
                        'completion_time_s': run.completion_time_s,
                        'throughput_tokens_per_sec': run.throughput_tokens_per_sec,
                        'success': run.success,
                        'timestamp': run.timestamp
                    }
                    writer.writerow(row)

            logger.info(f"Exported {len(runs)} runs to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed training history database connection")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
