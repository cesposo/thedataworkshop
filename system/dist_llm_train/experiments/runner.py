import csv
import glob as _glob
import os
import time
from typing import List, Optional

from dist_llm_train.logging_utils import configure_logging


def run_experiments(configs: List[str], mode: str = 'ml', repeats: int = 1, output_csv: str = 'experiments.csv', profile: str = '', window: int = 0) -> str:
    """Run a set of simulations and record summary metrics to CSV.

    Returns the path to the CSV file.
    """
    configure_logging('INFO')
    rows = []
    for cfg in configs:
        for r in range(repeats):
            start = time.perf_counter()
            # Optional EWMA tuning via window -> alpha mapping: alpha = 2/(N+1)
            prev_alpha = os.environ.get('TELEMETRY_ALPHA')
            if window and window > 0:
                alpha = 2.0 / (window + 1.0)
                os.environ['TELEMETRY_ALPHA'] = f"{alpha}"
            if mode == 'ml':
                from ml_training_simulation import run_ml_training_simulation

                status = run_ml_training_simulation(cfg)
            elif mode == 'basic':
                from simulation import run_simulation

                status = run_simulation(cfg)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            duration = time.perf_counter() - start
            # Restore previous alpha if changed
            if window and window > 0:
                if prev_alpha is None:
                    os.environ.pop('TELEMETRY_ALPHA', None)
                else:
                    os.environ['TELEMETRY_ALPHA'] = prev_alpha
            workers = status.get('workers', {}) if status else {}
            pending = status.get('pending_tasks', []) if status else []
            completed = status.get('completed_tasks', []) if status else []
            telem_roll = status.get('telemetry_rollups', {}) if status else {}
            # Compute mean EWMA metrics across workers
            ewma_tps_vals = [v.get('ewma_tps') for v in telem_roll.values() if v.get('ewma_tps') is not None]
            ewma_step_vals = [v.get('ewma_step') for v in telem_roll.values() if v.get('ewma_step') is not None]
            mean_ewma_tps = sum(ewma_tps_vals)/len(ewma_tps_vals) if ewma_tps_vals else None
            mean_ewma_step = sum(ewma_step_vals)/len(ewma_step_vals) if ewma_step_vals else None
            rows.append({
                'config': cfg,
                'mode': mode,
                'profile': profile,
                'window': window,
                'repeat': r + 1,
                'duration_s': f"{duration:.3f}",
                'num_workers': len(workers),
                'num_completed_tasks': len(completed),
                'num_pending_tasks': len(pending),
                'mean_ewma_tokens_per_sec': f"{mean_ewma_tps:.3f}" if mean_ewma_tps is not None else '',
                'mean_ewma_step_time_s': f"{mean_ewma_step:.6f}" if mean_ewma_step is not None else '',
            })

    # Write CSV
    fieldnames = ['config', 'mode', 'profile', 'window', 'repeat', 'duration_s', 'num_workers', 'num_completed_tasks', 'num_pending_tasks', 'mean_ewma_tokens_per_sec', 'mean_ewma_step_time_s']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def expand_glob(pattern: str) -> List[str]:
    return sorted(_glob.glob(pattern))
"""Experiment runner utilities.

Runs a sequence of configurations using either the ML or basic simulation,
captures the final status, and writes a CSV row per run including telemetry
EWMA summaries. Intended for quick local benchmarking and CI.
"""
