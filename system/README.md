# 🚀 dist-llm-train: Distributed LLM Training Simulator

[![Build Status](https://img.shields.io/travis/com/your-username/your-repo.svg?style=flat-square)](https://travis-ci.com/your-username/your-repo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0-beta-blue.svg?style=flat-square)](https://github.com/your-username/llm_distributed_training)
[![Tests](https://img.shields.io/badge/tests-29%20passing-green.svg?style=flat-square)](tests/)

`dist-llm-train` is a **research/simulation framework** for distributed training of Large Language Models (LLMs) on heterogeneous and wide-area networks. It lets you explore scheduling, fault-tolerance, and WAN impairment scenarios with real PyTorch tasks and pluggable schedulers.

**Current status:**
- ✅ 29 tests passing locally (scheduler, controller, worker, integration, PS robustness, netem)
- ✅ Real PyTorch training demo (tiny LSTM) with parameter-server sync
- ✅ Pluggable schedulers (Gale–Shapley, priority, capability with telemetry)
- ✅ WAN netem presets (latency/jitter/loss/bandwidth caps) via CLI/config
- ⚠️ ZMQ communicator is optional; falls back to XML-RPC if `pyzmq` is missing

The main goal of this project is to democratize LLM training by developing and evaluating strategies for leveraging non-dedicated, "transient" resources, such as those in Open Science pool environments and federated learning deployments.

## 🎯 Key Features

### Training & Simulation
*   ✅ **Real PyTorch Training:** Train small models (LSTMs/Transformers) with a parameter server and sync coordinator
*   ✅ **Simulation Mode:** Lightweight infra simulation for controller/worker/heartbeat/scheduling loops
*   ✅ **WAN Netem:** Latency/jitter/loss/bandwidth presets (good/cellular/degraded/brownout/straggler/satellite) via `--netem-profile`

### Async Training & Optimization
*   ⚡ **Synchronous + Async (bounded staleness):** Simple sync and optional async coordinator
*   📦 **Gradient Compression (optional):** Top-k/quantization/FP16 hooks (use when ZMQ is available)

### Security & Robustness
*   🔒 **Robust Aggregation:** Mean, trimmed mean; hooks for robust/DP extensions
*   🛡️ **Fault Tolerance:** Worker failure detection, heartbeat monitoring, task rescheduling, and checkpointing

### Scheduling & Configuration
*   🔄 **Pluggable Scheduling:** Gale-Shapley stable matching, Priority-based, Capability-based (EWMA throughput)
*   ⚙️ **Flexible Configuration:** YAML/JSON configs; CLI flags to select scheduler, netem profile, and telemetry smoothing
*   🔧 **Extensible Architecture:** Modular design for adding schedulers, communicators, or training policies

## 🏛️ Architecture Overview

The system follows a controller-worker architecture optimized for wide-area networks:

*   **Controller:** Central coordinator managing cluster state, worker registration, task scheduling, and fault detection. Supports adaptive staleness coordination for 100% gradient acceptance.
*   **Worker Nodes:** Execute real PyTorch training (forward/backward passes), compress gradients, and submit to parameter server without blocking on slow peers.
*   **Communication Layer:**
    - **ZeroMQ + MessagePack (optional):** Binary-efficient gradient transfer (requires `pyzmq`)
    - **XML-RPC (default/fallback):** Control-plane RPC; netem is applied here
*   **Parameter Server:** Central gradient aggregation with:
    - **Adaptive Staleness:** Per-worker bounds (fast workers: tight bounds, slow workers: loose bounds)
    - **Byzantine Defense:** Krum, Bulyan, trimmed mean for malicious gradient filtering (f < n/3 tolerance)
    - **Differential Privacy:** Gradient clipping + optional noise for privacy
*   **Compression Module:** Top-k (100x), quantization (4x), FP16 (2x) for bandwidth reduction
*   **Testing Framework:** Real PyTorch convergence tests, Byzantine attack simulations, full integration validation

## 🚀 Getting Started

1.  Clone the repository:
    ```bash
    git clone https://github.com/cesposo/thedataworkshop.git
    cd llm_distributed_training
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Install the project in editable mode:
    ```bash
    pip install -e .
    ```

## ⚡ Quick Start: Running a Simulation

You can run the built-in simulations using the command-line interface:

```bash
# Run a simulation with real PyTorch ML training (configurable)
dist-llm-train demo-ml --config config.yaml --log-level INFO

# Run a basic simulation of the infrastructure (no ML training)
dist-llm-train demo-basic --config configs/priority_demo.yaml --log-level DEBUG
```

Alternatively, you can run the simulation scripts directly:

```bash
# Run the ML training simulation
python ml_training_simulation.py

# Run the basic infrastructure simulation
python simulation.py
```

Sample configs are available under `configs/` to help you experiment quickly:

- `configs/priority_demo.yaml` – priority scheduler with two workers
- `configs/gale_shapley_demo.yaml` – Gale–Shapley scheduler with three workers
- `configs/submit_job_demo.yaml` – config for the `submit-job` CLI demo
- `configs/capability_demo.yaml` – capability scheduler with three workers

To use a sample config with the submit-job demo:

```bash
dist-llm-train submit-job --config configs/submit_job_demo.yaml
```

To inspect controller status or metrics/telemetry from a running controller:

```bash
# Status (workers, pending/completed tasks, metrics, telemetry)
dist-llm-train status --controller http://localhost:8000

# Pretty-print metrics and worker telemetry
dist-llm-train metrics --controller http://localhost:8000

# Simulate WAN conditions (latency/loss/bandwidth caps) with a preset profile
dist-llm-train demo-basic --config configs/priority_demo.yaml --netem-profile cellular

# Override netem per message class (heartbeats/task/telemetry/ps_sync)
# Place in your config under `network`, or pass inline JSON/YAML via --netem-profile
# network:
#   profile: degraded
#   overrides:
#     heartbeat:
#       loss_pct: 0.0
#     task:
#       loss_pct: 0.05
#     telemetry:
#       loss_pct: 0.1
#     ps_sync:
#       base_rtt_ms: 200
```

## ⚙️ Configuration

Simulations are configured via a central YAML file: `config.yaml` in the repo root. The loader in `dist_llm_train/config.py` supports YAML and JSON, so you can also provide JSON if preferred.

Key sections in `config.yaml`:

```yaml
# Controller configuration
controller:
  host: localhost
  port: 8000
  scheduler: priority   # options: priority, gale-shapley, capability

# Worker configuration
workers:
  - name: worker-1
    memory: 16
    flops_per_second: 100
    network_bandwidth: 1000
    host: localhost
    port: 8001
  - name: worker-2
    memory: 32
    flops_per_second: 200
    network_bandwidth: 1000
    host: localhost
    port: 8002

# Model configuration
model:
  name: tiny-lstm        # see ModelLoader for presets

# Training configuration
training:
  learning_rate: 0.001
  batch_size: 8
  num_epochs: 1
  num_samples: 10
  seq_length: 32
  num_workers: 2

  # Async training (recommended for WAN)
  use_async_training: true       # Enable bounded-staleness async SGD
  max_staleness: 50              # Base staleness bound (v2.0: increased from 5)

  # Gradient compression (essential for WAN)
  compression: topk              # Options: topk, quantize, fp16, none
  compression_ratio: 0.01        # Keep top 1% (100x compression)

  # Byzantine-robust aggregation (v2.0)
  aggregation_rule: bulyan       # Options: mean, trimmed_mean, krum, bulyan
  bulyan_f: 2                    # Tolerate 2 Byzantine workers (f < n/3)

# Controller configuration (v2.0 features)
controller:
  adaptive_staleness: true       # Per-worker staleness bounds (NEW!)
  use_zmq: true                  # Binary communication (recommended)
  gradient_clip_norm: 1.0        # Gradient clipping for stability
  enable_differential_privacy: false  # Optional privacy protection

# Task configuration
tasks:
  - id: ml-task-0
    model_shard_size: 0.5
    data_size: 0.1
    required_flops: 1000
    priority: 1
  - id: ml-task-1
    model_shard_size: 0.5
    data_size: 0.1
    required_flops: 1000
    priority: 2

# Simulation configuration
simulation:
  heartbeat_cycles: 5
  heartbeat_interval: 1
  scheduling_cycles: 3
  scheduling_interval: 2
```

**Notes:**
- Change `controller.scheduler` to switch between `priority`, `gale-shapley`, and `capability` schedulers.
- Ports set to `0` will auto-select a free port where supported.
- The simulation scripts read `config.yaml` by default.

## 🛠️ Development and Testing

We welcome contributions! To set up a development environment, follow the installation instructions above.

To run the test suite:

```bash
# With unittest (ships with Python)
python -m unittest discover -v -s tests

# Or with pytest
pytest -q
```

## 🤝 Contributing

Please read our (forthcoming) `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## 📜 Citing This Work

If you use this framework in your research, please cite it as follows:

```bibtex
@misc{your-name-2025-dist-llm-train,
  author = {Chris Esposo},
  title = {dist-llm-train: A Simulation Framework for Distributed LLM Training},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cesposo/thedataworkshop}},
}
```

## 📚 Documentation

See the documentation index for detailed architecture, design, and how-to guides:

- docs/index.md
- docs/overview.md
- docs/architecture.md
- docs/scheduler_design.md
- docs/training_protocol.md
- docs/telemetry_observability.md
- docs/persistence_and_resume.md
- docs/configuration.md
- docs/cli_guide.md
- docs/experiments.md
- docs/testing_and_quality.md
