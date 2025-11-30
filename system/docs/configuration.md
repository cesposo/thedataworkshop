# Configuration Guide

Complete reference for configuring distributed LLM training via YAML.

## Configuration File Structure

Top-level YAML keys in `config.yaml`:

### controller

Controller/coordinator configuration:

```yaml
controller:
  host: localhost
  port: 8000
  scheduler: gale-shapley              # Options: priority, capability, gale-shapley

  # Async training configuration
  use_async_training: true             # Enable async training (default: true, recommended for WAN)
  max_staleness: 50                    # Base max gradient staleness (default: 50, was 5)
  adaptive_staleness: true             # Enable adaptive staleness per worker (default: true, NEW!)
  min_staleness: 5                     # Minimum staleness bound (default: 5)
  max_staleness_multiplier: 10.0       # Max multiplier for slow workers (default: 10.0)

  # Communication protocol
  use_zmq: true                        # Use ZMQ+MessagePack for binary efficiency (default: true)

  # Gradient aggregation
  gradient_accumulation_size: 5        # Mini-batch size for aggregation (default: 3)

  # Security and privacy
  gradient_clip_norm: 1.0              # Max gradient L2 norm, 0=disabled (default: 1.0)
  enable_differential_privacy: false   # Add DP noise to gradients (default: false)
  dp_noise_multiplier: 0.1             # DP noise scale (default: 0.1)
```

**Parameters:**
- `host`, `port`: Controller listener address
- `scheduler`: Scheduling algorithm selection
- `use_async_training`: Enable bounded-staleness async SGD (recommended for WAN)
- `max_staleness`: Base maximum staleness for gradient acceptance (async mode only)
  - **Default:** 50 (updated from 5 based on simulation evidence)
  - **Previous default (5):** Too aggressive, caused 60-79% rejection in WAN simulations
  - **Recommended:** 50-100 for heterogeneous WAN environments
  - With adaptive_staleness=true, this is the base value; actual bounds vary per worker
- `adaptive_staleness`: **[NEW]** Enable adaptive staleness bounds per worker speed
  - **Default:** true (recommended for heterogeneous environments)
  - Fast workers get tighter staleness bounds (less tolerance for being behind)
  - Slow workers get looser staleness bounds (more tolerance for being behind)
  - Formula: `worker_staleness = max_staleness / worker_speed_multiplier`
  - Dramatically reduces gradient rejection rates in heterogeneous clusters
  - Set to false for homogeneous environments or debugging
- `min_staleness`: Minimum staleness bound for any worker (adaptive mode)
  - **Default:** 5
  - Even fast workers won't have staleness bound below this value
  - Prevents overly aggressive rejection
- `max_staleness_multiplier`: Maximum multiplier for slow worker staleness bounds
  - **Default:** 10.0
  - Slow workers can have staleness up to `max_staleness * max_staleness_multiplier`
  - Example: max_staleness=50, multiplier=10.0 → slowest worker gets bound of 500
  - Prevents unbounded staleness for extremely slow workers
- `use_zmq`: Use ZMQ+MessagePack instead of XML-RPC for binary gradient transfer
  - **Default:** true (preserves compression benefits)
  - Set to false for backward compatibility or debugging
- `gradient_accumulation_size`: Number of gradients to buffer before aggregation
  - **Default:** 3 (prevents micro-aggregations)
  - Higher values = more stable convergence, higher latency
- `gradient_clip_norm`: Maximum L2 norm for gradients (stability + basic privacy)
  - **Default:** 1.0 (recommended)
  - Set to 0 to disable clipping
- `enable_differential_privacy`: Enable DP noise for formal privacy guarantees
  - **Default:** false (disabled for performance)
  - Enable for privacy-sensitive deployments
- `dp_noise_multiplier`: Scale of Gaussian noise for DP
  - **Default:** 0.1 (moderate privacy)
  - Higher values = stronger privacy, lower accuracy

---

### workers

Worker node specifications:

```yaml
workers:
  - name: worker-1
    memory: 16                    # GB
    flops_per_second: 100         # Compute capability
    network_bandwidth: 1000       # Mbps
    host: localhost
    port: 8001                    # Set to 0 for auto-assignment
  - name: worker-2
    memory: 32
    flops_per_second: 200
    network_bandwidth: 50         # Simulates WAN worker
    host: localhost
    port: 8002
```

**Parameters:**
- `name`: Worker identifier
- `memory`: Available RAM in GB
- `flops_per_second`: Relative compute capability
- `network_bandwidth`: Network bandwidth in Mbps (used by scheduler)
- `host`, `port`: Worker RPC listener address

---

### model

Model configuration:

```yaml
model:
  name: tiny-lstm  # ModelLoader preset
```

**Available presets:**
- `tiny-lstm`: Small LSTM for testing
- `small-transformer`: Small transformer model
- See `ModelLoader` class for full list

---

### training

Training hyperparameters and protocol configuration:

```yaml
training:
  # Basic hyperparameters
  learning_rate: 0.001
  batch_size: 8
  num_epochs: 1
  num_samples: 100
  seq_length: 32
  num_workers: 2

  # Async training configuration (NEW - recommended for WAN)
  use_async_training: true      # Enable async mode (default: true)
  max_staleness: 5               # Max gradient staleness (in steps)

  # Gradient compression (NEW - essential for WAN)
  compression: topk              # Options: topk, quantize, fp16, none
  compression_ratio: 0.01        # For topk: keep top X% (0.01 = 1%)

  # Aggregation and Byzantine tolerance
  aggregation_rule: trimmed_mean # Options: mean, trimmed_mean, krum, bulyan
  trim_ratio: 0.3                # Trim 30% of extremes (default: 0.3, was 0.0)
  krum_f: 2                      # Byzantine workers to tolerate for Krum (default: n//4)
  bulyan_f: 2                    # Byzantine workers to tolerate for Bulyan (default: n//4)

  # Legacy sync mode parameters (deprecated for WAN)
  sync_window: 2                 # Barrier window size (sync mode only)
  sync_max_wait_s: 30            # Barrier timeout (sync mode only)
```

#### Training Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 0.001 | SGD learning rate |
| `batch_size` | int | 8 | Batch size per worker |
| `num_epochs` | int | 1 | Training epochs |
| `num_samples` | int | 100 | Dataset size |
| `seq_length` | int | 32 | Sequence length for language models |
| `num_workers` | int | 2 | Number of workers |
| `use_async_training` | bool | true | Enable async coordination |
| `max_staleness` | int | 5 | Max gradient staleness (async only) |
| `compression` | str | none | Compression method |
| `compression_ratio` | float | 0.01 | Sparsity for top-k compression |
| `aggregation_rule` | str | mean | Gradient aggregation method |
| `trim_ratio` | float | 0.3 | Trim fraction for trimmed_mean |
| `krum_f` | int | n//4 | Byzantine workers to tolerate (Krum) |
| `bulyan_f` | int | n//4 | Byzantine workers to tolerate (Bulyan) |

#### Aggregation Rules

| Rule | Byzantine Tolerance | Overhead | Use Case |
|------|-------------------|----------|----------|
| `mean` | None (0%) | 1x | Trusted homogeneous LAN |
| `trimmed_mean` | ~trim_ratio (30% default) | 2.5x | Basic Byzantine tolerance |
| `krum` | f < n/3 formal guarantee | 18x | Untrusted federated learning |
| `bulyan` | f < n/3 strongest guarantee | 20x | High-security production |

**Detailed Descriptions:**
- **mean**: Simple averaging, no Byzantine tolerance, fastest
- **trimmed_mean**: Remove extreme gradients before averaging, moderate tolerance
  - `trim_ratio=0.3` removes 15% smallest + 15% largest gradients
  - Default changed from 0.0 to 0.3 for basic Byzantine protection
- **krum**: Distance-based gradient selection, formal Byzantine tolerance up to f < n/3
  - Selects most representative gradients via pairwise distance scoring
  - `krum_f` parameter specifies number of Byzantine workers to tolerate (default: n//4)
- **bulyan**: Krum + trimmed mean, strongest Byzantine tolerance guarantee
  - Combines Krum selection with trimmed mean aggregation
  - `bulyan_f` parameter specifies Byzantine tolerance (default: n//4)
  - Recommended for high-security environments

**See Also:** `BYZANTINE_DEFENSE.md` for detailed algorithm descriptions and security analysis

#### Compression Options

| Value | Description | Compression | Bandwidth Reduction | Use Case |
|-------|-------------|-------------|---------------------|----------|
| `none` | No compression | 1x | 0% | LAN training |
| `fp16` | Half-precision | 2x | 50% | Safe default |
| `quantize` | 8-bit quantization | 4x | 75% | WAN training |
| `topk` | Top-k sparsification | Configurable | Up to 99% | Bandwidth-limited WAN |

---

### tasks

Task/job specifications for scheduling:

```yaml
tasks:
  - id: ml-task-0
    model_shard_size: 0.5    # Fraction of model (0-1)
    data_size: 0.1           # Data shard size
    required_flops: 1000     # Compute requirement
    priority: 1              # Scheduling priority
  - id: ml-task-1
    model_shard_size: 0.5
    data_size: 0.1
    required_flops: 1000
    priority: 2
```

---

### simulation

Simulation/experiment parameters:

```yaml
simulation:
  heartbeat_cycles: 5
  heartbeat_interval: 1          # seconds
  scheduling_cycles: 3
  scheduling_interval: 2         # seconds
```

---

## Complete Example Configurations

### LAN Cluster (Homogeneous)

```yaml
controller:
  host: localhost
  port: 8000
  scheduler: gale-shapley
  use_async_training: false    # Sync mode for homogeneous cluster

workers:
  - name: gpu-1
    memory: 32
    flops_per_second: 500
    network_bandwidth: 10000   # 10 Gbps
    host: localhost
    port: 8001
  - name: gpu-2
    memory: 32
    flops_per_second: 500
    network_bandwidth: 10000
    host: localhost
    port: 8002

model:
  name: small-transformer

training:
  learning_rate: 0.001
  batch_size: 16
  num_epochs: 5
  use_async_training: false
  compression: none            # No compression needed on LAN
  aggregation_rule: mean       # Simple averaging
```

### WAN Cluster (Heterogeneous) - Conservative

```yaml
controller:
  host: 0.0.0.0
  port: 8000
  scheduler: gale-shapley
  use_async_training: true
  max_staleness: 5

workers:
  - name: datacenter-gpu
    memory: 64
    flops_per_second: 1000
    network_bandwidth: 1000    # 1 Gbps
    host: 192.168.1.10
    port: 8001
  - name: edge-device
    memory: 8
    flops_per_second: 50
    network_bandwidth: 50      # 50 Mbps WAN
    host: 203.0.113.45
    port: 8002
  - name: volunteer-node
    memory: 16
    flops_per_second: 100
    network_bandwidth: 100
    host: 198.51.100.88
    port: 8003

model:
  name: tiny-lstm

training:
  learning_rate: 0.001
  batch_size: 8
  num_epochs: 3
  use_async_training: true
  max_staleness: 5
  compression: fp16            # Safe 2x compression
  aggregation_rule: trimmed_mean
  trim_ratio: 0.1              # Byzantine tolerance
```

### WAN Cluster - Aggressive Bandwidth Optimization

```yaml
controller:
  host: 0.0.0.0
  port: 8000
  scheduler: gale-shapley
  use_async_training: true
  max_staleness: 10            # Higher tolerance for heterogeneity

workers:
  - name: server-1
    memory: 32
    flops_per_second: 500
    network_bandwidth: 100
    host: 192.168.1.10
    port: 8001
  - name: mobile-worker
    memory: 4
    flops_per_second: 10
    network_bandwidth: 5       # 5 Mbps mobile connection
    host: 203.0.113.45
    port: 8002

model:
  name: tiny-lstm

training:
  learning_rate: 0.001
  batch_size: 4
  num_epochs: 5
  use_async_training: true
  max_staleness: 10
  compression: topk            # Aggressive compression
  compression_ratio: 0.01      # Keep top 1% (100x compression)
  aggregation_rule: trimmed_mean
  trim_ratio: 0.2              # Higher Byzantine tolerance
```

### Federated Learning (Untrusted Workers)

```yaml
controller:
  host: 0.0.0.0
  port: 8000
  scheduler: gale-shapley
  use_async_training: true
  max_staleness: 100           # High tolerance for heterogeneous federated clients
  adaptive_staleness: true     # Adapt to client speeds
  use_zmq: true                # Binary efficiency
  gradient_accumulation_size: 10  # Larger buffer for Krum variance reduction
  gradient_clip_norm: 0.5      # Tighter clipping for privacy
  enable_differential_privacy: true  # Privacy protection
  dp_noise_multiplier: 0.2

workers:
  - name: client-1
    memory: 8
    flops_per_second: 100
    network_bandwidth: 50
    host: 203.0.113.10
    port: 8001
  - name: client-2
    memory: 16
    flops_per_second: 200
    network_bandwidth: 100
    host: 198.51.100.20
    port: 8002
  # ... more clients ...

model:
  name: small-transformer

training:
  learning_rate: 0.001
  batch_size: 8
  num_epochs: 10
  use_async_training: true
  max_staleness: 100
  compression: topk
  compression_ratio: 0.05      # 5% sparsity (20x compression)
  aggregation_rule: krum       # Byzantine-robust aggregation
  krum_f: 5                    # Tolerate up to 5 malicious clients
```

**Rationale:**
- Krum provides formal Byzantine tolerance for untrusted federated clients
- Differential privacy protects against gradient inversion attacks
- Adaptive staleness handles heterogeneous client speeds (mobile, desktop, server)
- High `max_staleness=100` tolerates slow/intermittent clients

### High-Security Production Environment

```yaml
controller:
  host: 0.0.0.0
  port: 8000
  scheduler: gale-shapley
  use_async_training: true
  max_staleness: 50
  adaptive_staleness: true
  use_zmq: true
  gradient_accumulation_size: 10  # Large buffer for Bulyan stability
  gradient_clip_norm: 0.5      # Gradient clipping
  enable_differential_privacy: true
  dp_noise_multiplier: 0.3     # Strong privacy

workers:
  # Mix of trusted and untrusted workers
  - name: datacenter-1
    memory: 64
    flops_per_second: 1000
    network_bandwidth: 1000
    host: 192.168.1.10
    port: 8001
  - name: edge-node-1
    memory: 16
    flops_per_second: 100
    network_bandwidth: 100
    host: 203.0.113.30
    port: 8002
  - name: volunteer-1
    memory: 8
    flops_per_second: 50
    network_bandwidth: 50
    host: 198.51.100.40
    port: 8003
  # ... more workers ...

model:
  name: small-transformer

training:
  learning_rate: 0.001
  batch_size: 16
  num_epochs: 20
  use_async_training: true
  max_staleness: 50
  compression: topk
  compression_ratio: 0.01      # 1% sparsity (100x compression)
  aggregation_rule: bulyan     # Strongest Byzantine defense
  bulyan_f: 3                  # Tolerate up to 3 Byzantine workers
```

**Rationale:**
- Bulyan provides strongest Byzantine tolerance guarantee
- Combined with differential privacy for defense-in-depth
- Gradient clipping + DP noise for privacy and stability
- Adaptive staleness handles heterogeneous worker speeds
- High compression for bandwidth efficiency
- Defense against: malicious workers, gradient inversion, backdoor attacks

---

## Environment Variables

Additional configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEMETRY_ALPHA` | 0.2 | EWMA smoothing for telemetry (0-1) |
| `LOG_LEVEL` | INFO | Logging verbosity |

**Usage:**
```bash
export TELEMETRY_ALPHA=0.3
export LOG_LEVEL=DEBUG
dist-llm-train demo-ml --config config.yaml
```

---

## CLI Overrides

Many configuration options can be overridden via CLI:

```bash
# Use custom config file
dist-llm-train demo-ml --config custom_config.yaml

# Override logging
dist-llm-train demo-ml --config config.yaml --log-level DEBUG

# Persistence options
dist-llm-train demo-ml --config config.yaml \
  --state-db state.db \
  --ps-checkpoint checkpoint.pt
```

---

## Configuration Best Practices

### For Development/Testing
- Use `tiny-lstm` model
- Small `num_samples` (10-100)
- `use_async_training: false` for deterministic behavior
- `compression: none` to isolate training bugs

### For WAN Experiments
- Always set `use_async_training: true`
- Start with `compression: fp16` (safe default)
- Monitor rejection rate, increase `max_staleness` if >20%
- Use `aggregation_rule: trimmed_mean` for untrusted workers

### For Production
- Profile network bandwidth and set worker `network_bandwidth` accurately
- Choose compression based on actual bandwidth measurements
- Set `max_staleness` based on heterogeneity (more variance = higher staleness)
- Enable checkpointing for fault recovery

---

## Troubleshooting

**High rejection rate (>20%)**
- Increase `max_staleness`
- Reduce heterogeneity (remove very slow workers)
- Check network latency

**Slow convergence**
- Reduce `compression_ratio` (less aggressive compression)
- Lower `max_staleness` (fresher gradients)
- Increase `learning_rate`

**High bandwidth usage**
- Enable compression if not already
- Use more aggressive compression (`topk` with lower ratio)
- Reduce `batch_size` to reduce gradient size

**Byzantine workers suspected**
- Enable Byzantine-robust aggregation:
  - Low risk: `aggregation_rule: trimmed_mean` with `trim_ratio: 0.3`
  - Medium risk: `aggregation_rule: krum` with `krum_f: n//4`
  - High risk: `aggregation_rule: bulyan` with `bulyan_f: n//4`
- Monitor per-worker metrics for anomalies
- Consider enabling differential privacy for additional protection
- See `BYZANTINE_DEFENSE.md` for detailed security analysis

---

## See Also

- **Training Protocol**: `docs/training_protocol.md` - Detailed protocol documentation
- **Async + Compression**: `docs/async_training_and_compression.md` - Technical deep dive
- **Byzantine Defense**: `BYZANTINE_DEFENSE.md` - Byzantine-robust aggregation algorithms and security analysis
- **Integration Fixes**: `INTEGRATION_FIXES.md` - ZMQ, gradient accumulation, and differential privacy
- **Adaptive Staleness**: `ADAPTIVE_STALENESS.md` - Adaptive staleness bounds for heterogeneous environments
- **Examples**: `TESTING_AND_DEMOS.md` - Configuration examples for different scenarios
- **CLI Guide**: `docs/cli_guide.md` - Command-line interface reference
