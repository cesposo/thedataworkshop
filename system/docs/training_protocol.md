# Training Protocol

Workers train PyTorch models and synchronize through a Parameter Server (PS) with either Synchronous or Asynchronous coordination. The system supports two protocols optimized for different environments:

- **Synchronous Protocol**: Traditional barrier-based training for homogeneous LAN clusters
- **Asynchronous Protocol** (Recommended): Bounded-staleness async SGD for heterogeneous/WAN environments

Flow:
1) Worker loads model/dataset and configures sync via RPC:
   - `configure_training_sync(num_workers, window_size?, max_wait_s?)`
   - `ps_initialize_if_empty(model_state)` (first worker sets parameters)
   - `ps_get_parameters()`
2) Per batch:
   - Compute forward/backward; collect gradients as serializable tensors.
   - `ps_sync_step(worker_id, gradients, lr)`; server aggregates and updates.
   - Receive updated parameters; continue training.
3) Report metrics and telemetry back to controller.

Partial sync windows:
- `window_size` specifies how many workers the barrier waits for (default = num_workers).
- `max_wait_s` releases the barrier early if peers lag or fail.

Aggregation rules:
- Mean (default): average gradients across the last window.
- Trimmed mean: drop extremes by a configured fraction `trim_ratio` before averaging (per-element), improving robustness to outliers.

Telemetry and EWMA:
- Workers publish per-batch tokens/sec and step_time_s.
- Controller computes EWMA (alpha configurable via `TELEMETRY_ALPHA`), making the scheduler responsive yet stable.

Failure handling:
- Heartbeats with timeouts mark workers offline and requeue in-flight tasks.
- PS state can be checkpointed/restored; controller status snapshots enable resume.

---

## Asynchronous Training Protocol (NEW)

**Recommended for heterogeneous/WAN environments**

The asynchronous protocol eliminates synchronous barriers, allowing workers to progress at their own pace while preventing divergence through bounded staleness.

### Configuration

```yaml
training:
  use_async_training: true      # Enable async mode (default: true)
  max_staleness: 5               # Maximum gradient staleness (in steps)
  compression: topk              # Gradient compression method
  compression_ratio: 0.01        # For topk: keep top 1%
  aggregation_rule: trimmed_mean # Byzantine tolerance
  trim_ratio: 0.1                # Trim 10% of extremes
```

### Flow

1) **Worker initialization** (same as sync protocol):
   - `configure_training_sync(num_workers, use_async=True)`
   - `ps_initialize_if_empty(model_state)`
   - `ps_get_parameters()`

2) **Per batch (no barriers)**:
   - Compute forward/backward; collect gradients
   - **Compress gradients** (if enabled)
   - `ps_sync_step(worker_id, gradients, lr, worker_step, metadata?)`
   - Controller checks staleness: `global_step - worker_step ≤ max_staleness`
   - If accepted: gradients applied immediately, worker receives updated parameters
   - If rejected: worker continues with current parameters (no blocking)

3) **Continuous aggregation**:
   - Parameter server accumulates gradients as they arrive
   - Updates applied immediately without waiting for all workers
   - Fast workers don't wait for slow workers

### Staleness Management

**Staleness** = Global step counter - Worker's step number

- **Fresh gradients** (staleness = 0-2): Always accepted
- **Stale gradients** (staleness = 3-5): Accepted within bound
- **Too stale** (staleness > max_staleness): Rejected to prevent divergence

**Monitoring**: Controller tracks rejection rate and average staleness
- Healthy: <10% rejection rate
- Increase `max_staleness` if rejection rate >20%

---

## Gradient Compression (NEW)

To address the "bandwidth wall" in WAN training, gradients can be compressed before transmission.

### Available Methods

| Method | Compression | Bandwidth Reduction | Use Case |
|--------|-------------|---------------------|----------|
| `none` | 1x | 0% | LAN (no compression needed) |
| `fp16` | 2x | 50% | Safe default, minimal accuracy loss |
| `quantize` | 4x | 75% | Good balance for WAN |
| `topk` (10%) | 10x | 90% | Bandwidth-constrained WAN |
| `topk` (1%) | 100x | 99% | Severely limited bandwidth |

### Configuration

```yaml
training:
  compression: topk              # Method: topk, quantize, fp16, none
  compression_ratio: 0.01        # For topk only: keep top 1%
```

### How It Works

1) **Worker side**:
   - Compute gradients normally
   - Compress using selected method
   - Transmit compressed gradients + metadata
   - Bandwidth used: original_size / compression_ratio

2) **Parameter server side**:
   - Receive compressed gradients + metadata
   - Decompress to original shape
   - Aggregate and update as normal

### Example: 1B Parameter Model

| Configuration | Gradient Size | 50 Mbps Transmission Time |
|---------------|---------------|---------------------------|
| No compression | 4 GB | 10+ minutes |
| FP16 | 2 GB | 5 minutes |
| Quantize | 1 GB | 2.5 minutes |
| Top-k 10% | 400 MB | 1 minute |
| Top-k 1% | 40 MB | 6 seconds |

**See also**: `docs/async_training_and_compression.md` for detailed technical guide

---

## Protocol Comparison

| Aspect | Synchronous | Asynchronous |
|--------|------------|--------------|
| **Barriers** | Yes (workers wait) | No (non-blocking) |
| **Straggler problem** | Cluster speed = slowest worker | Each worker progresses independently |
| **Fault tolerance** | Failed worker blocks all | Failed worker doesn't block |
| **Convergence** | Deterministic | May be slower but better wall-clock time |
| **Best for** | Homogeneous LAN clusters | Heterogeneous/WAN environments |
| **Configuration** | `use_async_training: false` | `use_async_training: true` |

---

## Choosing the Right Configuration

### Homogeneous LAN Cluster
```yaml
training:
  use_async_training: false
  compression: none
  aggregation_rule: mean
```

### Heterogeneous WAN (Conservative)
```yaml
training:
  use_async_training: true
  max_staleness: 5
  compression: fp16
  aggregation_rule: trimmed_mean
  trim_ratio: 0.1
```

### Bandwidth-Constrained WAN (Aggressive)
```yaml
training:
  use_async_training: true
  max_staleness: 10
  compression: topk
  compression_ratio: 0.05  # 5% sparsity
  aggregation_rule: trimmed_mean
  trim_ratio: 0.2
```

---

## Adaptive Staleness Bounds (NEW - Recommended)

**Problem:** Fixed staleness bounds cause high rejection rates (60-79%) in heterogeneous clusters.

**Solution:** Per-worker adaptive staleness bounds based on worker speed.

### How It Works

1. **Track Worker Speed:**
   - Monitor gradient submission intervals
   - Compute speed multiplier relative to cluster median
   - Fast worker: speed=2.0 (submits 2x faster than median)
   - Slow worker: speed=0.3 (submits 0.3x slower than median)

2. **Adjust Staleness Bounds:**
   - Fast workers: tighter bounds (less tolerance for being behind)
   - Slow workers: looser bounds (more tolerance for being behind)
   - Formula: `worker_staleness = max_staleness / worker_speed`

3. **Clamp to Reasonable Range:**
   - Minimum: `min_staleness` (default: 5)
   - Maximum: `max_staleness * max_staleness_multiplier` (default: 50 * 10 = 500)

### Configuration

```yaml
controller:
  use_async_training: true
  max_staleness: 50                 # Base staleness (increased from 5)
  adaptive_staleness: true          # Enable adaptive bounds (NEW!)
  min_staleness: 5                  # Minimum for fast workers
  max_staleness_multiplier: 10.0    # Maximum multiplier for slow workers
```

### Example

**Cluster:** 5 workers with speeds [2.0, 1.5, 1.0, 0.5, 0.3]
**Base staleness:** 50

| Worker | Speed | Staleness Bound | Calculation |
|--------|-------|----------------|-------------|
| Fast-1 | 2.0x | 25 | 50 / 2.0 |
| Fast-2 | 1.5x | 33 | 50 / 1.5 |
| Medium | 1.0x | 50 | 50 / 1.0 |
| Slow-1 | 0.5x | 100 | 50 / 0.5 |
| Slow-2 | 0.3x | 166 | 50 / 0.3 |

**Result:** All workers accepted instead of 60-79% rejection with fixed staleness.

**See also:** `ADAPTIVE_STALENESS.md` for detailed design and analysis

---

## Byzantine-Robust Aggregation (NEW - Recommended)

**Problem:** Malicious or buggy workers can send corrupted gradients, poisoning the global model.

**Solution:** Byzantine-robust aggregation algorithms that filter corrupted gradients.

### Available Algorithms

| Rule | Byzantine Tolerance | Overhead | Use Case |
|------|-------------------|----------|----------|
| `mean` | None (0%) | 1x | Trusted homogeneous LAN |
| `trimmed_mean` | ~trim_ratio (30% default) | 2.5x | Basic Byzantine tolerance |
| `krum` | f < n/3 formal guarantee | 18x | Untrusted federated learning |
| `bulyan` | f < n/3 strongest guarantee | 20x | High-security production |

### Algorithm Details

**Trimmed Mean:**
- Remove extreme gradients before averaging
- `trim_ratio=0.3` removes 15% smallest + 15% largest
- Simple, efficient, moderate Byzantine tolerance

**Krum:**
- Distance-based gradient selection
- Selects most representative gradients via pairwise distances
- Formal Byzantine tolerance: f < n/3 malicious workers
- Ref: Blanchard et al. (2017)

**Bulyan:**
- Combines Krum selection with trimmed mean
- Strongest Byzantine tolerance guarantee
- Double filtering (Krum + trimmed mean)
- Ref: El Mhamdi et al. (2018)

### Configuration

```yaml
training:
  aggregation_rule: bulyan          # Choose: mean, trimmed_mean, krum, bulyan
  bulyan_f: 2                       # Number of Byzantine workers to tolerate
  # OR
  aggregation_rule: krum
  krum_f: 3
  # OR
  aggregation_rule: trimmed_mean
  trim_ratio: 0.3                   # Trim 30% (15% each side)
```

### Example: 9 Workers with 2 Byzantine

**Without Byzantine Defense (mean):**
- Byzantine workers send corrupted gradients
- Model diverges or produces poor results
- No protection against malicious workers

**With Byzantine Defense (Bulyan, f=2):**
- Krum selects 5 most representative gradients (n-2f = 9-4 = 5)
- Trimmed mean applied to selected gradients
- Byzantine gradients filtered out
- Training converges despite malicious workers

**See also:** `BYZANTINE_DEFENSE.md` for complete guide and attack scenarios

---

## Differential Privacy (Optional)

**Problem:** Gradients can leak private training data (gradient inversion attacks).

**Solution:** Differential privacy via gradient clipping + noise.

### Configuration

```yaml
controller:
  gradient_clip_norm: 1.0                 # Gradient clipping (always recommended)
  enable_differential_privacy: false      # Enable DP noise (optional)
  dp_noise_multiplier: 0.1                # Noise scale
```

### How It Works

1. **Gradient Clipping:**
   - Bounds L2 norm of gradients to `gradient_clip_norm`
   - Prevents gradient explosions
   - Required for differential privacy
   - Always enabled (recommended: 1.0)

2. **DP Noise (Optional):**
   - Adds Gaussian noise to clipped gradients
   - Noise scale: `sensitivity * dp_noise_multiplier`
   - Provides formal (ε,δ)-differential privacy guarantees
   - Trade-off: privacy vs. accuracy

### Privacy-Accuracy Trade-Off

| DP Noise Multiplier | Privacy | Accuracy Impact |
|-------------------|---------|-----------------|
| 0.0 (disabled) | None | 0% (baseline) |
| 0.05 | Weak | ~1-2% degradation |
| 0.1 | Moderate | ~3-5% degradation |
| 0.3 | Strong | ~10-15% degradation |

**Recommendation:** Start with DP disabled, enable only for privacy-sensitive applications.

---

## Communication Protocols

### ZMQ + MessagePack (Recommended)

**Binary-efficient protocol for gradient transfer:**
- ZMQ: ROUTER/DEALER sockets for async communication
- MessagePack: Binary serialization (no base64 overhead)
- Preserves compression benefits (100x stays 100x)
- 33% faster than XML-RPC

```yaml
controller:
  use_zmq: true  # Enable ZMQ (default: true, recommended)
```

### XML-RPC (Legacy Fallback)

**Fallback for compatibility:**
- Automatic fallback if ZMQ unavailable
- Base64 encoding adds 33% overhead
- 100x compression degraded to 75x effective
- Slower but universally compatible

**Recommendation:** Use ZMQ for production, XML-RPC for debugging/compatibility.

---

## Updated Configuration Recommendations

### Production WAN Training (Recommended)

```yaml
controller:
  use_async_training: true
  max_staleness: 50                    # Increased from 5
  adaptive_staleness: true             # NEW: Per-worker bounds
  use_zmq: true                        # Binary efficiency
  gradient_accumulation_size: 10       # Mini-batch aggregation
  gradient_clip_norm: 1.0              # Stability + basic privacy

training:
  compression: topk
  compression_ratio: 0.01              # 1% sparsity (100x)
  aggregation_rule: bulyan             # Byzantine-robust
  bulyan_f: 2                          # Tolerate 2 malicious workers
```

### Federated Learning (Untrusted Clients)

```yaml
controller:
  use_async_training: true
  max_staleness: 100                   # High tolerance for slow clients
  adaptive_staleness: true
  use_zmq: true
  gradient_clip_norm: 0.5              # Tighter clipping
  enable_differential_privacy: true    # Privacy protection
  dp_noise_multiplier: 0.2

training:
  compression: topk
  compression_ratio: 0.05              # 5% sparsity (20x)
  aggregation_rule: krum               # Byzantine-robust
  krum_f: 5                            # Tolerate up to 5 malicious clients
```

### High-Security Production

```yaml
controller:
  use_async_training: true
  max_staleness: 50
  adaptive_staleness: true
  use_zmq: true
  gradient_accumulation_size: 10       # Larger buffer for Bulyan
  gradient_clip_norm: 0.5
  enable_differential_privacy: true
  dp_noise_multiplier: 0.3             # Strong privacy

training:
  compression: topk
  compression_ratio: 0.01              # 1% sparsity
  aggregation_rule: bulyan             # Strongest Byzantine defense
  bulyan_f: 3
```

**See also:** `docs/configuration.md` for complete reference with all options.

---

## Troubleshooting

### High Gradient Rejection (>20%)

**Symptoms:**
- Many gradients rejected
- Slow workers starved
- Low worker utilization

**Solutions:**
1. **Enable adaptive staleness:**
   ```yaml
   controller:
     adaptive_staleness: true
   ```

2. **Increase base staleness:**
   ```yaml
   controller:
     max_staleness: 100  # From 50
   ```

3. **Remove extremely slow workers** (>10x slower than median)

### Poor Convergence Quality

**Symptoms:**
- Final accuracy much lower than baseline (>20% degradation)
- Model diverges

**Solutions:**
1. **Reduce compression aggressiveness:**
   ```yaml
   training:
     compression_ratio: 0.05  # From 0.01 (5% instead of 1%)
   ```

2. **Reduce Byzantine defense aggressiveness:**
   ```yaml
   training:
     aggregation_rule: trimmed_mean  # From bulyan
     trim_ratio: 0.2                 # From 0.3
   ```

3. **Adjust learning rate for async:**
   - Try 0.5x-2x baseline learning rate
   - Async may need different LR tuning

4. **Disable differential privacy if enabled:**
   ```yaml
   controller:
     enable_differential_privacy: false
   ```

### High Bandwidth Usage

**Symptoms:**
- Network saturation
- >100 GB/hour per worker

**Solutions:**
1. **Enable compression if not already:**
   ```yaml
   training:
     compression: topk
     compression_ratio: 0.01
   ```

2. **Use more aggressive compression:**
   ```yaml
   training:
     compression_ratio: 0.005  # 0.5% sparsity (200x)
   ```

3. **Reduce gradient synchronization frequency:**
   - Increase steps per sync
   - Larger local batch sizes

---

## References

**Documentation:**
- `ADAPTIVE_STALENESS.md` - Adaptive staleness design and analysis
- `BYZANTINE_DEFENSE.md` - Byzantine algorithms and security
- `docs/async_training_and_compression.md` - Technical deep dive
- `docs/configuration.md` - Complete configuration reference

**Papers:**
- Dean et al. (2012): "Large Scale Distributed Deep Networks"
- Blanchard et al. (2017): "Machine Learning with Adversaries"
- El Mhamdi et al. (2018): "Hidden Vulnerability of Distributed Learning"
- Abadi et al. (2016): "Deep Learning with Differential Privacy"

---

**Last Updated:** 2025-11-19
**Version:** 2.0 (Production Beta)
