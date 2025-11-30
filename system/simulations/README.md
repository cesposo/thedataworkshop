# WAN LLM Training Simulation Suite

Comprehensive simulation framework for testing distributed LLM training in WAN environments with progressive complexity and realistic failure scenarios.

## Overview

This simulation suite provides:

- **Progressive DNN Scenarios**: From simple MNIST-style to 1B parameter LLM-scale models
- **WAN Challenge Scenarios**: 8 progressive challenges testing different aspects of distributed training
- **Network Simulation**: Realistic WAN conditions (bandwidth, latency, packet loss, jitter)
- **Failure Injection**: Crash-stop, transient failures, slowdowns, Byzantine behavior
- **Automated Test Harness**: Systematic experiment execution and metric collection
- **Results Analysis**: Comprehensive statistical analysis and reporting

## Directory Structure

```
simulations/
├── framework/           # Core simulation infrastructure
│   ├── network_simulator.py      # Network condition simulation
│   └── failure_injector.py       # Failure injection & churn
│
├── scenarios/           # DNN scenario definitions
│   └── dnn_scenarios.py           # 5 progressive scenarios (simple → LLM-scale)
│
├── challenges/          # WAN challenge scenarios
│   └── wan_challenges.py          # 8 progressive challenges
│
├── experiments/         # Experiment orchestration
│   ├── test_harness.py            # Automated test harness
│   └── results_analyzer.py        # Results collection & analysis
│
└── results/            # Experiment results (auto-created)
```

## Quick Start

### 1. Run a Single Experiment

```python
from experiments.test_harness import SimulationTestHarness
from scenarios.dnn_scenarios import get_scenario
from challenges.wan_challenges import get_challenge

# Create test harness
harness = SimulationTestHarness(verbose=True)

# Run experiment
result = harness.run_experiment(
    scenario=get_scenario('moderate'),           # DNN scenario
    challenge=get_challenge('wan_bandwidth'),    # WAN challenge
    use_async=True,
    compression_method='topk',
    compression_ratio=0.01,
    max_staleness=5,
    max_steps=100
)

print(f"Success: {result.success_criteria_met}")
print(f"Throughput: {result.metrics.steps_per_second:.2f} steps/s")
```

### 2. Run a Batch of Experiments

```python
from experiments.test_harness import SimulationTestHarness

harness = SimulationTestHarness(verbose=False)

# Test all combinations
scenarios = ['simple', 'moderate', 'large']
challenges = ['baseline', 'wan_bandwidth', 'heterogeneous']

results = harness.run_experiment_batch(scenarios, challenges, max_steps=100)

# Print summary
harness.print_batch_summary()
```

### 3. Analyze Results

```python
from experiments.results_analyzer import ResultsAnalyzer, ResultsStore

# Save results
store = ResultsStore()
store.save_batch_results(results, "my_experiment.json")
store.save_results_csv(results, "my_experiment.csv")

# Analyze
analyzer = ResultsAnalyzer(results)
analyzer.print_statistics()
analyzer.print_comparison_by_scenario()
analyzer.print_comparison_by_challenge()
analyzer.generate_report("report.txt")
```

## DNN Scenarios

Five progressive scenarios with increasing complexity:

| Scenario | Parameters | Gradient Size | Complexity | Description |
|----------|-----------|---------------|------------|-------------|
| **simple** | 235K | 0.9 MB | ⭐ | Simple 2-layer DNN (MNIST-style) |
| **moderate** | 638K | 2.4 MB | ⭐⭐ | Moderate CNN (CIFAR-10 style) |
| **large** | 11.7M | 44.6 MB | ⭐⭐⭐ | Large ResNet-18 style |
| **very_large** | 102M | 390.6 MB | ⭐⭐⭐⭐ | Very large transformer |
| **llm_scale** | 1B | 3.96 GB | ⭐⭐⭐⭐⭐ | LLM scale (ultimate challenge) |

## WAN Challenge Scenarios

Eight progressive challenges testing different aspects:

| Challenge | Difficulty | Description | Key Tests |
|-----------|-----------|-------------|-----------|
| **baseline** | 1/10 | Homogeneous LAN, no failures | Basic functionality |
| **wan_bandwidth** | 2/10 | WAN bandwidth constraints | Compression effectiveness |
| **wan_latency** | 3/10 | High latency WAN | Async training benefits |
| **heterogeneous** | 4/10 | Mixed worker speeds | Straggler handling |
| **unreliable** | 5/10 | Packet loss + failures | Fault tolerance |
| **churn** | 6/10 | Workers joining/leaving | Dynamic management |
| **byzantine** | 7/10 | Malicious workers | Byzantine tolerance |
| **extreme** | 10/10 | All challenges combined | Ultimate stress test |

## Network Profiles

Predefined network profiles for realistic simulation:

| Profile | Bandwidth | Latency | Packet Loss | Jitter |
|---------|-----------|---------|-------------|--------|
| Datacenter | 10 Gbps | 0.1 ms | 0% | 0.01 ms |
| LAN | 1 Gbps | 1 ms | 0% | 0.1 ms |
| Fast WAN | 100 Mbps | 50 ms | 0.1% | 5 ms |
| Typical WAN | 50 Mbps | 100 ms | 1% | 10 ms |
| Slow WAN | 10 Mbps | 200 ms | 2% | 20 ms |
| Edge | 5 Mbps | 300 ms | 5% | 30 ms |
| Satellite | 1 Mbps | 800 ms | 15% | 100 ms |

## Failure Modes

Supported failure injection modes:

- **CRASH_STOP**: Node crashes and stays down (permanent)
- **TRANSIENT**: Node temporarily unavailable (recovers after duration)
- **SLOWDOWN**: Node becomes slow (straggler simulation)
- **BYZANTINE**: Node sends corrupted gradients
- **NETWORK_PARTITION**: Network partition (isolated workers)

## Example Workflows

### Testing Compression Methods

```python
from experiments.test_harness import SimulationTestHarness
from scenarios.dnn_scenarios import get_scenario
from challenges.wan_challenges import get_challenge

harness = SimulationTestHarness(verbose=False)
scenario = get_scenario('large')
challenge = get_challenge('wan_bandwidth')

# Test different compression methods
compression_methods = [
    ('none', 1.0),
    ('fp16', 1.0),
    ('quantize', 1.0),
    ('topk', 0.1),
    ('topk', 0.01),
]

results = []
for method, ratio in compression_methods:
    result = harness.run_experiment(
        scenario=scenario,
        challenge=challenge,
        compression_method=method,
        compression_ratio=ratio,
        max_steps=100
    )
    results.append(result)

# Compare
from experiments.results_analyzer import ResultsAnalyzer
analyzer = ResultsAnalyzer(results)
analyzer.print_comparison_by_compression()
```

### Testing Staleness Bounds

```python
# Test different staleness bounds for async training
staleness_bounds = [3, 5, 8, 10, 15]

results = []
for staleness in staleness_bounds:
    result = harness.run_experiment(
        scenario=get_scenario('moderate'),
        challenge=get_challenge('heterogeneous'),
        use_async=True,
        max_staleness=staleness,
        max_steps=100
    )
    results.append(result)

# Analyze rejection rates
for result in results:
    staleness = result.config_snapshot['max_staleness']
    rejection = result.metrics.rejection_rate
    throughput = result.metrics.steps_per_second
    print(f"Staleness={staleness}: rejection={rejection:.1%}, throughput={throughput:.2f}")
```

### Progressive Stress Testing

```python
# Test all scenarios against the extreme challenge
scenarios = ['simple', 'moderate', 'large', 'very_large', 'llm_scale']
challenge = get_challenge('extreme')

results = []
for scenario_name in scenarios:
    scenario = get_scenario(scenario_name)
    result = harness.run_experiment(
        scenario=scenario,
        challenge=challenge,
        use_async=True,
        compression_method='topk',
        compression_ratio=0.01,
        max_staleness=15,
        max_steps=500
    )
    results.append(result)

# Generate comprehensive report
from experiments.results_analyzer import ResultsAnalyzer
analyzer = ResultsAnalyzer(results)
analyzer.generate_report("stress_test_report.txt")
```

## Integration with Main System

To integrate simulations with the actual training system:

```python
# TODO: This would connect to the real MainController
from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.compression.compressor import GradientCompressor

# Use simulation recommendations to configure real system
challenge = get_challenge('wan_bandwidth')

controller = MainController(
    use_async_training=challenge.recommended_async,
    max_staleness=challenge.recommended_max_staleness,
    # ... other config
)

compressor = GradientCompressor(
    method=challenge.recommended_compression,
    compression_ratio=challenge.recommended_compression_ratio
)
```

## Running Command-Line Examples

### List Available Scenarios and Challenges

```bash
# List DNN scenarios
python scenarios/dnn_scenarios.py

# List WAN challenges
python challenges/wan_challenges.py
```

### Run Test Harness Examples

```bash
# Run example experiments
python experiments/test_harness.py

# Run with analysis
python experiments/results_analyzer.py
```

## Metrics Collected

Each experiment collects comprehensive metrics:

- **Throughput**: Steps/second, total steps, total time
- **Network**: Bytes sent/received, bandwidth usage, latency
- **Gradients**: Submitted, accepted, rejected, staleness
- **Compression**: Compression ratio, bandwidth saved
- **Failures**: Total failures, failures by type
- **Workers**: Active workers, per-worker acceptance rates

## Success Criteria

Each challenge defines success criteria:

- **Minimum convergence steps**: Experiment must complete sufficient training
- **Maximum rejection rate**: Gradient rejections must be below threshold
- **Minimum throughput ratio**: Throughput must be acceptable relative to ideal

## Performance Tips

1. **Start simple**: Begin with `simple` scenario and `baseline` challenge
2. **Progressive complexity**: Gradually increase scenario/challenge difficulty
3. **Tune staleness**: Monitor rejection rates and adjust `max_staleness`
4. **Choose compression wisely**: Balance compression ratio vs. accuracy
5. **Use recommended settings**: Each challenge provides recommended configuration

## Future Enhancements

Potential future additions:

- [ ] Actual PyTorch model training (currently simulated)
- [ ] Real convergence metrics (loss, accuracy)
- [ ] Visualization dashboard
- [ ] Parallel experiment execution
- [ ] GPU/CPU resource simulation
- [ ] Memory usage tracking
- [ ] Energy consumption modeling

## Contributing

When adding new scenarios or challenges:

1. Follow existing naming conventions
2. Include comprehensive documentation
3. Define clear success criteria
4. Test with various configurations
5. Update this README

## References

- Main documentation: `/docs/async_training_and_compression.md`
- Research plan: `/docs/research_plan.md`
- Testing guide: `/TESTING_AND_DEMOS.md`
