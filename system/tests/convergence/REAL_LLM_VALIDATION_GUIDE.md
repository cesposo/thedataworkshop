# Real LLM Training Validation Guide

**Date:** 2025-11-19
**Status:** 📋 READY - Guide and templates prepared
**Purpose:** Validate distributed training system with real LLM models

---

## Overview

This guide describes how to validate the complete distributed training system using real LLM models (GPT-2, BERT, etc.) to ensure production-ready performance.

**Why This Matters:**
- Prior tests used synthetic data and simple networks
- Real LLMs have different convergence characteristics
- Need to validate: final accuracy, convergence speed, bandwidth usage
- Confirm no degradation from compression + Byzantine defense

---

## Prerequisites

### 1. Install Additional Dependencies

```bash
# Transformers library for pre-trained models
pip install transformers datasets accelerate

# For GPT-2
pip install tiktoken

# Optional: Weights & Biases for experiment tracking
pip install wandb
```

### 2. Hardware Requirements

**Minimum (Testing):**
- 2-4 workers with 8GB RAM each
- CPU-only fine, but slow (GPU recommended)
- 100 Mbps network (simulated WAN)

**Recommended (Validation):**
- 5-10 workers with 16GB+ RAM
- GPUs (T4, V100, A100)
- 1 Gbps network + simulated bandwidth limits

**Production (Full Scale):**
- 50+ workers across multiple regions
- Mixed: datacenters (10 Gbps) + edge (10-100 Mbps)
- Real WAN topology

---

## Test Cases

### Test 1: GPT-2 Small Fine-Tuning

**Model:** GPT-2 Small (124M parameters)
**Dataset:** WikiText-2 or OpenWebText subset
**Goal:** Validate convergence quality matches baseline

**Baseline (Synchronous, No Compression):**
```python
# Standard fine-tuning
# - Synchronous SGD
# - No compression
# - Homogeneous workers
# Expected: Perplexity ~25-30 on WikiText-2
```

**Experimental (Full Features):**
```python
# Distributed training with all features
# - Async + adaptive staleness
# - Gradient compression (top-k 5%)
# - Byzantine defense (Krum/Bulyan)
# - Differential privacy (optional)
# Target: Perplexity within 10% of baseline
```

**Success Criteria:**
- Final perplexity: Within 10% of baseline (e.g., 27.5 if baseline is 25)
- Convergence speed: Similar or faster (due to async)
- Bandwidth: 10-20x reduction from compression
- No divergence or crashes

### Test 2: BERT Base MLM

**Model:** BERT Base (110M parameters)
**Dataset:** BookCorpus or Wikipedia subset
**Task:** Masked Language Modeling
**Goal:** Validate async training doesn't hurt MLM accuracy

**Baseline:**
```
- Batch size: 32 per worker
- Learning rate: 1e-4
- Steps: 100k
- Expected MLM accuracy: 60-65%
```

**Experimental:**
```
- Async + adaptive staleness
- Compression: top-k 10%
- Byzantine defense: Krum (f=2)
- Heterogeneous workers (5x speed range)
```

**Success Criteria:**
- MLM accuracy: Within 5% of baseline
- Training time: Faster with async (utilize all workers)
- Bandwidth: 10x reduction
- Gradient acceptance rate: >80%

### Test 3: Multi-Region WAN Training

**Setup:** Workers distributed across multiple regions/continents
**Model:** GPT-2 Medium (355M parameters)
**Challenge:** High latency (100-300ms), limited bandwidth

**Configuration:**
```yaml
workers:
  - name: us-east
    bandwidth: 1000  # Mbps
    latency: 50      # ms
  - name: eu-west
    bandwidth: 500
    latency: 150
  - name: asia-pacific
    bandwidth: 100
    latency: 300
  - name: edge-device
    bandwidth: 10
    latency: 500
```

**Features:**
```
- Adaptive staleness: base=200, multiplier=20
- Compression: top-k 1% (100x)
- Byzantine defense: Bulyan (f=3)
```

**Success Criteria:**
- Training completes without worker starvation
- Edge devices contribute meaningfully (>10% gradients accepted)
- Bandwidth usage: <10 GB/hour per worker
- Convergence: Within 20% of LAN baseline

---

## Validation Script Template

### `tests/convergence/validate_gpt2_small.py`

```python
"""
GPT-2 Small Fine-Tuning Validation

Validates distributed training system with real GPT-2 model.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from dist_llm_train.controller.main_controller import MainController
from dist_llm_train.sync.parameter_server import ParameterServer, BoundedAsyncCoordinator

def main():
    # Configuration
    config = {
        'model_name': 'gpt2',  # 124M parameters
        'dataset': 'wikitext',
        'dataset_config': 'wikitext-2-raw-v1',
        'num_workers': 5,
        'num_epochs': 3,
        'batch_size': 8,
        'learning_rate': 5e-5,

        # Distributed training features
        'use_async': True,
        'max_staleness': 50,
        'adaptive_staleness': True,

        # Compression
        'compression': 'topk',
        'compression_ratio': 0.05,  # 5% sparsity

        # Byzantine defense
        'aggregation_rule': 'krum',
        'krum_f': 1,  # Tolerate 1 Byzantine worker

        # Differential privacy (optional)
        'gradient_clip_norm': 1.0,
        'enable_dp': False,
    }

    # Load model
    print(f"Loading {config['model_name']}...")
    model = GPT2LMHeadModel.from_pretrained(config['model_name'])
    tokenizer = GPT2Tokenizer.from_pretrained(config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    # Load dataset
    print(f"Loading dataset: {config['dataset']}")
    dataset = load_dataset(config['dataset'], config['dataset_config'])

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    # Baseline training (for comparison)
    print("\n=== Baseline Training (Synchronous, No Compression) ===")
    baseline_args = TrainingArguments(
        output_dir='./baseline_gpt2',
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=100,
    )

    baseline_trainer = Trainer(
        model=model,
        args=baseline_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )

    # Train baseline
    baseline_result = baseline_trainer.train()
    baseline_perplexity = torch.exp(torch.tensor(baseline_result.metrics['eval_loss'])).item()

    print(f"\nBaseline Results:")
    print(f"  Final loss: {baseline_result.metrics['eval_loss']:.4f}")
    print(f"  Perplexity: {baseline_perplexity:.2f}")

    # Distributed training (experimental)
    print("\n=== Distributed Training (Async + Compression + Byzantine Defense) ===")

    # Create parameter server
    initial_state = {name: param.clone().detach() for name, param in model.named_parameters()}
    ps = ParameterServer(initial_model_state=initial_state)

    # Create coordinator
    coordinator = BoundedAsyncCoordinator(
        num_workers=config['num_workers'],
        max_staleness=config['max_staleness'],
        adaptive_staleness=config['adaptive_staleness']
    )

    # Training loop (simplified - full implementation would be more complex)
    # This is a template - actual implementation would need:
    # - Worker processes
    # - Gradient computation and communication
    # - Compression/decompression
    # - Byzantine aggregation
    # - Convergence tracking

    print("\n[NOTE: Full distributed training implementation would go here]")
    print("This template shows the structure - actual implementation requires:")
    print("  1. Worker process management")
    print("  2. Gradient computation loop")
    print("  3. Communication layer integration")
    print("  4. Compression/decompression pipeline")
    print("  5. Byzantine-robust aggregation")

    # Evaluation
    # TODO: Implement distributed training loop
    # TODO: Evaluate final perplexity
    # TODO: Compare with baseline

    print("\n=== Validation Checklist ===")
    print("[ ] Final perplexity within 10% of baseline")
    print("[ ] Training completed without crashes")
    print("[ ] Gradient acceptance rate >80%")
    print("[ ] Bandwidth reduction >10x (from compression)")
    print("[ ] Byzantine workers filtered successfully")

if __name__ == '__main__':
    main()
```

---

## Metrics to Track

### 1. Convergence Quality

| Metric | Baseline | Target (Distributed) | Tolerance |
|--------|----------|---------------------|-----------|
| Final Loss | e.g., 3.2 | e.g., 3.4 | +10% |
| Perplexity (GPT-2) | e.g., 25 | e.g., 27 | +10% |
| MLM Accuracy (BERT) | e.g., 62% | e.g., 59% | -5% |
| Convergence Steps | e.g., 50k | e.g., 45k | ±20% |

### 2. System Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Gradient Acceptance Rate | >80% | With adaptive staleness |
| Bandwidth Reduction | 10-100x | From compression |
| Training Throughput | 1.5-3x | Async utilization |
| Byzantine Filtering | 100% | f < n/3 detected |

### 3. Scalability

| Workers | Speedup | Efficiency | Notes |
|---------|---------|------------|-------|
| 1 | 1.0x | 100% | Baseline |
| 5 | 3.5x | 70% | Communication overhead |
| 10 | 6.0x | 60% | More heterogeneity |
| 50 | 20x | 40% | WAN bottlenecks |

---

## Experiment Tracking

### Using Weights & Biases

```python
import wandb

wandb.init(
    project="distributed-llm-training",
    name="gpt2-async-compressed",
    config={
        "model": "gpt2-small",
        "num_workers": 5,
        "async": True,
        "compression": "topk-5%",
        "byzantine_defense": "krum-f1"
    }
)

# Track metrics
wandb.log({
    "train/loss": loss,
    "train/perplexity": perplexity,
    "system/acceptance_rate": acceptance_rate,
    "system/bandwidth_gb": bandwidth,
    "system/byzantine_filtered": num_byzantine_filtered
})
```

---

## Troubleshooting

### Issue: Low Convergence Quality

**Symptoms:**
- Final perplexity >20% worse than baseline
- Model diverges or gets stuck

**Possible Causes:**
1. Compression ratio too aggressive (try 10% instead of 1%)
2. Byzantine defense filtering too many good gradients (reduce f)
3. Learning rate not tuned for async (try 0.5x-2x baseline LR)
4. Staleness too high (reduce max_staleness)

**Solutions:**
- Start with conservative settings (10% compression, f=1)
- Gradually increase aggressiveness
- Monitor gradient norms and staleness distribution

### Issue: Low Gradient Acceptance

**Symptoms:**
- <50% of gradients accepted
- Slow workers starved

**Possible Causes:**
1. Adaptive staleness not enabled
2. max_staleness too low for heterogeneity
3. Worker speed variance >10x

**Solutions:**
- Enable adaptive_staleness=True
- Increase max_staleness (try 100-200)
- Remove extremely slow workers

### Issue: High Bandwidth Usage

**Symptoms:**
- >100 GB/hour per worker
- Network saturation

**Possible Causes:**
1. Compression not enabled
2. Compression ratio too high (e.g., 50%)
3. Large model (355M+ parameters)

**Solutions:**
- Use top-k compression with 1-5% ratio
- Consider FP16 + top-k combination
- Reduce gradient synchronization frequency

---

## Real-World Deployment Checklist

### Pre-Deployment

- [ ] Validated on GPT-2 Small with <10% perplexity degradation
- [ ] Validated on BERT with <5% MLM accuracy degradation
- [ ] Tested with simulated WAN conditions (bandwidth limits, latency)
- [ ] Byzantine attack tests passed
- [ ] Convergence quality meets requirements

### Deployment

- [ ] Production monitoring in place (Prometheus, Grafana)
- [ ] Logging configured (track rejections, Byzantine detections)
- [ ] Checkpointing enabled (fault tolerance)
- [ ] Bandwidth monitoring per worker
- [ ] Alert thresholds configured

### Post-Deployment

- [ ] Compare final model quality to baseline
- [ ] Analyze bandwidth savings (should be 10-100x)
- [ ] Review staleness distribution (adaptive bounds working)
- [ ] Check Byzantine detection logs (false positives <5%)
- [ ] Measure cost savings (worker utilization, bandwidth)

---

## Expected Results

### GPT-2 Small (124M parameters)

**Baseline (Synchronous, No Compression):**
- Training time: 10 hours (5 workers, GPU)
- Bandwidth: 500 GB total
- Final perplexity: 25-30

**Distributed (Async + Compression + Byzantine Defense):**
- Training time: 6-8 hours (faster async utilization)
- Bandwidth: 25-50 GB total (10-20x reduction)
- Final perplexity: 27-33 (within 10% tolerance)
- Gradient acceptance: 80-90%

**Improvement:**
- 25-40% faster training
- 90-95% bandwidth savings
- Byzantine tolerance (f < n/3)

### BERT Base (110M parameters)

**Baseline:**
- MLM accuracy: 60-65%
- Training time: 100k steps

**Distributed:**
- MLM accuracy: 57-62% (within 5%)
- Training time: 75-90k steps (async speedup)
- Bandwidth: 10x reduction

---

## References

### Papers to Compare Against

1. **Dean et al. (2012)**: "Large Scale Distributed Deep Networks"
   - Async SGD baseline
   - Staleness bounds: 100-300

2. **McMahan et al. (2017)**: "Federated Learning of Deep Networks"
   - Heterogeneous clients
   - Expected convergence degradation: 10-20%

3. **Alistarh et al. (2017)**: "QSGD: Communication-Efficient SGD via Gradient Quantization"
   - Compression: 16-32x with <2% accuracy loss
   - Our target: 10-100x with <10% degradation

4. **Lin et al. (2018)**: "Deep Gradient Compression"
   - Top-k 0.1% (1000x compression)
   - Convergence maintained on ResNet
   - Our target: 1-10% sparsity (10-100x)

---

## Next Steps

### Immediate (This Session - If Time Permits)

1. Create basic GPT-2 validation script
2. Test with small dataset (100 samples)
3. Verify no crashes with real model

### Short-Term (Next Session)

1. Full GPT-2 Small training (WikiText-2)
2. Measure convergence quality vs baseline
3. Analyze bandwidth savings
4. Document results

### Long-Term (Production)

1. BERT Base MLM validation
2. GPT-2 Medium (355M parameters)
3. Multi-region WAN deployment
4. 50+ worker scalability testing

---

**Document End**

*Real LLM validation is the ultimate proof that our system is production-ready. Start with GPT-2 Small, then scale up.*
