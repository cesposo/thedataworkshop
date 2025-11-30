# Documentation Index

Welcome to the dist-llm-train documentation. This index links to concept overviews, deep dives, and how-to guides that explain the current system end-to-end.

## Getting Started

- **Overview and Motivation**: `docs/overview.md` - Project overview, key features, use cases
- **System Status**: `docs/STATUS.md` - Current status, roadmap, test coverage (v2.0 Production Beta)
- **CLI Guide**: `docs/cli_guide.md` - Command-line interface reference
- **Configuration Guide**: `docs/configuration.md` - Complete YAML configuration reference

## Core Concepts

- **System Architecture**: `docs/architecture.md` - Core components, async training flow, v2.0 features
- **Training Protocol**: `docs/training_protocol.md` - Protocol details, adaptive staleness, Byzantine defense, DP
- **Async Training & Compression**: `docs/async_training_and_compression.md` - Technical deep dive on async SGD and gradient compression
- **Scheduler Design**: `docs/scheduler_design.md` - Scheduling algorithms (Gale-Shapley, Priority, Capability)

## Operational Guides

- **Testing & Development**: `docs/testing_and_quality.md` - Test suites, development workflow, CI/CD
- **Telemetry & Observability**: `docs/telemetry_observability.md` - Metrics, monitoring, debugging
- **Persistence & Resume**: `docs/persistence_and_resume.md` - Checkpointing and fault recovery
- **Experiments & Benchmarks**: `docs/experiments.md` - Running experiments, performance benchmarking

## Integration & Implementation

- **Integration Fixes**: `../INTEGRATION_FIXES.md` - ZMQ, gradient accumulation, differential privacy
- **Testing & Demos**: `../TESTING_AND_DEMOS.md` - Demo scripts, benchmarks, validation
- **Implementation Summary**: `../IMPLEMENTATION_SUMMARY.md` - Complete implementation details

## Quick Links

**For Users:**
- Start here: `docs/overview.md` → `docs/configuration.md` → `docs/cli_guide.md`
- Production deployment: `docs/training_protocol.md` + `../ADAPTIVE_STALENESS.md` + `../BYZANTINE_DEFENSE.md`

**For Developers:**
- Architecture: `docs/architecture.md` → `docs/async_training_and_compression.md`
- Testing: `docs/testing_and_quality.md` → `../CONVERGENCE_TESTING.md`
