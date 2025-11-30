"""DNN scenario definitions with progressive complexity.

Defines training scenarios ranging from simple DNNs to LLM-scale models,
each with different characteristics relevant to WAN distributed training.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any

# Make PyTorch optional for simulations
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    # Mock nn.Module for when PyTorch is not available
    TORCH_AVAILABLE = False

    class MockModule:
        """Mock PyTorch module for simulations without PyTorch."""
        pass

    class MockNN:
        Module = MockModule

    nn = MockNN()
    torch = None


@dataclass
class ScenarioConfig:
    """Configuration for a training scenario."""

    name: str
    description: str

    # Model characteristics
    model_factory: Callable[[], Any]
    param_count: int
    gradient_size_mb: float

    # Training characteristics
    batch_size: int
    num_batches: int
    input_shape: tuple
    num_classes: int

    # Performance characteristics
    estimated_step_time_s: float  # On reference GPU
    estimated_cpu_slowdown: float = 10.0  # CPU is 10x slower

    # Difficulty level (1-5)
    complexity: int = 1


# ============================================================================
# SCENARIO 1: SIMPLE DNN (MNIST-style)
# ============================================================================

def create_simple_dnn() -> Any:
    """Create a simple 2-layer DNN for MNIST-style tasks.

    Returns:
        Small DNN (~100K parameters) or mock for simulation
    """
    if not TORCH_AVAILABLE:
        return MockModule()  # Return mock for simulation without PyTorch

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )


SIMPLE_DNN = ScenarioConfig(
    name="simple_dnn",
    description="Simple 2-layer DNN (MNIST-style) - Baseline scenario",

    model_factory=create_simple_dnn,
    param_count=235_146,  # ~235K parameters
    gradient_size_mb=0.9,  # ~0.9 MB in FP32

    batch_size=64,
    num_batches=1000,
    input_shape=(1, 28, 28),
    num_classes=10,

    estimated_step_time_s=0.05,  # 50ms on GPU
    estimated_cpu_slowdown=8.0,

    complexity=1
)


# ============================================================================
# SCENARIO 2: MODERATE DNN (CIFAR-10 style)
# ============================================================================

def create_moderate_dnn() -> Any:
    """Create a moderate CNN for CIFAR-10 style tasks.

    Returns:
        Medium CNN (~600K parameters) or mock for simulation
    """
    if not TORCH_AVAILABLE:
        return MockModule()  # Return mock for simulation without PyTorch

    return nn.Sequential(
        # Conv block 1
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.2),

        # Conv block 2
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.3),

        # Classifier
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 10)
    )


MODERATE_DNN = ScenarioConfig(
    name="moderate_dnn",
    description="Moderate CNN (CIFAR-10 style) - Tests compression effectiveness",

    model_factory=create_moderate_dnn,
    param_count=638_090,  # ~638K parameters
    gradient_size_mb=2.43,  # ~2.4 MB in FP32

    batch_size=128,
    num_batches=500,
    input_shape=(3, 32, 32),
    num_classes=10,

    estimated_step_time_s=0.15,  # 150ms on GPU
    estimated_cpu_slowdown=10.0,

    complexity=2
)


# ============================================================================
# SCENARIO 3: LARGE DNN (Small ResNet style)
# ============================================================================

if TORCH_AVAILABLE:
    class ResidualBlock(nn.Module):
        """Basic residual block."""

        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = torch.relu(out)
            return out


def create_large_dnn() -> Any:
    """Create a ResNet-18 style network.

    Returns:
        ResNet-18 (~11M parameters) or mock for simulation
    """
    if not TORCH_AVAILABLE:
        return MockModule()  # Return mock for simulation without PyTorch

    class SmallResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(3, 2, 1)

            self.layer1 = self._make_layer(64, 64, 2, stride=1)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 1000)

        def _make_layer(self, in_channels, out_channels, num_blocks, stride):
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    return SmallResNet()


LARGE_DNN = ScenarioConfig(
    name="large_dnn",
    description="Large ResNet-18 style - Tests async training benefits",

    model_factory=create_large_dnn,
    param_count=11_689_512,  # ~11.7M parameters
    gradient_size_mb=44.6,  # ~44.6 MB in FP32

    batch_size=256,
    num_batches=200,
    input_shape=(3, 224, 224),
    num_classes=1000,

    estimated_step_time_s=0.5,  # 500ms on GPU
    estimated_cpu_slowdown=15.0,

    complexity=3
)


# ============================================================================
# SCENARIO 4: VERY LARGE DNN (Approaching LLM scale)
# ============================================================================

def create_very_large_dnn() -> Any:
    """Create a very large transformer-style network.

    Returns:
        Large transformer (~100M parameters) or mock for simulation
    """
    if not TORCH_AVAILABLE:
        return MockModule()  # Return mock for simulation without PyTorch

    class LargeTransformer(nn.Module):
        def __init__(self, vocab_size=50000, d_model=1024, nhead=16,
                     num_layers=12, dim_feedforward=4096, max_seq_len=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Embedding(max_seq_len, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.fc = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            # x shape: (batch, seq_len)
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = self.embedding(x) + self.pos_encoding(positions)
            x = self.transformer(x)
            x = self.fc(x)
            return x

    return LargeTransformer()


VERY_LARGE_DNN = ScenarioConfig(
    name="very_large_dnn",
    description="Very large transformer (~100M params) - Real-world WAN challenges",

    model_factory=create_very_large_dnn,
    param_count=102_453_000,  # ~102M parameters
    gradient_size_mb=390.6,  # ~391 MB in FP32

    batch_size=32,
    num_batches=100,
    input_shape=(512,),  # Sequence length
    num_classes=50000,  # Vocabulary size

    estimated_step_time_s=2.0,  # 2s on GPU
    estimated_cpu_slowdown=20.0,

    complexity=4
)


# ============================================================================
# SCENARIO 5: LLM SCALE (1B parameters)
# ============================================================================

def create_llm_scale() -> Any:
    """Create a 1B parameter LLM-scale network.

    Returns:
        Very large transformer (~1B parameters) or mock for simulation
    """
    if not TORCH_AVAILABLE:
        return MockModule()  # Return mock for simulation without PyTorch

    class LLMScale(nn.Module):
        def __init__(self, vocab_size=50000, d_model=2048, nhead=32,
                     num_layers=24, dim_feedforward=8192, max_seq_len=1024):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Embedding(max_seq_len, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.fc = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = self.embedding(x) + self.pos_encoding(positions)
            x = self.transformer(x)
            x = self.fc(x)
            return x

    return LLMScale()


LLM_SCALE = ScenarioConfig(
    name="llm_scale",
    description="LLM scale (~1B params) - Ultimate WAN challenge",

    model_factory=create_llm_scale,
    param_count=1_037_352_000,  # ~1B parameters
    gradient_size_mb=3955.6,  # ~3.96 GB in FP32

    batch_size=16,
    num_batches=50,
    input_shape=(1024,),  # Sequence length
    num_classes=50000,  # Vocabulary size

    estimated_step_time_s=10.0,  # 10s on GPU
    estimated_cpu_slowdown=25.0,

    complexity=5
)


# ============================================================================
# SCENARIO REGISTRY
# ============================================================================

ALL_SCENARIOS = {
    'simple': SIMPLE_DNN,
    'moderate': MODERATE_DNN,
    'large': LARGE_DNN,
    'very_large': VERY_LARGE_DNN,
    'llm_scale': LLM_SCALE
}


def get_scenario(name: str) -> ScenarioConfig:
    """Get a scenario configuration by name.

    Args:
        name: Scenario name ('simple', 'moderate', 'large', 'very_large', 'llm_scale')

    Returns:
        Scenario configuration

    Raises:
        ValueError: If scenario name not found
    """
    if name not in ALL_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{name}'. "
            f"Available: {list(ALL_SCENARIOS.keys())}"
        )
    return ALL_SCENARIOS[name]


def list_scenarios() -> Dict[str, ScenarioConfig]:
    """Get all available scenarios.

    Returns:
        Dictionary mapping scenario names to configurations
    """
    return ALL_SCENARIOS.copy()


def print_scenario_summary():
    """Print a summary of all scenarios."""
    print("=" * 100)
    print("DNN SCENARIO SUMMARY")
    print("=" * 100)
    print(f"{'Scenario':<15} {'Params':<12} {'Grad Size':<12} {'Batch':<8} {'Step Time':<12} {'Complexity':<10}")
    print("-" * 100)

    for name, config in ALL_SCENARIOS.items():
        params_str = f"{config.param_count / 1e6:.1f}M" if config.param_count >= 1e6 else f"{config.param_count / 1e3:.1f}K"
        grad_str = f"{config.gradient_size_mb:.1f} MB"
        step_str = f"{config.estimated_step_time_s:.2f}s"

        print(f"{config.name:<15} {params_str:<12} {grad_str:<12} {config.batch_size:<8} {step_str:<12} {'⭐' * config.complexity:<10}")

    print("=" * 100)


if __name__ == "__main__":
    # Print scenario summary
    print_scenario_summary()

    print("\n" + "=" * 100)
    print("BANDWIDTH REQUIREMENTS (per step)")
    print("=" * 100)

    for name, config in ALL_SCENARIOS.items():
        print(f"\n{config.name.upper()}:")
        print(f"  Gradient size: {config.gradient_size_mb:.2f} MB")
        print(f"  Transmission time at 50 Mbps: {config.gradient_size_mb * 8 / 50:.1f}s")
        print(f"  Transmission time at 10 Mbps: {config.gradient_size_mb * 8 / 10:.1f}s")
        print(f"  With top-k 1% compression: {config.gradient_size_mb * 0.01:.2f} MB")
        print(f"  Compressed transmission at 50 Mbps: {config.gradient_size_mb * 0.01 * 8 / 50:.2f}s")
