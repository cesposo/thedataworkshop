"""Network impairment (netem) wrapper around a base communicator.

Injects latency, jitter, loss/duplication, bandwidth throttling, and
optional brownout/partition behavior for WAN-like testing.
"""

import random
import time
from typing import Any, Callable, Dict, Optional

from dist_llm_train.communication.communicator import Communicator
from dist_llm_train.communication.rpc import RPCCommunicator


class ChaoticCommunicator(Communicator):
    """Wraps a communicator and injects WAN-like impairments per send."""

    def __init__(
        self,
        base: Communicator,
        profile: Dict[str, Any],
        *,
        seed: Optional[int] = None,
    ):
        self.base = base
        self.profile = profile or {}
        self._rng = random.Random(seed)
        self.stats = {
            "sent": 0,
            "dropped": 0,
            "duplicated": 0,
            "latency_ms": [],
            "bytes": 0,
        }
        self._start_ts = time.time()
        # Expose host/port for compatibility with RPCCommunicator usage
        self.host = getattr(base, "host", None)
        self.port = getattr(base, "port", None)

    # Communicator API
    def register_function(self, function: Callable[..., Any], name: str):
        if hasattr(self.base, "register_function"):
            return self.base.register_function(function, name)
        raise NotImplementedError("Base communicator does not support register_function")

    def start_server(self):
        if hasattr(self.base, "start_server"):
            return self.base.start_server()

    def stop_server(self):
        if hasattr(self.base, "stop_server"):
            return self.base.stop_server()

    def receive(self):
        if hasattr(self.base, "receive"):
            return self.base.receive()

    def broadcast(self, message):
        if hasattr(self.base, "broadcast"):
            return self.base.broadcast(message)

    def send(self, destination, message):
        msg_class = None
        try:
            msg_class = message.get('method')
        except Exception:
            msg_class = None
        profile = self._merged_profile(msg_class)

        # Brownout/partition schedule: drop all during "down" windows
        if self._is_in_brownout(profile):
            self.stats["dropped"] += 1
            return None

        # Loss
        loss_pct = float(profile.get("loss_pct", 0.0))
        if self._rng.random() < loss_pct:
            self.stats["dropped"] += 1
            return None

        # Latency + jitter + bandwidth delay
        latency_ms = float(profile.get("base_rtt_ms", 0.0))
        jitter_ms = float(profile.get("jitter_ms", 0.0))
        if jitter_ms > 0:
            latency_ms += self._rng.uniform(-jitter_ms, jitter_ms)
        latency_ms = max(0.0, latency_ms)

        bw_mbps = profile.get("bandwidth_mbps")
        size_bytes = len(str(message)) if message is not None else 0
        if bw_mbps and bw_mbps > 0:
            # simple throughput delay approximation
            bw_delay_ms = (size_bytes * 8) / (bw_mbps * 1_000_000) * 1000.0
            latency_ms += bw_delay_ms

        if latency_ms > 0:
            time.sleep(latency_ms / 1000.0)

        self.stats["sent"] += 1
        self.stats["latency_ms"].append(latency_ms)
        self.stats["bytes"] += size_bytes

        result = self.base.send(destination, message)

        # Duplication (after first delivery)
        dup_pct = float(profile.get("dup_pct", 0.0))
        if dup_pct > 0 and self._rng.random() < dup_pct:
            self.stats["duplicated"] += 1
            _ = self.base.send(destination, message)

        return result

    # Helpers
    def _is_in_brownout(self, profile: Dict[str, Any]) -> bool:
        cycle = profile.get("brownout_cycle")
        if not cycle:
            return False
        up_s = float(cycle.get("up_s", 0.0))
        down_s = float(cycle.get("down_s", 0.0))
        if down_s <= 0:
            return False
        period = up_s + down_s
        if period <= 0:
            return False
        elapsed = time.time() - self._start_ts
        phase = elapsed % period
        return phase > up_s

    def _merged_profile(self, msg_class: Optional[str]) -> Dict[str, Any]:
        """Merge global profile with class-specific override."""
        base = dict(self.profile or {})
        overrides = base.get("overrides", {})
        if msg_class and msg_class in overrides:
            merged = {k: v for k, v in base.items() if k != "overrides"}
            merged.update(overrides[msg_class] or {})
            return merged
        return base


PRESET_PROFILES: Dict[str, Dict[str, Any]] = {
    "good": {"base_rtt_ms": 20, "jitter_ms": 5, "loss_pct": 0.001, "bandwidth_mbps": 100},
    "cellular": {"base_rtt_ms": 80, "jitter_ms": 40, "loss_pct": 0.02, "bandwidth_mbps": 10},
    "satellite": {"base_rtt_ms": 400, "jitter_ms": 100, "loss_pct": 0.01, "bandwidth_mbps": 5},
    "degraded": {"base_rtt_ms": 200, "jitter_ms": 80, "loss_pct": 0.1, "dup_pct": 0.03, "bandwidth_mbps": 2},
    "brownout": {
        "base_rtt_ms": 80,
        "jitter_ms": 40,
        "loss_pct": 0.05,
        "bandwidth_mbps": 5,
        "brownout_cycle": {"up_s": 5, "down_s": 2},
    },
}


def load_profile(name_or_dict: Any) -> Dict[str, Any]:
    """Return a normalized profile dict from preset name or dict."""
    if isinstance(name_or_dict, str):
        return dict(PRESET_PROFILES.get(name_or_dict, {}))
    return dict(name_or_dict or {})


def build_communicator(
    host: str,
    port: int,
    netem_profile: Optional[Any] = None,
    *,
    base_cls: Callable[..., Communicator] = RPCCommunicator,
    seed: Optional[int] = None,
) -> Communicator:
    """Factory that builds a (possibly chaotic) communicator."""
    base = base_cls(host, port)
    if netem_profile is None:
        return base
    profile = load_profile(netem_profile)
    return ChaoticCommunicator(base, profile, seed=seed)
