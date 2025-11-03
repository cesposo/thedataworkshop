import json
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON config file into a dict."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    path_lower = path.lower()
    if path_lower.endswith(('.yaml', '.yml')):
        if yaml is None:
            raise RuntimeError("pyyaml not installed; cannot parse YAML")
        return yaml.safe_load(content) or {}
    else:
        return json.loads(content)

