import os
import yaml


def load_config(config_path=None, _visited=None):
    if _visited is None:
        _visited = set()

    if config_path is None:
        default_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        config_path = os.getenv("CONFIG_PATH", default_path)

    config_path = os.path.abspath(config_path)

    if config_path in _visited:
        raise ValueError(f"Recursive inclusion detected: {config_path}")
    _visited.add(config_path)

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        def resolve_includes(node, base_path):
            if isinstance(node, dict):
                return {
                    k: resolve_includes(v, base_path)
                    for k, v in node.items()
                }
            elif isinstance(node, list):
                return [resolve_includes(i, base_path) for i in node]
            elif isinstance(node, str) and node.endswith(".yaml"):
                include_path = os.path.join(base_path, node)
                include_path = os.path.abspath(include_path)
                return load_config(include_path, _visited)
            else:
                return node

        base_path = os.path.dirname(config_path)
        return resolve_includes(data, base_path)

    finally:
        _visited.remove(config_path)


config = load_config()
