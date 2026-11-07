import os

import yaml


def load_config(path, **path_overrides):
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base = os.path.dirname(path)
    for key, value in config["paths"].items():
        if not os.path.isabs(value):
            config["paths"][key] = os.path.normpath(os.path.join(base, value))
    for key, value in path_overrides.items():
        if value is not None:
            config["paths"][key] = os.path.abspath(value)
    config["config_path"] = path
    return config
