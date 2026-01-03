import json
import os

def load_config(path="config_real.json"):
    full_path = os.path.abspath(path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)
