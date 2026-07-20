import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from omni.trainers.vlm.full_sft import main

if __name__ == "__main__":
    default_config = os.path.join(ROOT, "configs", "model", "vlm.yaml")
    main(default_config=default_config)
