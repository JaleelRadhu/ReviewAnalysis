import yaml
from pathlib import Path
import random
import numpy as np

def load_config(path: str = "/home/abdullahm/jaleel/Review_analysis/src/config/default.yaml") -> dict:
    """Load YAML config and return as dictionary."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
