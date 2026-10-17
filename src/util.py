import json
import os
import random
import warnings

import numpy as np
import torch
from scipy.stats import ConstantInputWarning, pearsonr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def pick_device(device_str="cuda"):
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check CUDA_VISIBLE_DEVICES, or pass --device cpu.")
    return torch.device(device_str)


def norm_id(x):
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pcc = np.nan
    if len(y_true) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            pcc = float(pearsonr(y_true, y_pred)[0])
    return {"RMSE": rmse, "PCC": pcc}
