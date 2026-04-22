# -*- coding: utf-8 -*-
import os
import json
import time
import random
import warnings

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import ConstantInputWarning


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
        json.dump(obj, f, indent=2, ensure_ascii=False)


def now_tag():
    return time.strftime("%Y%m%d-%H%M%S")


def pick_device(device_str="cuda"):
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    if device_str.startswith("cuda"):
        print("[WARN] CUDA not available, falling back to CPU.")
    return torch.device("cpu")


def norm_id(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s
    except Exception:
        return str(x)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    if len(y_true) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            pcc = pearsonr(y_true, y_pred)[0]
    else:
        pcc = np.nan

    return {"RMSE": rmse, "PCC": float(pcc) if pcc is not None else np.nan}
