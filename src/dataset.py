# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch

from util import norm_id


def build_dataset_config(repo_root, dataset):
    dataset = dataset.lower()
    if dataset not in ["gdscv1", "gdscv2"]:
        raise ValueError("dataset must be one of: gdscv1, gdscv2")

    data_root = os.path.join(repo_root, "data", dataset, "processed")
    splits_root = os.path.join(repo_root, "data", dataset, "splits")

    if dataset == "gdscv1":
        cfg = {
            "name": "gdscv1",
            "cell_feat_path": os.path.join(data_root, "GDSC1_734cell_expr_426_CELLWISE_z.csv"),
            "cell_key": "CELL_LINE",
            "cell_dim": 426,

            "drug_paths": {
                "molbert": os.path.join(data_root, "gdsc1_drug177_canonical_MolBERT_768dim.csv"),
                "chemformer": os.path.join(data_root, "gdsc1_drug177_canonical_chemformer_large_1024dim.csv"),
                "ecfp4": os.path.join(data_root, "gdsc1_drug177_canonical_Fingerprint_Morgan_ECFP4_2048bit.csv"),
            },
            "drug_key": "PubChemID",

            "desc_path": os.path.join(data_root, "gdsc1_drug177_canonical_Descriptor_DRUGWISE_RobustPower_280dim.csv"),
            "desc_key": "PubChemID",
            "desc_dim": 280,

            "pair_cell_col": "cell_id",
            "pair_drug_col": "drug_id",
            "pair_target_col": "ic50",

            "splits_root": splits_root,
            "mode_map": {"newPAIR": "mix", "newCELL": "cb", "newDRUG": "db"},
            "num_folds": 5,
            "test_is_global": True,
        }
        return cfg

    # gdscv2
    cfg = {
        "name": "gdscv2",
        "cell_feat_path": os.path.join(data_root, "GDSC2_941cell_expr_699_CELLWISE_z.csv"),
        "cell_key": "CELL_LINE_NAME",
        "cell_dim": 699,

        "drug_paths": {
            "molbert": os.path.join(data_root, "gdsc2_drug222_canonical_MolBERT_768dim.csv"),
            "chemformer": os.path.join(data_root, "gdsc2_drug222_canonical_chemformer_large_1024dim.csv"),
            "ecfp4": os.path.join(data_root, "gdsc2_drug222_canonical_Fingerprint_Morgan_ECFP4_2048bit.csv"),
        },
        "drug_key": "PubChemID",

        "desc_path": os.path.join(data_root, "gdsc2_drug222_canonical_Descriptor_DRUGWISE_RobustPower_280dim.csv"),
        "desc_key": "PubChemID",
        "desc_dim": 280,

        "pair_cell_col": "CELL_LINE_NAME",
        "pair_drug_col": "PubChemID",
        "pair_target_col": "LN_IC50",

        "splits_root": splits_root,
        "mode_map": {"newCELL": "cb", "newDRUG": "db", "newSCAFFOLD": "sb"},
        "num_folds": 10,
        "test_is_global": False,
    }
    return cfg


def _read_csv_str(path, key):
    df = pd.read_csv(path, dtype={key: str})
    df[key] = df[key].map(norm_id)
    return df


def load_embeddings(cfg):
    df_cell = _read_csv_str(cfg["cell_feat_path"], cfg["cell_key"])

    df_drug_dict = {}
    for k, p in cfg["drug_paths"].items():
        df_drug_dict[k] = _read_csv_str(p, cfg["drug_key"])

    df_desc = _read_csv_str(cfg["desc_path"], cfg["desc_key"])
    return df_cell, df_drug_dict, df_desc


def load_fold_pairs(cfg, split_mode, fold_idx0):
    if split_mode not in cfg["mode_map"]:
        raise ValueError("Invalid split_mode=%s, allowed=%s" % (split_mode, list(cfg["mode_map"].keys())))

    mode_dir = cfg["mode_map"][split_mode]
    base = os.path.join(cfg["splits_root"], mode_dir)
    fold_dir = os.path.join(base, "fold%d" % fold_idx0)

    train_path = os.path.join(fold_dir, "train_pairs.csv")
    val_path = os.path.join(fold_dir, "val_pairs.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError("Missing train_pairs.csv: %s" % train_path)
    if not os.path.exists(val_path):
        raise FileNotFoundError("Missing val_pairs.csv: %s" % val_path)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    if cfg["test_is_global"]:
        test_path = os.path.join(base, "test_pairs.csv")
    else:
        test_path = os.path.join(fold_dir, "test_pairs.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError("Missing test_pairs.csv: %s" % test_path)

    test_df = pd.read_csv(test_path)

    for df in [train_df, val_df, test_df]:
        df[cfg["pair_cell_col"]] = df[cfg["pair_cell_col"]].map(norm_id).astype(str)
        df[cfg["pair_drug_col"]] = df[cfg["pair_drug_col"]].map(norm_id).astype(str)

    return train_df, val_df, test_df


def filter_pairs_to_embeddings(df_pair, cfg, df_cell, df_drug_dict, df_desc):
    cell_set = set(df_cell[cfg["cell_key"]].astype(str))

    drug_sets = []
    for name in df_drug_dict.keys():
        drug_sets.append(set(df_drug_dict[name][cfg["drug_key"]].astype(str)))
    drug_sets.append(set(df_desc[cfg["desc_key"]].astype(str)))
    drug_set = set.intersection(*drug_sets)

    keep = df_pair[cfg["pair_cell_col"]].astype(str).isin(cell_set) & df_pair[cfg["pair_drug_col"]].astype(str).isin(drug_set)
    return df_pair[keep].copy()


class DrugResponseDatasetMulti(torch.utils.data.Dataset):
    def __init__(self, df_pair, cfg, df_drug_dict, df_cell, df_desc):
        super().__init__()
        self.df_pair = df_pair.reset_index(drop=True)
        self.cfg = cfg

        self.branch_names = ["molbert", "chemformer", "ecfp4"]

        # Drug embeddings cache
        self.drug_cache = {}
        for b in self.branch_names:
            dfd = df_drug_dict[b]
            tmp = {}
            for _, row in dfd.iterrows():
                pid = str(row[cfg["drug_key"]])
                tmp[pid] = row.iloc[1:].values.astype(np.float32)
            self.drug_cache[b] = tmp

        # Descriptor cache
        self.desc_cache = {}
        for _, row in df_desc.iterrows():
            pid = str(row[cfg["desc_key"]])
            self.desc_cache[pid] = row.iloc[1:].values.astype(np.float32)
        self.desc_dim = df_desc.shape[1] - 1

        # Cell cache
        self.cell_cache = {}
        for _, row in df_cell.iterrows():
            cid = str(row[cfg["cell_key"]])
            self.cell_cache[cid] = row.iloc[1:].values.astype(np.float32)

        self.cell_dim = cfg["cell_dim"]

    def __len__(self):
        return len(self.df_pair)

    def __getitem__(self, idx):
        row = self.df_pair.iloc[idx]
        cell_id = str(row[self.cfg["pair_cell_col"]])
        drug_id = str(row[self.cfg["pair_drug_col"]])
        y_val = float(row[self.cfg["pair_target_col"]])

        parts = []
        for b in self.branch_names:
            d = self.drug_cache[b][drug_id]
            desc = self.desc_cache[drug_id]
            parts.append(np.concatenate([d, desc], axis=0))
        parts.append(self.cell_cache[cell_id])

        x = np.concatenate(parts, axis=0)
        X = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y_val, dtype=torch.float32)
        return X, y
