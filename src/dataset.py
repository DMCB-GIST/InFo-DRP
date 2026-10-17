import os

import numpy as np
import pandas as pd
import torch

from util import norm_id

DRUG_BRANCHES = ["molbert", "chemformer", "ecfp4"]


def build_dataset_config(repo_root, dataset):
    data_root = os.path.join(repo_root, "data")

    if dataset == "gdscv1":
        proc = os.path.join(data_root, "gdscv1", "processed")
        return {
            "name": "gdscv1",
            "cell_feat_path": os.path.join(proc, "gdsc1_cell734_expr_426.csv"),
            "cell_key": "COSMIC_ID",
            "drug_paths": {
                "molbert": os.path.join(proc, "gdsc1_drug177_molbert_768.csv"),
                "chemformer": os.path.join(proc, "gdsc1_drug177_chemformer_1024.csv"),
                "ecfp4": os.path.join(proc, "gdsc1_drug177_ecfp4_2048.csv"),
            },
            "desc_path": os.path.join(proc, "gdsc1_drug177_descriptor_280.csv"),
            "pair_cell_col": "COSMIC_ID",
            "pair_drug_col": "PubChemID",
            "pair_target_col": "LN_IC50",
            "splits_root": os.path.join(data_root, "gdscv1", "splits"),
            "mode_map": {"newPAIR": "mix", "newCELL": "cb", "newDRUG": "db"},
            "num_folds": 5,
            "test_is_global": True,
        }

    if dataset == "gdscv2":
        proc = os.path.join(data_root, "gdscv2", "processed")
        return {
            "name": "gdscv2",
            "cell_feat_path": os.path.join(proc, "gdsc2_cell941_expr_699.csv"),
            "cell_key": "CELL_LINE_NAME",
            "drug_paths": {
                "molbert": os.path.join(proc, "gdsc2_drug222_molbert_768.csv"),
                "chemformer": os.path.join(proc, "gdsc2_drug222_chemformer_1024.csv"),
                "ecfp4": os.path.join(proc, "gdsc2_drug222_ecfp4_2048.csv"),
            },
            "desc_path": os.path.join(proc, "gdsc2_drug222_descriptor_280.csv"),
            "pair_cell_col": "CELL_LINE_NAME",
            "pair_drug_col": "PubChemID",
            "pair_target_col": "LN_IC50",
            "splits_root": os.path.join(data_root, "gdscv2", "splits"),
            "mode_map": {"newSCAFFOLD": "sb", "newDRUG": "db", "newCELL": "cb"},
            "num_folds": 10,
            "test_is_global": False,
        }

    if dataset == "ccle":
        # Train on GDSCv2 (scaffolds of all CCLE test drugs removed), test on CCLE.
        # CCLE cell lines are represented by their GDSCv2 expression profiles.
        proc = os.path.join(data_root, "ccle", "processed")
        return {
            "name": "ccle",
            "cell_feat_path": os.path.join(data_root, "gdscv2", "processed", "gdsc2_cell941_expr_699.csv"),
            "cell_key": "CELL_LINE_NAME",
            "drug_paths": {
                "molbert": os.path.join(proc, "gdsc2_ccle_drug232_molbert_768.csv"),
                "chemformer": os.path.join(proc, "gdsc2_ccle_drug232_chemformer_1024.csv"),
                "ecfp4": os.path.join(proc, "gdsc2_ccle_drug232_ecfp4_2048.csv"),
            },
            "desc_path": os.path.join(proc, "gdsc2_ccle_drug232_descriptor_280.csv"),
            "pair_cell_col": "CELL_LINE_NAME",
            "pair_drug_col": "PubChemID",
            "pair_target_col": "LN_IC50",
            "splits_root": os.path.join(data_root, "ccle", "splits"),
            "mode_map": {"newSCAFFOLD": "sb"},
            "num_folds": 1,
            "test_is_global": True,
        }

    raise ValueError("dataset must be one of: gdscv1, gdscv2, ccle")


def _read_table(path, key):
    df = pd.read_csv(path, dtype={key: str})
    df[key] = df[key].map(norm_id)
    return df.set_index(key)


def load_features(cfg):
    df_cell = _read_table(cfg["cell_feat_path"], cfg["cell_key"])
    df_drugs = {b: _read_table(cfg["drug_paths"][b], "PubChemID") for b in DRUG_BRANCHES}
    df_desc = _read_table(cfg["desc_path"], "PubChemID")
    return df_cell, df_drugs, df_desc


def load_fold_pairs(cfg, split_mode, fold):
    if split_mode not in cfg["mode_map"]:
        raise ValueError(f"Invalid split_mode={split_mode} for {cfg['name']}; allowed={list(cfg['mode_map'])}")

    base = os.path.join(cfg["splits_root"], cfg["mode_map"][split_mode])
    fold_dir = os.path.join(base, f"fold{fold}")
    test_dir = base if cfg["test_is_global"] else fold_dir

    dfs = []
    for path in (os.path.join(fold_dir, "train_pairs.csv"),
                 os.path.join(fold_dir, "val_pairs.csv"),
                 os.path.join(test_dir, "test_pairs.csv")):
        df = pd.read_csv(path)
        df[cfg["pair_cell_col"]] = df[cfg["pair_cell_col"]].map(norm_id)
        df[cfg["pair_drug_col"]] = df[cfg["pair_drug_col"]].map(norm_id)
        dfs.append(df)
    return tuple(dfs)


def filter_pairs(df_pair, cfg, df_cell, df_drugs, df_desc):
    drug_ids = set(df_desc.index)
    for b in DRUG_BRANCHES:
        drug_ids &= set(df_drugs[b].index)
    keep = df_pair[cfg["pair_cell_col"]].isin(df_cell.index) & df_pair[cfg["pair_drug_col"]].isin(drug_ids)
    return df_pair[keep].reset_index(drop=True)


class DrugResponseDataset(torch.utils.data.Dataset):
    """Each sample: [MolBERT|desc, Chemformer|desc, ECFP4|desc, cell expression] -> ln(IC50)."""

    def __init__(self, df_pair, cfg, df_cell, df_drugs, df_desc):
        drug_ids = sorted(set(df_pair[cfg["pair_drug_col"]]))
        cell_ids = sorted(set(df_pair[cfg["pair_cell_col"]]))
        desc = df_desc.loc[drug_ids].values.astype(np.float32)
        drug_feat = np.concatenate(
            [np.concatenate([df_drugs[b].loc[drug_ids].values.astype(np.float32), desc], axis=1) for b in DRUG_BRANCHES],
            axis=1,
        )
        self.drug_feat = torch.from_numpy(drug_feat)
        self.cell_feat = torch.from_numpy(df_cell.loc[cell_ids].values.astype(np.float32))

        drug_pos = {d: i for i, d in enumerate(drug_ids)}
        cell_pos = {c: i for i, c in enumerate(cell_ids)}
        self.drug_idx = torch.tensor([drug_pos[d] for d in df_pair[cfg["pair_drug_col"]]])
        self.cell_idx = torch.tensor([cell_pos[c] for c in df_pair[cfg["pair_cell_col"]]])
        self.y = torch.tensor(df_pair[cfg["pair_target_col"]].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        idx = torch.as_tensor(idx)
        x = torch.cat([self.drug_feat[self.drug_idx[idx]], self.cell_feat[self.cell_idx[idx]]], dim=1)
        return x, self.y[idx]


def make_loader(ds, batch_size, shuffle):
    """Mini-batch loader (the dataset is indexed with a list of sample indices)."""
    sampler = torch.utils.data.RandomSampler(ds) if shuffle else torch.utils.data.SequentialSampler(ds)
    batches = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=shuffle)
    return torch.utils.data.DataLoader(ds, batch_size=None, sampler=batches)
