# -*- coding: utf-8 -*-
import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from util import set_seed, pick_device, ensure_dir, save_json, compute_metrics
from dataset import (
    build_dataset_config,
    load_embeddings,
    load_fold_pairs,
    filter_pairs_to_embeddings,
    DrugResponseDatasetMulti,
)
from model import (
    InFoDRP,
    compute_invariant_loss,
    compute_score_regularization,
)


def build_optimizer(model, lr, weight_decay=1e-4):
    decay, no_decay = [], []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            is_norm = isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm))
            is_embed = isinstance(module, (nn.Embedding,))
            is_bias = param_name.endswith("bias")
            is_scorer = "scorer" in module_name

            if is_norm or is_embed or is_bias or is_scorer:
                no_decay.append(param)
            else:
                decay.append(param)

    return optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


class EarlyStopping:
    def __init__(self, patience=10, mode="min"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best = None
        self.early_stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            self.counter = 0
            return True  # first time treated as improved
        improved = (metric < self.best) if self.mode == "min" else (metric > self.best)
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved


def train_one_epoch(model, loader, optimizer, device, lam_inv, lam_reg, lam_cmt):
    model.train()
    criterion = nn.MSELoss()

    preds, trues = [], []
    mse_sum = 0.0
    total = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out, z_inv_list, z_spu_list, S_list, cmt_list, cb_list, z_inv_c, z_spu_c, S_c, cmt_c, cb_c = model.forward_invariant(X)

        loss_pred = criterion(out, y)

        inv_losses, reg_losses, cmt_losses = [], [], []
        for i, name in enumerate(model.drug_names):
            inv_losses.append(compute_invariant_loss(z_inv_list[i], z_spu_list[i], model.proj_mlps_drug[name]))
            reg_losses.append(compute_score_regularization(S_list[i], model.gamma_drug))
            cmt_losses.append(cmt_list[i] + cb_list[i])

        loss_inv_d = torch.stack(inv_losses).mean()
        loss_reg_d = torch.stack(reg_losses).mean()
        loss_cmt_d = torch.stack(cmt_losses).mean()

        loss_inv_c = compute_invariant_loss(z_inv_c, z_spu_c, model.proj_mlp_cell)
        loss_reg_c = compute_score_regularization(S_c, model.gamma_cell)
        loss_cmt_c = cmt_c + cb_c

        loss = loss_pred + lam_inv * (loss_inv_d + loss_inv_c) + lam_reg * (loss_reg_d + loss_reg_c) + lam_cmt * (loss_cmt_d + loss_cmt_c)
        loss.backward()
        optimizer.step()

        bs = X.size(0)
        mse_sum += loss_pred.item() * bs
        total += bs

        preds.extend(out.detach().cpu().numpy().tolist())
        trues.extend(y.detach().cpu().numpy().tolist())

    train_mse = float(mse_sum / max(total, 1))
    metrics = compute_metrics(trues, preds)
    return train_mse, metrics["PCC"]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        preds.extend(out.detach().cpu().numpy().tolist())
        trues.extend(y.detach().cpu().numpy().tolist())
    return preds, trues


def run_fold(cfg, df_cell, df_drug_dict, df_desc, split_mode, fold_idx0, device, args, out_ckpt_dir, out_res_dir):
    train_df, val_df, test_df = load_fold_pairs(cfg, split_mode, fold_idx0)

    train_df = filter_pairs_to_embeddings(train_df, cfg, df_cell, df_drug_dict, df_desc)
    val_df = filter_pairs_to_embeddings(val_df, cfg, df_cell, df_drug_dict, df_desc)
    test_df = filter_pairs_to_embeddings(test_df, cfg, df_cell, df_drug_dict, df_desc)

    train_ds = DrugResponseDatasetMulti(train_df, cfg, df_drug_dict, df_cell, df_desc)
    val_ds = DrugResponseDatasetMulti(val_df, cfg, df_drug_dict, df_cell, df_desc)
    test_ds = DrugResponseDatasetMulti(test_df, cfg, df_drug_dict, df_cell, df_desc)

    train_loader_shuf = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader_eval = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = InFoDRP(
        cell_dim=cfg["cell_dim"],
        desc_dim=cfg["desc_dim"],
        dropout_rate=args.dropout,
        codebook_size=args.codebook_size,
        gamma_drug=args.gamma_drug,
        gamma_cell=args.gamma_cell,
    ).to(device)

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=args.patience, mode="min")

    best_val_mse = float("inf")
    best_epoch = 0
    ckpt_path = os.path.join(out_ckpt_dir, "fold%d_best.pt" % fold_idx0)

    for epoch in tqdm(range(1, args.max_epochs + 1), desc="[%s %s fold%d]" % (cfg["name"], split_mode, fold_idx0)):
        tr_mse, tr_pcc = train_one_epoch(
            model, train_loader_shuf, optimizer, device,
            lam_inv=args.lam_inv,
            lam_reg=args.lam_reg,
            lam_cmt=args.lam_cmt,
        )

        val_preds, val_trues = predict(model, val_loader, device)
        val_mse = float(np.mean((np.asarray(val_trues) - np.asarray(val_preds)) ** 2))
        
        # compute val RMSE/PCC for monitoring
        val_metrics = compute_metrics(val_trues, val_preds)
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "val_mse": best_val_mse,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "meta": {
                        "dataset": cfg["name"],
                        "split_mode": split_mode,
                        "fold0": fold_idx0,
                        "seed": args.seed,
                    },
                    "hparams": vars(args),
                },
                ckpt_path
            )

            # log best save
            print("[BEST] fold=%d epoch=%d | model saved" % (fold_idx0, epoch))

        # log every epoch (val only)
        print(
            "[VAL] fold=%d epoch=%d | val_rmse=%.4f | es=%d/%d"
            % (
                fold_idx0, epoch,
                val_metrics["RMSE"],
                stopper.counter, stopper.patience
            )
        )

        
        improved = stopper(val_mse)
        if stopper.early_stop:
            print("[EARLY STOP] fold=%d at epoch=%d" % (fold_idx0, epoch))
            break

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tr_preds, tr_trues = predict(model, train_loader_eval, device)
    va_preds, va_trues = predict(model, val_loader, device)
    te_preds, te_trues = predict(model, test_loader, device)

    train_metrics = compute_metrics(tr_trues, tr_preds)
    val_metrics = compute_metrics(va_trues, va_preds)
    test_metrics = compute_metrics(te_trues, te_preds)

    pred_csv_path = os.path.join(out_res_dir, "%s_%s_fold%d_test_predictions.csv" % (cfg["name"], split_mode, fold_idx0))
    out_pred = pd.DataFrame({
        "cell_id": test_df[cfg["pair_cell_col"]].astype(str).tolist(),
        "drug_id": test_df[cfg["pair_drug_col"]].astype(str).tolist(),
        "y_true": te_trues,
        "y_pred": te_preds
    })
    out_pred.to_csv(pred_csv_path, index=False)

    return {
        "dataset": cfg["name"],
        "split_mode": split_mode,
        "fold0": fold_idx0,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "ckpt_path": os.path.relpath(ckpt_path, os.path.dirname(os.path.dirname(__file__))),
        "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
        "artifacts": {"test_pred_csv": os.path.relpath(pred_csv_path, os.path.dirname(os.path.dirname(__file__)))}
    }


def eval_only(cfg, df_cell, df_drug_dict, df_desc, split_mode, fold_idx0, device, args, out_res_dir):
    if (args.ckpt_path is None) and (args.ckpt_dir is None):
        raise ValueError("--eval_only requires --ckpt_path or --ckpt_dir")

    train_df, val_df, test_df = load_fold_pairs(cfg, split_mode, fold_idx0)

    train_df = filter_pairs_to_embeddings(train_df, cfg, df_cell, df_drug_dict, df_desc)
    val_df = filter_pairs_to_embeddings(val_df, cfg, df_cell, df_drug_dict, df_desc)
    test_df = filter_pairs_to_embeddings(test_df, cfg, df_cell, df_drug_dict, df_desc)

    train_ds = DrugResponseDatasetMulti(train_df, cfg, df_drug_dict, df_cell, df_desc)
    val_ds = DrugResponseDatasetMulti(val_df, cfg, df_drug_dict, df_cell, df_desc)
    test_ds = DrugResponseDatasetMulti(test_df, cfg, df_drug_dict, df_cell, df_desc)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = InFoDRP(
        cell_dim=cfg["cell_dim"],
        desc_dim=cfg["desc_dim"],
        dropout_rate=args.dropout,
        codebook_size=args.codebook_size,
        gamma_drug=args.gamma_drug,
        gamma_cell=args.gamma_cell,
    ).to(device)

    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = os.path.join(args.ckpt_dir, "fold%d_best.pt" % fold_idx0)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Checkpoint not found for fold%d: %s" % (fold_idx0, ckpt_path))

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tr_preds, tr_trues = predict(model, train_loader, device)
    va_preds, va_trues = predict(model, val_loader, device)
    te_preds, te_trues = predict(model, test_loader, device)

    train_metrics = compute_metrics(tr_trues, tr_preds)
    val_metrics = compute_metrics(va_trues, va_preds)
    test_metrics = compute_metrics(te_trues, te_preds)

    pred_csv_path = os.path.join(out_res_dir, "evalonly_%s_%s_fold%d_test_predictions.csv" % (cfg["name"], split_mode, fold_idx0))
    out_pred = pd.DataFrame({
        "cell_id": test_df[cfg["pair_cell_col"]].astype(str).tolist(),
        "drug_id": test_df[cfg["pair_drug_col"]].astype(str).tolist(),
        "y_true": te_trues,
        "y_pred": te_preds
    })
    out_pred.to_csv(pred_csv_path, index=False)

    return {
        "dataset": cfg["name"],
        "split_mode": split_mode,
        "fold0": fold_idx0,
        "ckpt_path": ckpt_path,
        "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
        "artifacts": {"test_pred_csv": pred_csv_path}
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=["gdscv1", "gdscv2"])
    p.add_argument("--split_mode", type=str, required=True)
    p.add_argument("--exp_name", type=str, default="run_default")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--codebook_size", type=int, default=1000)
    p.add_argument("--gamma_drug", type=float, default=0.9)
    p.add_argument("--gamma_cell", type=float, default=0.7)
    p.add_argument("--lam_inv", type=float, default=1e-4)
    p.add_argument("--lam_reg", type=float, default=0.1)
    p.add_argument("--lam_cmt", type=float, default=0.05)

    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--fold", type=int, default=None)

    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--ckpt_dir", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    set_seed(args.seed)
    device = pick_device(args.device)

    cfg = build_dataset_config(repo_root, args.dataset)
    if args.split_mode not in cfg["mode_map"]:
        raise ValueError("Invalid split_mode=%s, allowed=%s" % (args.split_mode, list(cfg["mode_map"].keys())))

    out_ckpt_dir = os.path.join(repo_root, "checkpoint", args.exp_name)
    out_res_dir = os.path.join(repo_root, "result", args.exp_name)
    if not args.eval_only:
        ensure_dir(out_ckpt_dir)

    ensure_dir(out_res_dir)

    df_cell, df_drug_dict, df_desc = load_embeddings(cfg)

    folds = [args.fold] if args.fold is not None else list(range(cfg["num_folds"]))

    all_results = []
    for fold_idx0 in folds:
        if args.eval_only:
            res = eval_only(cfg, df_cell, df_drug_dict, df_desc, args.split_mode, fold_idx0, device, args, out_res_dir)
        else:
            res = run_fold(cfg, df_cell, df_drug_dict, df_desc, args.split_mode, fold_idx0, device, args, out_ckpt_dir, out_res_dir)
        all_results.append(res)

    test_rmses = [r["metrics"]["test"]["RMSE"] for r in all_results]
    test_pccs = [r["metrics"]["test"]["PCC"] for r in all_results]

    summary = {
        "exp_name": args.exp_name,
        "dataset": cfg["name"],
        "split_mode": args.split_mode,
        "seed": args.seed,
        "folds": [r["fold0"] for r in all_results],
        "mean_test_RMSE": float(np.mean(test_rmses)) if len(test_rmses) else None,
        "mean_test_PCC": float(np.mean(test_pccs)) if len(test_pccs) else None,
        "per_fold": all_results,
    }

    out_json = os.path.join(out_res_dir, "summary_%s_%s.json" % (cfg["name"], args.split_mode))
    save_json(summary, out_json)

    print("\n====================")
    print("DONE")
    print("Saved summary:", out_json)
    print("Mean Test RMSE:", summary["mean_test_RMSE"])
    print("Mean Test PCC :", summary["mean_test_PCC"])
    print("====================\n")


if __name__ == "__main__":
    main()
