import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import DrugResponseDataset, build_dataset_config, filter_pairs, load_features, load_fold_pairs, make_loader
from model import InFoDRP, compute_invariant_loss, compute_score_regularization
from util import compute_metrics, ensure_dir, pick_device, save_json, set_seed

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default learning rate for each setting (CCLE: same as GDSCv2 scaffold-blind).
DEFAULT_LR = {
    ("gdscv1", "newDRUG"): 5e-5,
    ("gdscv1", "newCELL"): 5e-5,
    ("gdscv1", "newPAIR"): 5e-5,
    ("gdscv2", "newSCAFFOLD"): 1e-5,
    ("gdscv2", "newDRUG"): 5e-5,
    ("gdscv2", "newCELL"): 3e-5,
    ("ccle", "newSCAFFOLD"): 1e-5,
}


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if (isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm, nn.Embedding))
                    or param_name.endswith("bias") or "scorer" in module_name):
                no_decay.append(param)
            else:
                decay.append(param)
    return optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    criterion = nn.MSELoss()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out, z_inv_d, z_spu_d, S_d, cmt_d, cb_d, z_inv_c, z_spu_c, S_c, cmt_c, cb_c = model.forward_invariant(X)
        loss_pred = criterion(out, y)

        inv_losses, reg_losses, cmt_losses = [], [], []
        for i, name in enumerate(model.drug_names):
            inv_losses.append(compute_invariant_loss(z_inv_d[i], z_spu_d[i], model.proj_mlps_drug[name]))
            reg_losses.append(compute_score_regularization(S_d[i], model.gamma_drug))
            cmt_losses.append(cmt_d[i] + cb_d[i])
        loss_inv_d = torch.stack(inv_losses).mean()
        loss_reg_d = torch.stack(reg_losses).mean()
        loss_cmt_d = torch.stack(cmt_losses).mean()
        loss_inv_c = compute_invariant_loss(z_inv_c, z_spu_c, model.proj_mlp_cell)
        loss_reg_c = compute_score_regularization(S_c, model.gamma_cell)
        loss_cmt_c = cmt_c + cb_c

        loss = (loss_pred
                + args.lam_inv * (loss_inv_d + loss_inv_c)
                + args.lam_reg * (loss_reg_d + loss_reg_c)
                + args.lam_cmt * (loss_cmt_d + loss_cmt_c))
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def run_one(cfg, feats, split_mode, fold, run_name, lr, device, args, ckpt_dir, res_dir):
    """Train (or load) one model and evaluate it."""
    df_cell, df_drugs, df_desc = feats
    splits = [filter_pairs(df, cfg, df_cell, df_drugs, df_desc) for df in load_fold_pairs(cfg, split_mode, fold)]
    train_ds, val_ds, test_ds = [DrugResponseDataset(df, cfg, df_cell, df_drugs, df_desc) for df in splits]
    test_df = splits[2]

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False)

    model = InFoDRP(
        cell_dim=df_cell.shape[1],
        desc_dim=df_desc.shape[1],
        dropout_rate=args.dropout,
        codebook_size=args.codebook_size,
        gamma_drug=args.gamma_drug,
        gamma_cell=args.gamma_cell,
    ).to(device)

    if args.eval_only:
        ckpt_path = os.path.join(args.ckpt_dir, f"{run_name}_best.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
    else:
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")
        optimizer = build_optimizer(model, lr, args.weight_decay)
        best_val_mse, bad_epochs = float("inf"), 0
        for _ in tqdm(range(args.max_epochs), desc=f"[{cfg['name']} {split_mode} {run_name} lr={lr:g}]"):
            train_one_epoch(model, train_loader, optimizer, device, args)
            val_pred, val_true = predict(model, val_loader, device)
            val_mse = float(np.mean((val_true.astype(np.float64) - val_pred.astype(np.float64)) ** 2))
            if val_mse < best_val_mse:
                best_val_mse, bad_epochs = val_mse, 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {"dataset": cfg["name"], "split_mode": split_mode, "run": run_name, "lr": lr,
                               "cell_dim": df_cell.shape[1], "desc_dim": df_desc.shape[1]},
                }, ckpt_path)
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break
        ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    pred, true = predict(model, test_loader, device)
    pd.DataFrame({
        "cell_id": test_df[cfg["pair_cell_col"]],
        "drug_id": test_df[cfg["pair_drug_col"]],
        "y_true": true,
        "y_pred": pred,
    }).to_csv(os.path.join(res_dir, f"{run_name}_test_predictions.csv"), index=False)

    metrics = compute_metrics(true, pred)
    print(f"[{run_name}] test RMSE={metrics['RMSE']:.4f} | test PCC={metrics['PCC']:.4f}")
    return {"run": run_name, **metrics}


def run_setting(cfg, feats, args, lr, device, ckpt_dir, res_dir):
    ensure_dir(res_dir)
    if not args.eval_only:
        ensure_dir(ckpt_dir)

    results = []
    if cfg["name"] == "ccle":
        # One fixed GDSCv2 train/val split, repeated over training seeds.
        for seed in args.seeds:
            set_seed(seed)
            results.append(run_one(cfg, feats, args.split_mode, 0, f"seed{seed}", lr, device, args, ckpt_dir, res_dir))
    else:
        set_seed(args.seed)
        folds = [args.fold] if args.fold is not None else range(cfg["num_folds"])
        for fold in folds:
            results.append(run_one(cfg, feats, args.split_mode, fold, f"fold{fold}", lr, device, args, ckpt_dir, res_dir))

    summary = {"dataset": cfg["name"], "split_mode": args.split_mode, "runs": results}
    for m in ("RMSE", "PCC"):
        vals = [r[m] for r in results]
        summary[f"test_{m}_mean"] = float(np.mean(vals))
        if len(vals) > 1:
            summary[f"test_{m}_std"] = float(np.std(vals, ddof=1))
    save_json(summary, os.path.join(res_dir, "summary.json"))
    if len(results) > 1:
        print(f"\n{cfg['name']} {args.split_mode}: "
              f"test RMSE {summary['test_RMSE_mean']:.3f} ({summary['test_RMSE_std']:.3f}) | "
              f"test PCC {summary['test_PCC_mean']:.3f} ({summary['test_PCC_std']:.3f})")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Train / evaluate InFo-DRP")
    p.add_argument("--dataset", required=True, choices=["gdscv1", "gdscv2", "ccle"])
    p.add_argument("--split_mode", required=True,
                   help="gdscv1: newDRUG | newCELL | newPAIR; gdscv2: newSCAFFOLD | newDRUG | newCELL; ccle: newSCAFFOLD")
    p.add_argument("--exp_name", default=None, help="Output subdirectory under checkpoint/ and result/")

    p.add_argument("--lr", type=float, default=None, help="Default: setting-specific learning rate")
    p.add_argument("--seed", type=int, default=0, help="Random seed (gdscv1/gdscv2)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Training seeds (ccle)")
    p.add_argument("--fold", type=int, default=None, help="Run a single fold (0-based) instead of all folds")

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

    p.add_argument("--eval_only", action="store_true", help="Evaluate checkpoints in --ckpt_dir without training")
    p.add_argument("--ckpt_dir", default=None, help="Checkpoint directory for --eval_only")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    key = (args.dataset, args.split_mode)
    if key not in DEFAULT_LR:
        raise ValueError(f"Unsupported setting {key}; choose from {sorted(DEFAULT_LR)}")
    if args.eval_only and args.ckpt_dir is None:
        raise ValueError("--eval_only requires --ckpt_dir")

    exp_name = args.exp_name or f"{args.dataset}_{args.split_mode}" + ("_eval" if args.eval_only else "_train")
    device = pick_device(args.device)
    cfg = build_dataset_config(REPO_ROOT, args.dataset)
    feats = load_features(cfg)

    lr = args.lr if args.lr is not None else DEFAULT_LR[key]
    run_setting(cfg, feats, args, lr, device,
                os.path.join(REPO_ROOT, "checkpoint", exp_name), os.path.join(REPO_ROOT, "result", exp_name))


if __name__ == "__main__":
    main()
