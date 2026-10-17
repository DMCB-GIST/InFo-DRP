"""Build all InFo-DRP drug features from a SMILES table.

Input : CSV with a drug identifier column (default `PubChemID`) and a SMILES column (default `SMILES`).
Output: <out_dir>/<prefix>_smiles_canonical.csv and four feature files
        <prefix>_molbert_768.csv, <prefix>_chemformer_1024.csv,
        <prefix>_ecfp4_2048.csv, <prefix>_descriptor_280.csv

Example:
    python preprocess/drug_features.py --input data/gdscv2/input/gdsc2_drug222_smiles.csv --out_dir features --prefix gdsc2_drug222
"""
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoders.chemformer import ChemformerEncoder  # noqa: E402
from encoders.molbert import MolBertEncoder  # noqa: E402
from encoders.rdkit_features import canonicalize, descriptors_280, ecfp4  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRETRAINED = os.path.join(REPO_ROOT, "pretrained")
FEATURES = ("molbert", "chemformer", "ecfp4", "descriptor")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", required=True, help="File name prefix, e.g. gdsc2_drug222")
    p.add_argument("--id_col", default="PubChemID")
    p.add_argument("--smiles_col", default="SMILES",
                   help="Raw SMILES column; RDKit canonical SMILES are computed from it")
    p.add_argument("--features", nargs="+", default=list(FEATURES), choices=FEATURES)
    p.add_argument("--molbert_ckpt", default=os.path.join(PRETRAINED, "molbert_100epochs", "checkpoints", "last.ckpt"))
    p.add_argument("--molbert_hparams", default=os.path.join(PRETRAINED, "molbert_100epochs", "hparams.yaml"))
    p.add_argument("--chemformer_ckpt", default=os.path.join(PRETRAINED, "chemformer", "combined_large.ckpt"))
    p.add_argument("--chemformer_vocab", default=os.path.join(REPO_ROOT, "preprocess", "assets", "bart_vocab.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def save(df_ids, feats, path):
    out = pd.concat([df_ids.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    out.to_csv(path, index=False)
    print(f"saved {path} {feats.shape}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input, dtype={args.id_col: str})
    df["canonical_SMILES"] = df[args.smiles_col].map(canonicalize)
    ids = df[[args.id_col]].rename(columns={args.id_col: "PubChemID"})
    smiles = df["canonical_SMILES"].tolist()
    df[[args.id_col, args.smiles_col, "canonical_SMILES"]].to_csv(
        os.path.join(args.out_dir, f"{args.prefix}_smiles_canonical.csv"), index=False)

    stem = os.path.join(args.out_dir, f"{args.prefix}_")
    if "molbert" in args.features:
        enc = MolBertEncoder(args.molbert_ckpt, args.molbert_hparams, device=args.device)
        save(ids, pd.DataFrame(enc.embed(smiles)), stem + "molbert_768.csv")
        del enc
    if "chemformer" in args.features:
        enc = ChemformerEncoder(args.chemformer_ckpt, args.chemformer_vocab, device=args.device)
        save(ids, pd.DataFrame(enc.embed(smiles)), stem + "chemformer_1024.csv")
        del enc
    if "ecfp4" in args.features:
        save(ids, ecfp4(smiles), stem + "ecfp4_2048.csv")
    if "descriptor" in args.features:
        save(ids, descriptors_280(smiles), stem + "descriptor_280.csv")


if __name__ == "__main__":
    main()
