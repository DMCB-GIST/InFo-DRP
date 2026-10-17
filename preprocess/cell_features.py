"""Build cell line features: COSMIC Cancer Gene Census (CGC) genes, z-scored within each cell line.

Requires the GDSC RMA expression matrix `Cell_line_RMA_proc_basalExp.txt`.

Example:
    python preprocess/cell_features.py --dataset gdscv2 --rma Cell_line_RMA_proc_basalExp.txt
"""
import argparse
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SETTINGS = {
    "gdscv1": {
        "genes": "data/gdscv1/input/cgc_genes_426.txt",
        "cells": "data/gdscv1/input/cell_lines_734.csv",
        "out": "gdsc1_cell734_expr_426.csv",
        "id_col": "COSMIC_ID",
        "keep_duplicate": "first",
    },
    "gdscv2": {
        "genes": "data/gdscv2/input/cgc_genes_699.txt",
        "cells": "data/gdscv2/input/cell_lines_941.csv",
        "out": "gdsc2_cell941_expr_699.csv",
        "id_col": "CELL_LINE_NAME",
        "keep_duplicate": "last",
    },
}

# The GDSCv1 gene list uses current HGNC symbols; the RMA file still uses these previous symbols.
GDSCV1_ALIASES = {
    "NSD3": "WHSC1L1", "KMT2C": "MLL3", "WDCP": "C2orf44", "SHTN1": "KIAA1598", "MYCL": "MYCL1",
    "NSD2": "WHSC1", "KNL1": "CASC5", "TENT5C": "FAM46C", "ACKR3": "CXCR7", "LHFPL6": "LHFP",
}


def load_rma(path, keep_duplicate):
    """Cell x gene matrix indexed by COSMIC ID.

    A few COSMIC IDs appear twice in the RMA file; `keep_duplicate` selects the profile
    used in each benchmark dataset.
    """
    df = pd.read_csv(path, sep="\t").drop(columns=["GENE_title"]).set_index("GENE_SYMBOLS").T
    cosmic = df.index.str.replace("DATA.", "", regex=False).str.split(".").str[0].astype(int)
    df.index = cosmic
    return df[~df.index.duplicated(keep=keep_duplicate)]


def cellwise_z(X):
    v = X.to_numpy(dtype=np.float64)
    z = (v - v.mean(axis=1, keepdims=True)) / v.std(axis=1, ddof=0, keepdims=True)
    return pd.DataFrame(z, index=X.index, columns=X.columns)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(SETTINGS))
    p.add_argument("--rma", required=True, help="Path to Cell_line_RMA_proc_basalExp.txt")
    p.add_argument("--out_dir", default="features")
    args = p.parse_args()

    s = SETTINGS[args.dataset]
    genes = [g.strip() for g in open(os.path.join(REPO_ROOT, s["genes"])) if g.strip()]
    cells = pd.read_csv(os.path.join(REPO_ROOT, s["cells"]))

    X = load_rma(args.rma, s["keep_duplicate"])
    if args.dataset == "gdscv1":
        X = X.rename(columns={old: new for new, old in GDSCV1_ALIASES.items()})
    X = X.loc[cells["COSMIC_ID"], genes]
    X.index = cells[s["id_col"]]

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, s["out"])
    cellwise_z(X).to_csv(out)
    print(f"saved {out} {X.shape}")


if __name__ == "__main__":
    main()
