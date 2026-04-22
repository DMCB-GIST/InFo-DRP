# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RVQ(nn.Module):
    def __init__(self, latent_dim=256, codebook_size=1000):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.codebook.weight, -0.1, 0.1)

    def forward(self, z):
        # z: (B, D)
        z_exp = z.unsqueeze(1)  # (B,1,D)
        code_exp = self.codebook.weight.unsqueeze(0)  # (1,K,D)
        dist = torch.sum((z_exp - code_exp) ** 2, dim=2)  # (B,K)
        _, idx = torch.min(dist, dim=1)  # (B,)
        e = self.codebook(idx)  # (B,D)

        z_res = z + e
        commit_loss = F.mse_loss(e.detach(), z)
        codebook_loss = F.mse_loss(e, z.detach())
        return z_res, commit_loss, codebook_loss


class ScoringMLP(nn.Module):
    def __init__(self, in_dim, gamma=0.9):
        super().__init__()
        self.fc = nn.Linear(in_dim, 256)
        self.sigmoid = nn.Sigmoid()
        with torch.no_grad():
            nn.init.xavier_uniform_(self.fc.weight)
            self.fc.bias.fill_(float(np.log(gamma / (1.0 - gamma))))

    def forward(self, x):
        return self.sigmoid(self.fc(x))


class ProjectionMLP(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


def compute_invariant_loss(z_inv, z_spu, proj_mlp):
    B = z_inv.size(0)
    idx = torch.randperm(B, device=z_inv.device)
    z_spu_shuf = z_spu[idx]

    view = torch.cat([z_inv.detach(), z_spu_shuf], dim=1)
    pred = proj_mlp(view)

    target = F.normalize(z_inv.detach(), dim=1)
    online = F.normalize(pred, dim=1)

    cos = F.cosine_similarity(online, target, dim=1).mean()
    return -cos


def compute_score_regularization(S, gamma):
    return torch.abs(S.mean() - gamma)


class InFoDRP(nn.Module):
    def __init__(
        self,
        cell_dim,
        desc_dim=280,
        dropout_rate=0.1,
        codebook_size=1000,
        gamma_drug=0.9,
        gamma_cell=0.7,
    ):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.gamma_drug = gamma_drug
        self.gamma_cell = gamma_cell

        self.drug_names = ["molbert", "chemformer", "ecfp4"]
        self.drug_in_dims = {
            "molbert": 768 + desc_dim,
            "chemformer": 1024 + desc_dim,
            "ecfp4": 2048 + desc_dim,
        }

        def _build_encoder(in_dim):
            if in_dim > 2048:
                return nn.Sequential(
                    nn.Linear(in_dim, 2048),
                    nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(dropout_rate),
                    nn.Linear(2048, 1024),
                    nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout_rate),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
                    nn.Linear(512, 256),
                )
            elif in_dim > 1024:
                return nn.Sequential(
                    nn.Linear(in_dim, 1024),
                    nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout_rate),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
                    nn.Linear(512, 256),
                )
            elif in_dim > 512:
                return nn.Sequential(
                    nn.Linear(in_dim, 512),
                    nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
                    nn.Linear(512, 256),
                )
            else:
                return nn.Sequential(nn.Linear(in_dim, 256))

        self.drug_encoders = nn.ModuleDict({n: _build_encoder(self.drug_in_dims[n]) for n in self.drug_names})
        self.drug_bns = nn.ModuleDict({n: nn.BatchNorm1d(256) for n in self.drug_names})
        self.rvq_drugs = nn.ModuleDict({n: RVQ(256, codebook_size) for n in self.drug_names})
        self.scorers_drug = nn.ModuleDict({n: ScoringMLP(self.drug_in_dims[n], gamma_drug) for n in self.drug_names})
        self.proj_mlps_drug = nn.ModuleDict({n: ProjectionMLP(256) for n in self.drug_names})

        # cell encoder
        if cell_dim >= 512:
            self.cell_fc1 = nn.Linear(cell_dim, 512)
            self.cell_bn1 = nn.BatchNorm1d(512)
            self.cell_fc2 = nn.Linear(512, 256)
            self.cell_bn2 = nn.BatchNorm1d(256)
        else:
            self.cell_fc1 = nn.Linear(cell_dim, 256)
            self.cell_bn1 = nn.BatchNorm1d(256)
            self.cell_fc2 = None
            self.cell_bn2 = None

        self.rvq_cell = RVQ(256, codebook_size)
        self.scorer_cell = ScoringMLP(cell_dim, gamma_cell)
        self.proj_mlp_cell = ProjectionMLP(256)

        # fusion mlps
        self.fuse_mlps = nn.ModuleDict({
            n: nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            ) for n in self.drug_names
        })

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024, dropout=dropout_rate, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)

        num_tokens = len(self.drug_names)
        self.pre_fuse_mlp = nn.Sequential(
            nn.Linear(256 * num_tokens, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

        self.cell_dim = cell_dim
        self.desc_dim = desc_dim

    def _forward_single_drug(self, name, d_raw):
        d = self.drug_encoders[name](d_raw)
        d = self.drug_bns[name](d)
        d = self.relu(d)
        d = self.dropout(d)

        z_res, cmt, cb = self.rvq_drugs[name](d)
        S = self.scorers_drug[name](d_raw)
        z_inv = z_res * S
        z_spu = z_res * (1.0 - S)
        return z_inv, z_spu, S, cmt, cb

    def _forward_cell(self, c_raw):
        if self.cell_fc2 is None:
            c = self.cell_fc1(c_raw)
            c = self.cell_bn1(c)
            c = self.relu(c)
            c = self.dropout(c)
        else:
            c = self.cell_fc1(c_raw)
            c = self.cell_bn1(c)
            c = self.relu(c)
            c = self.dropout(c)

            c = self.cell_fc2(c)
            c = self.cell_bn2(c)
            c = self.relu(c)
            c = self.dropout(c)

        z_res, cmt, cb = self.rvq_cell(c)
        S = self.scorer_cell(c_raw)
        z_inv = z_res * S
        z_spu = z_res * (1.0 - S)
        return z_inv, z_spu, S, cmt, cb

    def forward_invariant(self, x):
        lengths = [self.drug_in_dims[n] for n in self.drug_names] + [self.cell_dim]
        drug_chunks = torch.split(x, lengths, dim=1)
        c_raw = drug_chunks[-1]
        d_map = {n: drug_chunks[i] for i, n in enumerate(self.drug_names)}

        z_inv_c, z_spu_c, S_c, cmt_c, cb_c = self._forward_cell(c_raw)

        tokens = []
        z_inv_list, z_spu_list, S_list, cmt_list, cb_list = [], [], [], [], []

        for name in self.drug_names:
            z_inv_d, z_spu_d, S_d, cmt_d, cb_d = self._forward_single_drug(name, d_map[name])

            cat_1 = torch.cat([z_inv_d, z_inv_c], dim=1)
            mul_1 = z_inv_d * torch.sigmoid(z_inv_c)
            mul_2 = z_inv_c * torch.sigmoid(z_inv_d)
            fuse = torch.cat([cat_1, mul_1, mul_2], dim=1)  # 1024

            tok = self.fuse_mlps[name](fuse)  # 256
            tokens.append(tok)

            z_inv_list.append(z_inv_d)
            z_spu_list.append(z_spu_d)
            S_list.append(S_d)
            cmt_list.append(cmt_d)
            cb_list.append(cb_d)

        B = x.size(0)
        cls = self.cls_token.expand(B, 1, -1)
        seq_pre = torch.stack(tokens, dim=1)  # (B,N,256)
        seq = torch.cat([cls, seq_pre], dim=1)  # (B,N+1,256)

        enc = self.encoder(seq)
        cls_out = enc[:, 0, :]

        flat_pre = seq_pre.reshape(B, -1)
        pre_summary = self.pre_fuse_mlp(flat_pre)

        final_in = torch.cat([cls_out, pre_summary], dim=1)  # 512
        out = self.final_mlp(final_in).squeeze(1)

        return out, z_inv_list, z_spu_list, S_list, cmt_list, cb_list, z_inv_c, z_spu_c, S_c, cmt_c, cb_c

    def forward(self, x):
        out, *_ = self.forward_invariant(x)
        return out
