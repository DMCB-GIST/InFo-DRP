import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RVQ(nn.Module):
    """Residual vector quantization: z + nearest codebook entry."""

    def __init__(self, latent_dim=256, codebook_size=1000):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.codebook.weight, -0.1, 0.1)

    def forward(self, z):
        dist = torch.sum((z.unsqueeze(1) - self.codebook.weight.unsqueeze(0)) ** 2, dim=2)
        _, idx = torch.min(dist, dim=1)
        e = self.codebook(idx)
        z_res = z + e
        commit_loss = F.mse_loss(e.detach(), z)
        codebook_loss = F.mse_loss(e, z.detach())
        return z_res, commit_loss, codebook_loss


class ScoringMLP(nn.Module):
    """Element-wise sigmoid gate; bias initialised so the mean gate starts at gamma."""

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
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


def compute_invariant_loss(z_inv, z_spu, proj_mlp):
    """Shuffled similarity regularization (negative cosine similarity)."""
    idx = torch.randperm(z_inv.size(0), device=z_inv.device)
    view = torch.cat([z_inv.detach(), z_spu[idx]], dim=1)
    online = F.normalize(proj_mlp(view), dim=1)
    target = F.normalize(z_inv.detach(), dim=1)
    return -F.cosine_similarity(online, target, dim=1).mean()


def compute_score_regularization(S, gamma):
    return torch.abs(S.mean() - gamma)


def _mlp_encoder(in_dim, dropout_rate):
    """in -> (2048) -> (1024) -> (512) -> 256, depth chosen by input size."""
    widths = [w for w in (2048, 1024, 512) if in_dim > w]
    layers, prev = [], in_dim
    for w in widths:
        layers += [nn.Linear(prev, w), nn.BatchNorm1d(w), nn.ReLU(), nn.Dropout(dropout_rate)]
        prev = w
    layers.append(nn.Linear(prev, 256))
    return nn.Sequential(*layers)


class InFoDRP(nn.Module):
    DRUG_DIMS = {"molbert": 768, "chemformer": 1024, "ecfp4": 2048}

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
        self.cell_dim = cell_dim
        self.desc_dim = desc_dim

        self.drug_names = list(self.DRUG_DIMS)
        self.drug_in_dims = {n: d + desc_dim for n, d in self.DRUG_DIMS.items()}

        # Drug branches: encoder -> RVQ -> invariant/spurious gate
        self.drug_encoders = nn.ModuleDict({n: _mlp_encoder(self.drug_in_dims[n], dropout_rate) for n in self.drug_names})
        self.drug_bns = nn.ModuleDict({n: nn.BatchNorm1d(256) for n in self.drug_names})
        self.rvq_drugs = nn.ModuleDict({n: RVQ(256, codebook_size) for n in self.drug_names})
        self.scorers_drug = nn.ModuleDict({n: ScoringMLP(self.drug_in_dims[n], gamma_drug) for n in self.drug_names})
        self.proj_mlps_drug = nn.ModuleDict({n: ProjectionMLP(256) for n in self.drug_names})

        # Per-branch projector P for the drug-cell interaction feature (1024 -> 256)
        self.fuse_mlps = nn.ModuleDict({
            n: nn.Sequential(
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            ) for n in self.drug_names
        })

        # Cell branch
        self.rvq_cell = RVQ(256, codebook_size)
        self.scorer_cell = ScoringMLP(cell_dim, gamma_cell)
        self.proj_mlp_cell = ProjectionMLP(256)
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

        # Dual-path fusion: Transformer over [CLS, branch tokens] + MLP aggregate R
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024, dropout=dropout_rate, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.pre_fuse_mlp = nn.Sequential(
            nn.Linear(256 * len(self.drug_names), 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )

    def _block(self, x):
        return self.dropout(self.relu(x))

    def _forward_drug(self, name, d_raw):
        d = self._block(self.drug_bns[name](self.drug_encoders[name](d_raw)))
        z_res, cmt, cb = self.rvq_drugs[name](d)
        S = self.scorers_drug[name](d_raw)
        return z_res * S, z_res * (1.0 - S), S, cmt, cb

    def _forward_cell(self, c_raw):
        c = self._block(self.cell_bn1(self.cell_fc1(c_raw)))
        if self.cell_fc2 is not None:
            c = self._block(self.cell_bn2(self.cell_fc2(c)))
        z_res, cmt, cb = self.rvq_cell(c)
        S = self.scorer_cell(c_raw)
        return z_res * S, z_res * (1.0 - S), S, cmt, cb

    def forward_invariant(self, x):
        lengths = [self.drug_in_dims[n] for n in self.drug_names] + [self.cell_dim]
        chunks = torch.split(x, lengths, dim=1)
        z_inv_c, z_spu_c, S_c, cmt_c, cb_c = self._forward_cell(chunks[-1])

        tokens = []
        z_inv_list, z_spu_list, S_list, cmt_list, cb_list = [], [], [], [], []
        for i, name in enumerate(self.drug_names):
            z_inv_d, z_spu_d, S_d, cmt_d, cb_d = self._forward_drug(name, chunks[i])
            phi = torch.cat([
                z_inv_d, z_inv_c,
                z_inv_d * torch.sigmoid(z_inv_c),
                z_inv_c * torch.sigmoid(z_inv_d),
            ], dim=1)
            tokens.append(self.fuse_mlps[name](phi))
            z_inv_list.append(z_inv_d)
            z_spu_list.append(z_spu_d)
            S_list.append(S_d)
            cmt_list.append(cmt_d)
            cb_list.append(cb_d)

        B = x.size(0)
        seq_pre = torch.stack(tokens, dim=1)
        seq = torch.cat([self.cls_token.expand(B, 1, -1), seq_pre], dim=1)
        cls_out = self.encoder(seq)[:, 0, :]
        pre_summary = self.pre_fuse_mlp(seq_pre.reshape(B, -1))
        out = self.final_mlp(torch.cat([cls_out, pre_summary], dim=1)).squeeze(1)

        return out, z_inv_list, z_spu_list, S_list, cmt_list, cb_list, z_inv_c, z_spu_c, S_c, cmt_c, cb_c

    def forward(self, x):
        return self.forward_invariant(x)[0]
