"""Chemformer embeddings (1024-d) from SMILES: mean of the encoder outputs.

Adapted from Chemformer (https://github.com/MolecularAI/Chemformer), Apache License 2.0.
`preprocess/assets/bart_vocab.json` is the Chemformer vocabulary file.
"""
import json
import math
import re

import numpy as np
import torch
import torch.nn as nn

from .ckpt_utils import load_checkpoint


class ChemformerTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        self.regex = re.compile(vocab["properties"]["regex"])
        self.token_to_idx = {t: i for i, t in enumerate(vocab["vocabulary"])}
        self.unk_idx = self.token_to_idx[vocab["properties"]["special_tokens"]["unknown"]]
        self.pad_idx = self.token_to_idx[vocab["properties"]["special_tokens"]["pad"]]

    def __len__(self):
        return len(self.token_to_idx)

    def encode(self, smiles):
        return [self.token_to_idx.get(t, self.unk_idx) for t in self.regex.findall(smiles)]


class PreNormEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        att = self.norm1(src)
        att = self.self_attn(att, att, att, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        att = src + self.dropout1(att)
        out = self.norm2(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        return att + self.dropout2(out)


def sinusoidal_positions(d_model, max_seq_len):
    encs = torch.tensor([dim / d_model for dim in range(0, d_model, 2)])
    encs = 10000 ** encs
    encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(max_seq_len)]
    encs = [torch.stack(enc, dim=1).flatten()[:d_model] for enc in encs]
    return torch.stack(encs)


class ChemformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model=1024, num_layers=8, num_heads=16, d_feedforward=4096,
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_emb", sinusoidal_positions(d_model, max_seq_len))
        self.encoder = nn.TransformerEncoder(
            PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation="relu"),
            num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, token_ids):
        # token_ids: (seq_len, batch)
        seq_len = min(token_ids.size(0), self.max_seq_len)
        x = self.emb(token_ids) * math.sqrt(self.d_model)
        x = x + self.pos_emb[:seq_len].unsqueeze(1)
        pad_mask = torch.zeros(token_ids.size(1), token_ids.size(0), dtype=torch.bool, device=token_ids.device)
        return self.encoder(self.dropout(x), src_key_padding_mask=pad_mask)


class ChemformerEncoder:
    def __init__(self, checkpoint_path, vocab_path, device="cpu"):
        self.tokenizer = ChemformerTokenizer(vocab_path)
        self.device = torch.device(device)
        ckpt = load_checkpoint(checkpoint_path)
        hp = ckpt["hyper_parameters"]
        self.model = ChemformerEncoderModel(
            vocab_size=len(self.tokenizer),
            pad_idx=self.tokenizer.pad_idx,
            d_model=hp["d_model"],
            num_layers=hp["num_layers"],
            num_heads=hp["num_heads"],
            d_feedforward=hp["d_feedforward"],
            max_seq_len=hp["max_seq_len"],
        )
        state = {k: v for k, v in ckpt["state_dict"].items() if k.startswith(("encoder.", "emb.", "pos_emb"))}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def embed_one(self, smiles):
        ids = self.tokenizer.encode(smiles)[: self.model.max_seq_len]
        token_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(1)
        out = self.model(token_ids)
        return out.mean(dim=0)[0].cpu().numpy()

    def embed(self, smiles_list):
        return np.stack([self.embed_one(s) for s in smiles_list]).astype(np.float32)
