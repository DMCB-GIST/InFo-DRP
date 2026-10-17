"""MolBERT pooled embeddings (768-d) from SMILES.

Adapted from MolBERT (https://github.com/BenevolentAI/MolBERT),
Copyright (c) 2020 BenevolentAI, MIT License.
"""

import numpy as np
import torch
import torch.nn as nn
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler

from .ckpt_utils import load_checkpoint

RDLogger.DisableLog("rdApp.*")

ELEMENTS = (
    "Ac", "Al", "Am", "Sb", "Ar", "As", "At", "Ba", "Bk", "Be", "Bi", "Bh", "B", "Br", "Cd", "Ca", "Cf", "C",
    "Ce", "Cs", "Cl", "Cr", "Co", "Cu", "Cm", "Ds", "Db", "Dy", "Es", "Er", "Eu", "Fm", "F", "Fr", "Gd", "Ga",
    "Ge", "Au", "Hf", "Hs", "He", "Ho", "H", "In", "I", "Ir", "Fe", "Kr", "La", "Lr", "Pb", "Li", "Lu", "Mg",
    "Mn", "Mt", "Md", "Hg", "Mo", "Nd", "Ne", "Np", "Ni", "Nb", "N", "No", "Uuo", "Os", "O", "Pd", "P", "Pt",
    "Pu", "Po", "K", "Pr", "Pm", "Pa", "Ra", "Rn", "Re", "Rh", "Rg", "Rb", "Ru", "Rf", "Sm", "Sc", "Sg", "Se",
    "Si", "Ag", "Na", "Sr", "S", "Ta", "Tc", "Te", "Tb", "Tl", "Th", "Tm", "Sn", "Ti", "W", "Uub", "Uuh",
    "Uup", "Uuq", "Uus", "Uut", "U", "V", "Xe", "Yb", "Y", "Zn", "Zr", "@@", "se",
)
ALLOWED = ("F", "H", "I", "B", "C", "N", "O", "P", "S", "Br", "Cl", "Si", "Se", "se", "@@")
# Multi-character symbols are mapped to single characters (order follows MolBERT's elements table).
MULTI_CHAR = (("Br", "β"), ("Cl", "χ"), ("Se", "σ"), ("Si", "π"), ("@@", "α"), ("se", "Σ"))
SMILES_CHARS = tuple("0123456789=@#%/\\()+-.[]")


class MolBertTokenizer:
    def __init__(self, max_length=128):
        self.max_length = max_length
        self.encode_dict = dict(MULTI_CHAR)
        self.forbidden = set(ELEMENTS) - set(ALLOWED)
        element_chars = [e if len(e) == 1 else self.encode_dict[e] for e in ALLOWED]
        idx_to_token = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", *element_chars, *SMILES_CHARS]
        self.token_to_idx = {t: i for i, t in enumerate(idx_to_token)}
        self.vocab_size = len(idx_to_token)

    @staticmethod
    def standardise(smiles):
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP
        Chem.SanitizeMol(mol, flags, catchErrors=True)
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
            return Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=False)
        except (ValueError, RuntimeError):
            return None

    def encode(self, smiles):
        """Return (token ids, valid)."""
        ids = np.zeros(self.max_length, dtype=np.int64)
        if len(smiles) == 0 or Chem.MolFromSmiles(smiles) is None:
            return ids, False
        if any(c in smiles for c in self.encode_dict.values()):
            return ids, False
        std = self.standardise(smiles)
        if std is None:
            return ids, False
        encoded = std
        for symbol, char in self.encode_dict.items():
            encoded = encoded.replace(symbol, char)
        tokens = ["[CLS]", *encoded, "[SEP]"]
        legal = not any(s in std for s in self.forbidden)
        valid = legal and len(tokens) <= self.max_length
        if valid:
            for i, t in enumerate(tokens):
                if t in self.token_to_idx:
                    ids[i] = self.token_to_idx[t]
        return ids, valid


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz):
        sinusoid = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        return pos_emb.unsqueeze(0).expand(bsz, -1, -1)


class MolBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = SinusoidalPositionalEmbedding(config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids):
        words = self.word_embeddings(input_ids)
        positions = torch.arange(input_ids.size(1), dtype=words.dtype, device=words.device)
        x = words + self.position_embeddings(positions, input_ids.size(0)) + self.token_type_embeddings(token_type_ids)
        return self.dropout(self.LayerNorm(x))


class MolBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = MolBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, attention_mask):
        token_type_ids = torch.zeros_like(input_ids)
        x = self.embeddings(input_ids, token_type_ids)
        ext_mask = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * -10000.0
        seq = self.encoder(x, attention_mask=ext_mask)[0]
        return self.pooler(seq)


class MolBertEncoder:
    def __init__(self, checkpoint_path, hparams_path, device="cpu"):
        with open(hparams_path) as f:
            hparams = yaml.safe_load(f)
        self.tokenizer = MolBertTokenizer(max_length=hparams["max_seq_length"])
        config = BertConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=hparams["max_position_embeddings"],
        )
        self.device = torch.device(device)
        self.model = MolBertModel(config)
        state = load_checkpoint(checkpoint_path)["state_dict"]
        prefix = "model.bert."
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _embed(self, smiles):
        ids, valid = self.tokenizer.encode(smiles)
        if not valid:
            return None
        ids = ids[: int(np.nonzero(ids)[0].max()) + 1]
        input_ids = torch.tensor(ids[None, :], dtype=torch.long, device=self.device)
        attention_mask = (input_ids != 0).long()
        return self.model(input_ids, attention_mask)[0].cpu().numpy()

    def embed_one(self, smiles):
        vec = self._embed(smiles)
        if vec is not None:
            return vec
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        largest = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumAtoms())
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        for candidate in (largest, scaffold):
            if candidate is not None and candidate.GetNumAtoms() > 0:
                vec = self._embed(Chem.MolToSmiles(candidate, canonical=True))
                if vec is not None:
                    return vec
        parts = [p for p in BRICS.BRICSDecompose(mol) if p][:20]
        vecs = [v for v in (self._embed(p) for p in parts) if v is not None]
        if vecs:
            return np.mean(np.stack(vecs), axis=0)
        raise ValueError(f"MolBERT could not featurize: {smiles}")

    def embed(self, smiles_list):
        return np.stack([self.embed_one(s) for s in smiles_list]).astype(np.float32)
