"""Load PyTorch Lightning checkpoints without installing pytorch-lightning.

Pre-trained MolBERT and Chemformer checkpoints pickle a few Lightning objects
(e.g. hyper-parameter containers). Unknown classes are replaced by plain
dict-like placeholders; tensors and standard containers load normally.
"""
import pickle

import torch


class _Placeholder(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.update(state)


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Placeholder


class _PickleModule:
    __name__ = "pickle"
    Unpickler = _TolerantUnpickler
    UnpicklingError = pickle.UnpicklingError
    load = pickle.load


def load_checkpoint(path):
    # torch 1.13 ignores the `pickle_module` argument, so swap the module it uses for the duration of the call.
    original = torch.serialization.pickle
    torch.serialization.pickle = _PickleModule
    try:
        return torch.load(path, map_location="cpu")
    finally:
        torch.serialization.pickle = original
