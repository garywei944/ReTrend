from typing import Any

import torch
from torch_sparse import SparseTensor


def sp_item(sp: SparseTensor) -> Any:
    return sp.storage.value().item() if sp.numel() != 0 else 0


def sp_mul_c(sp: SparseTensor, c: float) -> SparseTensor:
    return sp.set_value(sp.storage.value() * c, 'coo')


def sp_over_c(sp: SparseTensor, c: float) -> SparseTensor:
    return sp.set_value(sp.storage.value() / c, 'coo')


def sp_over_t(sp: SparseTensor, t: torch.Tensor) -> SparseTensor:
    value = sp.storage.value() / t[sp.storage.row(), sp.storage.col()]
    return sp.set_value(value, 'coo')


def sp_sub_sp(sp1: SparseTensor, sp2: SparseTensor) -> SparseTensor:
    sp2 = sp_mul_c(sp2, -1)
    return sp1 + sp2
