from pathlib import Path

import numpy as np
import pandas as pd
import torch

from typing import List, Dict, Union
from torch_sparse import SparseTensor
from collections import defaultdict

from src.utils.sparse_utils import sp_over_c

from .KCN import KCN


class SemNet:
    kcn: KCN

    feats: List[List[Union[torch.Tensor, SparseTensor]]]
    n_prev_years: int
    path_len_up_to: int

    def __init__(
            self,
            kcn: KCN,
            n_prev_years: int = 2,
            path_len_up_to: int = 4
    ):
        _start, _end = kcn.start_year, kcn.end_year
        n_years = _end - _start + 1
        n = kcn.n_keywords

        print('Start to build SemNet using')
        print(f'\t{n_prev_years} previous years')
        print(f'\tpath length up to {path_len_up_to}')

        paths_w_len = defaultdict(list)
        feats = defaultdict(list)

        for i, (V, E) in enumerate(zip(kcn.Vs, kcn.Es)):
            print(f'Building {i + 1} / {n_years} year')
            E_adj = E.set_value(
                torch.ones(E.numel()), 'coo'
            )
            degree: torch.Tensor = E_adj.sum(dim=0)

            # feat1 = degree[:, None].expand(
            #     -1, n) / degree_max  # (n, n), no memory replication
            # feat2 = degree[None, :].expand(n, -1) / degree_max

            feat1 = degree / degree.max()

            # feat3 = V[:, None].expand(
            #     -1, n) / V_max  # (n, n), no memory replication
            # feat4 = V[None, :].expand(n, -1) / V_max
            feat3 = V / V.max()

            # cos_similarity according to source code
            E_pow = E_adj @ E_adj  # number of paths with length 2, barely to save computation

            # The following code could be done easily in dense tensor, however,
            # I have to assume we cannot make a dense tensor here for limited
            # memory
            _indices = [E_pow.storage.row(), E_pow.storage.col()]
            _denom = torch.tensor([
                degree[ii] * degree[jj]
                for ii, jj in zip(*_indices)
            ])
            _values = E_pow.storage.value() / _denom
            feat5 = E_pow.set_value(_values, 'coo')

            if n_prev_years >= 2:
                paths_w_len[i].append(
                    sp_over_c(E_pow, E_pow.max())
                )
                for _ in range(path_len_up_to - 2):
                    E_pow = E_adj @ E_pow
                    paths_w_len[i].append(
                        sp_over_c(E_pow, E_pow.max())
                    )

            # Add features
            feats[i] = [feat1, feat3, feat5]
            feats[i].extend(paths_w_len[i])

            # TODO: 3 last features are corresponding to the distance

        # Add other features
        for i in range(n_prev_years, n_years):
            for j in range(1, n_prev_years + 1):
                feats[i].extend(paths_w_len[i - j])

        self.kcn = kcn
        self.n_prev_years = n_prev_years
        self.path_len_up_to = path_len_up_to

        self.feats = [
            feats[i] for i in range(n_prev_years, n_years)
        ]
        self.feat_len = len(self.feats[0]) + 2

        print(
            f'Built SemNet from {_start} to {_end} with {self.feat_len} features.')

    def save(self, file_path: Union[str, Path]):
        kcn = self.kcn
        self.kcn = None
        torch.save(self, file_path)
        self.kcn = kcn
        print(f'Saved SemNet to {file_path}')

    @staticmethod
    def load(file_path: Union[str, Path], kcn: KCN):
        print(f'Loading SemNet from {file_path}')
        obj = torch.load(file_path)
        obj.kcn = kcn

        return obj
