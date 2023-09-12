from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import math
from src.utils.sparse_utils import sp_item
from .SemNet import SemNet


class SemNetDataset(Dataset):
    def __init__(
            self,
            semnet: SemNet,
            end_year: int = 2017,
            pred_n_years: Optional[int] = 3,
            device='cuda:0'
    ):
        self.train_data = pred_n_years is not None

        target_year = end_year + pred_n_years
        assert pred_n_years or target_year <= semnet.kcn.end_year, \
            f'No ground truth data for year {target_year}'

        n = semnet.kcn.n_keywords
        self.n_edges = n * (n - 1) // 2

        start_year = semnet.kcn.start_year
        n_prev_years = semnet.n_prev_years
        print(f'Start to build SemNet dataset from {start_year} to {end_year}')
        print(f'Predicting {target_year}')

        end_idx = end_year - start_year - n_prev_years
        target_idx = target_year - start_year - n_prev_years

        feats = semnet.feats[end_idx]

        print('Start to build features, this may take a while...')
        # # Stack features to speed up
        # _indices, _values = [], []
        # _c = 0
        # for ii in range(n):
        #     for jj in range(ii):
        #         _feats = [
        #             feats[0][ii].item(),
        #             feats[0][jj].item(),
        #             feats[1][ii].item(),
        #             feats[1][jj].item()
        #         ]
        #         sp_feats = [
        #             feat[ii, jj] for feat in feats[2:]
        #         ]
        #         _feats.extend([sp_item(e) for e in sp_feats])
        #         _feats = torch.tensor(_feats).float()
        #
        #         _indices.append((ii, jj))
        #         _values.append(_feats)
        #
        #         _c += 1
        #         if _c % 1000 == 999:
        #             print(f'{_c} / {self.n_edges}')
        #
        # _indices = torch.tensor(zip(_indices), dtype=torch.int)
        # self.feats = torch.sparse_coo_tensor(
        #     _indices, _values, (n, n)
        # ).coalesce()

        feat1 = feats[0][:, None].expand(-1, n)
        feat2 = feats[0][None, :].expand(n, -1)
        feat3 = feats[1][:, None].expand(-1, n)
        feat4 = feats[1][None, :].expand(n, -1)

        feats = [
            e.to_dense()
            for e in feats[2:]
        ]
        feats = [feat1, feat2, feat3, feat4] + feats
        feats = torch.stack(feats, dim=-1)

        # Only store the lower triangle
        I, J = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
        musk = I < J
        self.feats = feats[musk]

        # Build labels, summing up all labels in pred_n_years
        # if self.train_data:
        labels = semnet.kcn.ind_Es[target_idx]
        for i in range(1, pred_n_years):
            labels = labels + semnet.kcn.ind_Es[target_idx-i]
        self.labels = (labels.to_dense() > 0)[musk]

        self.n = n
        self.device = device
        # self.feat_dim = len(self.feats) + 2
        self.feat_dim = self.feats.shape[-1]

        self.start_year = start_year
        self.end_year = end_year
        self.target_year = target_year

    def __getitem__(self, index):
        # n = self.n
        # try:
        #     ii, jj = index
        # except TypeError:
        #     # ii, jj = index // n, index % n  # BUG
        #     # if only 1 int is provided, use the lower triangle to compute
        #     # the correct index
        #     # https://math.stackexchange.com/a/2390129
        #     index = index % self.__len__()
        #
        #     ii = math.floor((math.sqrt(1 + 8 * index) - 1) / 2)
        #     jj = int(index - (ii ** 2 + ii) / 2)
        #     ii += 1  # We are not using the diagnal
        #
        #     # assert 0 <= ii < n and 0 <= jj < n, f'{index} {ii} {jj}'
        #
        # feats = self.feats[ii, jj].to(self.device)
        #
        # if self.train_data:
        #     labels = float(sp_item(self.labels[ii, jj]))
        #     return feats, torch.tensor(labels).to(self.device)
        #
        # return feats

        try:
            index = index % self.n_edges
        except TypeError:
            ii, jj = index
            ii, jj = ii % self.n, jj % self.n
            index = int((ii ** 2 - ii) / 2 + jj)
        feats = self.feats[index].to(self.device)

        if self.train_data:
            # labels = float(sp_item(self.labels[ii]))
            labels = self.labels[index].float().to(self.device)
            return feats, labels

        return feats

    def __len__(self):
        return self.n_edges

    def size(self):
        return torch.tensor([self.n, self.n])

    def get_labels(self):
        return self.labels.float()
