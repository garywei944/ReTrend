import torch
from torch.utils.data import Dataset
import os
from torch_sparse import SparseTensor
import pickle


class KeywordCoocurrenceData(Dataset):
    def __init__(self, keyword_coocurrence, keyword_embeddings, window=5, topk=10000, device=None) -> None:
        keyword_coocurrence_summed = torch.sparse.sum(keyword_coocurrence, 0)
        row, col, value = SparseTensor.from_torch_sparse_coo_tensor(keyword_coocurrence_summed).coo()
        topk_value_pos = torch.argsort(-value[row == col])[:topk]
        keyword_idx = row[row == col][topk_value_pos]
        keyword_coocurrence = [SparseTensor.from_torch_sparse_coo_tensor(M)[keyword_idx, keyword_idx] for M in keyword_coocurrence]
        self.keyword_coocurrence = keyword_coocurrence
        if keyword_embeddings.is_sparse:
            self.keyword_embeddings = SparseTensor.from_torch_sparse_coo_tensor(keyword_embeddings)[keyword_idx].to_dense()
        else:
            self.keyword_embeddings = keyword_embeddings[keyword_idx]
        # seq length = window - 1
        # for example, input [0, 1, 2] vs labels [1, 2, 3] for a window of 4
        self.cnt, self.seq_len, self.K = len(keyword_coocurrence) // window, window - 1, len(keyword_idx)
        self.seq_indices = torch.split(torch.arange(len(self.keyword_coocurrence)), window)
        self.keyword_idx = keyword_idx
        self.num_node = topk
        self.device = device
        with open(f'data{os.sep}keywords-list.pickle', 'rb') as f: keywords_data = pickle.load(f)    
        self.selected_keywords = [keywords_data[i] for i in self.keyword_idx]

    def __getitem__(self, idx):
        seq_indices = self.seq_indices[idx]

        Gs_unshifted = [self.keyword_coocurrence[s] for s in seq_indices[:-1]]
        Gs_shifted   = [self.keyword_coocurrence[s] for s in seq_indices[1:]]
 
        unshifted_Gs = [G.coo() for G in Gs_unshifted]
        unshifted_E_idx = [torch.vstack([row, col]).to(self.device) for (row, col, _) in unshifted_Gs]
        unshifted_E_weights = [values.to(self.device).float() for (_, _, values) in unshifted_Gs]

        shifted_Gs = [G.coo() for G in Gs_shifted]
        shifted_E_idx = [torch.vstack([row, col]).to(self.device) for (row, col, _) in shifted_Gs]
        shifted_E_weights = [values.to(self.device).float() for (_, _, values) in shifted_Gs]

        unshifted_node_cnt = torch.vstack([torch.diagonal(G.to_dense().to(self.device).float()) for G in Gs_unshifted])
        shifted_node_cnt = torch.vstack([torch.diagonal(G.to_dense().to(self.device).float()) for G in Gs_shifted])
        return unshifted_E_idx, unshifted_E_weights, shifted_E_idx, shifted_E_weights, unshifted_node_cnt, shifted_node_cnt

    def __len__(self):
        return self.cnt


def _get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x), device=x.device)
    return ranks

def spearman_correlation(x: torch.Tensor, y: torch.Tensor):
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    return 1.0 - (upper / down)

def pearson_correlation(x: torch.Tensor, y: torch.Tensor):
    centered_x = x - torch.mean(x)
    centered_y = y - torch.mean(y)
    return torch.sum(centered_x * centered_y) / (torch.sqrt(torch.sum(centered_x ** 2)) * torch.sqrt(torch.sum(centered_y ** 2)))


class PredictionResult:
    def __init__(self, prediction_2018, prediction_2019, truth_2018, ground_truth_2019, keywords) -> None:
        self.prediction_2018 = prediction_2018
        self.prediction_2019 = prediction_2019
        self.ground_truth_2018 = truth_2018
        self.ground_truth_2019 = ground_truth_2019
        self.keywords =  keywords
        print(f'2019 spearman correlation {spearman_correlation(prediction_2019, self.ground_truth_2019)}')
        print(f'2019 pearson correlation {pearson_correlation(prediction_2019, self.ground_truth_2019)}')
        
