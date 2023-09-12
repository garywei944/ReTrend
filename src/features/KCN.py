from pathlib import Path

import pandas as pd
import torch

from typing import List, Dict, Union, Tuple
from torch_sparse import SparseTensor
from collections import defaultdict, Counter


class KCN:
    n_papers: int
    n_keywords: int
    keywords: List
    keyword_map: Dict

    # listed by years, i.e., each entry of the Vs or Es represents
    # the vertices/edges of that year

    # each vertex represents a keyword (aka. concept, method)
    # the value of that vertex is the number of papers that use it as keywords
    Vs: List[torch.Tensor]
    # each edge represents a keyword co-occurrence
    # the value represents how many papers use both as keywords
    # Upper Triangular Matrix representing adjacency matrix
    Es: List[SparseTensor]  # PyTorch COO sparse matrix

    start_year: int
    end_year: int

    def __init__(
            self,
            _df: pd.DataFrame,
            start_year: int = 1980,
            end_year: int = 2025,
            top_n_kws: int = 10000,
    ):
        print(f'Start to build KCN from {start_year} to {end_year}')

        # https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
        # _df = _df.loc[(_df['year'] >= start_year) & (_df['year'] <= end_year)]
        _df = _df.query('@start_year <= year <= @end_year')

        all_keywords = []
        for kws in _df.keywords:
            all_keywords.extend(kws)

        keywords_cnt = Counter(all_keywords)
        n = len(keywords_cnt)

        if top_n_kws is not None and n > top_n_kws:
            keywords = [e[0] for e in keywords_cnt.most_common(top_n_kws)]
            n = top_n_kws
        else:
            keywords = list(set(all_keywords))

        keyword_map = dict(zip(range(n), keywords))

        print(f'{_df.shape[0]} papers, {n} keywords')

        print('-' * 20)
        # Accumulative KCN, i.e., accumulate info from all previous years
        Vs, Es = [], []
        # Independent KCN, i.e., only compute KCN for each year
        ind_Vs, ind_Es = [], []
        for year in range(start_year, end_year + 1):
            _df_year = _df[_df['year'] == year]
            print(f'Building KCN for year {year}, {_df_year.shape[0]} papers')

            V, E = self.build_kcn_per_year(_df_year, keywords)
            Vs.append(V)
            Es.append(E)
            ind_Vs.append(V)
            ind_Es.append(E)

        # Accumulative, means that each year contains information from all
        # previous years
        # if accumulative:
        for i in range(1, end_year - start_year + 1):
            Vs[i] = Vs[i] + Vs[i - 1]
            Es[i] = Es[i] + Es[i - 1]

        self.start_year = start_year
        self.end_year = end_year

        self.n_papers = _df.shape[0]
        self.n_keywords = n
        self.keywords = keywords
        self.keyword_map = keyword_map

        self.Vs = Vs
        self.Es = Es
        self.ind_Vs = ind_Vs
        self.ind_Es = ind_Es

        print('-' * 20)
        print('Finish building KCN')

    @staticmethod
    def build_kcn_per_year(
            _df_year,
            keywords
    ) -> Tuple[torch.Tensor, SparseTensor]:
        n = len(keywords)

        # make a dict whose key: keyword, values: set of papers
        graph_dict = defaultdict(set)

        for i, paper in _df_year.iterrows():
            _id = paper.name
            kws = [kw for kw in paper['keywords'] if kw in keywords]

            for kw in kws:
                graph_dict[kw].add(_id)

        # Build keyword frequency, i.e., number of papers
        # in which a keyword is mentioned
        keywords_frequency = torch.zeros(n)
        for i, kw in enumerate(keywords):
            keywords_frequency[i] = len(graph_dict[kw])

        # start to build COO sparse matrix
        rows, cols = [], []
        values = []
        _c = 0
        for i in range(n):
            for j in range(i + 1, n):
                e = set.intersection(
                    graph_dict[keywords[i]],
                    graph_dict[keywords[j]]
                )
                if (e := len(e)) != 0:
                    rows.append(i)
                    cols.append(j)
                    values.append(e)
                    _c += 1
        graph = torch.sparse_coo_tensor(
            torch.tensor([rows, cols]), values, (n, n)
        )
        graph = SparseTensor.from_torch_sparse_coo_tensor(
            graph
        ).to_symmetric()

        # Actually represents V, E of the graph
        V, E = keywords_frequency, graph
        return V, E

    def save(self, file_path: Union[str, Path]):
        torch.save(self, file_path)
        print(f'Saved KCN to {file_path}')

    @staticmethod
    def load(file_path: Union[str, Path]):
        print(f'Loading KCN from {file_path}')
        return torch.load(file_path)
