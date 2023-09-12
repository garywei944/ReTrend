import ijson
import os
from tqdm.auto import tqdm
import pickle
from collections import Counter
import pandas as pd
import numpy as np
import torch



T = 30
K = 6772408

with open(f'data{os.sep}ai-data.pickle', 'rb') as f:
    ai_data = pickle.load(f)

with open(f'data{os.sep}keywords-list.pickle', 'rb') as f:
    keywords_data = pickle.load(f)

keywords_idx_map = {k: i for i, k in enumerate(keywords_data)}

keywords_years_map = dict()
def inc_keywords_years_map(*args):
    if args in keywords_years_map: 
        keywords_years_map[args] += 1
    else:
        keywords_years_map[args] = 1


keywords_years = dict()
for line in tqdm(ai_data):
    yr = line['year'] - 1990
    for k1 in line['keywords']:
        for k2 in line['keywords']:
            inc_keywords_years_map(yr, keywords_idx_map[k1], keywords_idx_map[k2])    

with open(f'data{os.sep}ai-keywords-cooccurence-counter.pickle', 'wb') as f:
    pickle.dump(keywords_years_map, f)

# with open(f'data{os.sep}ai-keywords-cooccurence-counter.pickle', 'rb') as f:
#     keywords_years_map = pickle.load(f)

indices = torch.zeros(3, len(keywords_years_map), dtype=torch.int64)
values  = torch.zeros(len(keywords_years_map), dtype=torch.int64)
for i, (p, cnt) in tqdm(enumerate(keywords_years_map.items())):
    indices[0, i] = p[0]
    indices[1, i] = p[1]
    indices[2, i] = p[2]
    values[i] = cnt


# indices = torch.load(f'data{os.sep}ai-keyword_coocurrence-indices.pt')
# values = torch.load(f'data{os.sep}ai-keyword_coocurrence-values.pt')
torch.save(indices, f'data{os.sep}ai-keyword_coocurrence-indices.pt')
torch.save(values, f'data{os.sep}ai-keyword_coocurrence-values.pt')
keyword_coocurrence = torch.sparse_coo_tensor(indices, values, size=(T, K, K))
torch.save(keyword_coocurrence, f'data{os.sep}ai-keyword_coocurrence.pt')
