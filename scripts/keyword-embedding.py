import os
import pickle
import torch
from tqdm.auto import tqdm
import numpy as np
import gc


# paper_embeddings = torch.load(f'data{os.sep}security-paper-embeddings.pt')

# with open(f'data{os.sep}computer-security-data.pickle', 'rb') as f:
#     processed_data = pickle.load(f)
paper_embeddings = torch.load(f'data{os.sep}ai-paper-embeddings.pt')

with open(f'data{os.sep}ai-data.pickle', 'rb') as f:
    processed_data = pickle.load(f)

with open(f'data{os.sep}keywords-list.pickle', 'rb') as f:
    keywords_data = pickle.load(f)

keywords_idx_map = {k: i for i, k in enumerate(keywords_data)}
keywords_embeddings_map_time = dict()
keywords_embeddings_map = dict()
progress_counter = tqdm(range(len(processed_data)))

for i, d in enumerate(processed_data):
    yr = d['year'] - 1990
    for k in d['keywords']:
        pos = keywords_idx_map[k]
        if (yr, pos) in keywords_embeddings_map_time:
            keywords_embeddings_map_time[(yr, pos)].append(i)
        else:
            keywords_embeddings_map_time[(yr, pos)] = [i] 
        
        if pos in keywords_embeddings_map:
            keywords_embeddings_map[pos].append(i)
        else:
            keywords_embeddings_map[pos] = [i]
    progress_counter.update(1)
T = 30
K = 6772408

# k = len(keywords_embeddings_map_time)
# time_variant_indices = torch.zeros(2, k, dtype=torch.int64)
# time_variant_embeds = torch.zeros(k, 768)
# progress_counter = tqdm(range(k), miniters=(k//1e5))
# for i, ((yr, pos), embed_idx) in enumerate(keywords_embeddings_map_time.items()):
#     time_variant_embeds[i] = paper_embeddings[embed_idx].mean(axis=0)
#     time_variant_indices[0, i] = yr
#     time_variant_indices[1, i] = pos
#     progress_counter.update(1)

# time_variant_embeds = torch.sparse_coo_tensor(time_variant_indices, time_variant_embeds, size=(T, K, 768))
# # torch.save(time_variant_embeds, f'data{os.sep}security-keywords-embeddings-time-variant.pt')
# torch.save(time_variant_embeds, f'data{os.sep}ai-keywords-embeddings-time-variant.pt')

k = len(keywords_embeddings_map)
time_invariant_embeds = torch.zeros(K, 768)
progress_counter = tqdm(range(k), miniters=(k//100))
for i, (pos, embed_idx) in enumerate(keywords_embeddings_map.items()):
    time_invariant_embeds[pos] = paper_embeddings[embed_idx].mean(axis=0)
    progress_counter.update(1)

gc.collect()
# time_invariant_embeds = time_invariant_embeds.to_sparse()
# torch.save(time_invariant_embeds, f'data{os.sep}security-keywords-embeddings-time-invariant.pt')
# torch.save(time_invariant_embeds, f'data{os.sep}ai-keywords-embeddings-time-invariant.pt')
torch.save(time_invariant_embeds, f'data{os.sep}ai-keywords-embeddings-time-invariant-nonsparse.pt')
