import ijson
import os
from tqdm.auto import tqdm
import pickle
from collections import Counter
import pandas as pd
import numpy as np
import torch
import gc

# processed_data = []
# abstract_data = []
# keywords_data = set()
# remaining, total = 0, 0
# computer_security_data = []
# ai_data = []

# with open(f'data{os.sep}dblpv13.json', 'r') as f:
#     for i, record in tqdm(enumerate(ijson.items(f, "item"))):
#         total += 1
#         if 'lang' not in record or record['lang'] != 'en': continue
#         if 'fos' not in record: continue
#         if 'title' not in record: continue
#         if '_id' not in record: continue
#         if 'year' not in record or record['year'] < 1990 or record['year'] >= 2020: continue
#         if 'keywords' not in record: continue
#         if 'authors' not in record: continue
#         if 'title' not in record: continue
#         if 'n_citation' not in record: continue
#         if 'abstract' not in record or len(record['abstract']) < 300: continue
#         data = {
#             "fos": record['fos'],
#             'keywords': [k.lower().strip() for k in record['keywords']],
#             '_id': record['_id'],
#             'year': record['year'],
#             'authors': record['authors'],
#             'n_citation': record['n_citation'],
#             'references': record['references'] if 'references' in record else [],
#             'abstract': record['abstract']
#         }
#         abstract_data.append(record['abstract'])
#         processed_data.append(data)
#         keywords_data.update(data['keywords'])
#         remaining += 1

#         if 'Artificial intelligence' in data['fos']:
#             ai_data.append(data)

#         if 'Computer security' in data['fos']:
#             computer_security_data.append(data)


# print(f"remaining articles: {remaining} ({100 * remaining / total:.2f}%, or {remaining} / {total})")

# with open(f'data{os.sep}processed-data.pickle', 'wb') as f:
#     pickle.dump(processed_data, f)

# with open(f'data{os.sep}abstract-data.pickle', 'wb') as f:
#     pickle.dump(abstract_data, f)

# with open(f'data{os.sep}keywords-list.pickle', 'wb') as f:
#     pickle.dump(list(keywords_data), f)


# with open(f'data{os.sep}ai-data.pickle', 'wb') as f:
#     pickle.dump(ai_data, f)

# with open(f'data{os.sep}computer-security-data.pickle', 'wb') as f:
#     pickle.dump(computer_security_data, f)


# with open(f'data{os.sep}ai-data.pickle', 'rb') as f:
#     ai_data = pickle.load(f)

# with open(f'data{os.sep}computer-security-data.pickle', 'rb') as f:
#     computer_security_data = pickle.load(f)

with open(f'data{os.sep}processed-data.pickle', 'rb') as f:
    processed_data = pickle.load(f)

with open(f'data{os.sep}keywords-list.pickle', 'rb') as f:
    keywords_data = pickle.load(f)


# fos_counter = Counter()
# for p in tqdm(processed_data):
#     fos_counter.update(p['fos'])

# with open(f'data{os.sep}fos_counter.pickle', 'wb') as f:
#     pickle.dump(fos_counter, f)


keywords_idx_map = {k: i for i, k in enumerate(keywords_data)}

keywords_years_map = dict()
def inc_keywords_years_map(*args):
    if args in keywords_years_map: 
        keywords_years_map[args] += 1
    else:
        keywords_years_map[args] = 1


keywords_years = dict()
for line in tqdm(processed_data):
    yr = line['year'] - 1990
    for k1 in line['keywords']:
        for k2 in line['keywords']:
            inc_keywords_years_map(yr, keywords_idx_map[k1], keywords_idx_map[k2])    


with open(f'data{os.sep}keywords-cooccurence-counter.pickle', 'wb') as f:
    pickle.dump(keywords_years_map, f)

# with open(f'data{os.sep}keywords-cooccurence-counter.pickle', 'rb') as f:
#     keywords_years_map = pickle.load(f)

T = 30
K = 6772408
# T = len(Counter([p[0] for p in keywords_years_map.keys()]))
# K = len(keywords_data)
print(len(keywords_years_map))

del processed_data
del keywords_data

indices = torch.zeros(3, len(keywords_years_map), dtype=torch.int64)
values  = torch.zeros(len(keywords_years_map), dtype=torch.int64)
counter = tqdm(range(len(keywords_years_map)), miniters=(len(keywords_years_map)/100))
for i, (p, cnt) in enumerate(keywords_years_map.items()):
    indices[0, i] = p[0]
    indices[1, i] = p[1]
    indices[2, i] = p[2]
    values[i] = cnt
    counter.update(1)
    if i % 1e5 == 0: gc.collect()

torch.save(indices, f'data{os.sep}keyword_coocurrence-indices.pt')
torch.save(values, f'data{os.sep}keyword_coocurrence-values.pt')
# indices = torch.load(f'data{os.sep}keyword_coocurrence-indices.pt')
# values = torch.load(f'data{os.sep}keyword_coocurrence-values.pt')
keyword_coocurrence = torch.sparse_coo_tensor(indices, values, size=(T, K, K))
torch.save(keyword_coocurrence, f'data{os.sep}keyword_coocurrence.pt')

# keyword_coocurrence = torch.load(f'data{os.sep}keyword_coocurrence.pt')
