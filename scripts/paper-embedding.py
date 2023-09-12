import os
import pickle
import torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import sklearn.decomposition
import numpy as np
import gc

with open(f'data{os.sep}ai-data.pickle', 'rb') as f:
    ai_data = pickle.load(f)

# abstract_data = [record['abstract'] for record in ai_data]
# with open(f'data{os.sep}abstract-data.pickle', 'rb') as f:
#     abstract_data = pickle.load(f)

# with open(f'data{os.sep}computer-security-data.pickle', 'rb') as f:
#     processed_data = pickle.load(f)
abstract_data = [record['abstract'] for record in ai_data]


model = torch.load(f'model{os.sep}finetuned-scibert.pt').cuda()
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', model_max_length=256)

tokenized_abstract_full = tokenizer(abstract_data, return_special_tokens_mask=True, truncation=True, padding='max_length', return_tensors='pt')

# del abstract_data

with open(f'data{os.sep}tokenized_security_abstract_full.pickle', 'wb') as f:
    pickle.dump(tokenized_abstract_full, f)

# with open(f'data{os.sep}tokenized_abstract_full.pickle', 'wb') as f:
#     pickle.dump(tokenized_abstract_full, f)

# with open(f'data{os.sep}tokenized_abstract_full.pickle', 'rb') as f:
#     tokenized_abstract_full = pickle.load(f)


paper_embeddings = []
with torch.no_grad():
    for i in tqdm(range(0, len(tokenized_abstract_full['input_ids']), 4)):
        if i % 100 == 0:
            gc.collect()
        torch.cuda.empty_cache()
        batch = torch.arange(i, min(i + 4, len(tokenized_abstract_full['input_ids'])))
        data = {
            'input_ids': torch.tensor(tokenized_abstract_full['input_ids'][batch], dtype=torch.int64).cuda(),
            'attention_mask': torch.tensor(tokenized_abstract_full['attention_mask'][batch], dtype=torch.int64).cuda()
        }
        outputs = model.bert(**data)[0]
        paper_embeddings.append(outputs.mean(axis=1).cpu())

paper_embeddings = torch.vstack(paper_embeddings)
torch.save(paper_embeddings, f'data{os.sep}security-paper-embeddings.pt')

# paper_embeddings = torch.load(f'data{os.sep}all-paper-embeddings.pt')
# torch.save(paper_embeddings, f'data{os.sep}all-paper-embeddings.pt')

# pca_paper_embeddings_np = sklearn.decomposition.PCA(n_components=2).fit_transform(paper_embeddings.numpy())
# np.save(f'data{os.sep}pca-all-paper-embeddings.npy', pca_paper_embeddings_np)