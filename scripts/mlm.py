import argparse
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoModel, AutoTokenizer,
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    Trainer
)
import evaluate
import tokenizers
import torch
from torch.utils.data import Dataset
from itertools import chain
import pickle
import os
import math
import numpy as np
import sklearn.model_selection


# with open(f'data{os.sep}abstract-data.pickle', 'rb') as f:
#     abstract_data = pickle.load(f)

# raw_length = len(abstract_data)
# abstract_data = [abstract_data[i] for i in torch.randperm(raw_length)[:int(0.1 * raw_length)]]

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', model_max_length=256)
model = AutoModelForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased').cuda()


# tokenized_abstact = tokenizer(abstract_data, return_special_tokens_mask=True, truncation=True, padding='max_length')
# with open('data/tokenized_abstract.pickle', 'rb') as f:
#     tokenized_abstact = pickle.load(f)

# with open('data/tokenized_abstract.pickle', 'wb') as f:
#     pickle.dump(tokenized_abstact, f)


train_indices, test_indices = \
    sklearn.model_selection.train_test_split(np.arange(len(tokenized_abstact['input_ids'])), test_size=0.2, random_state=0)

train_tokens = {
    'input_ids'     : [tokenized_abstact['input_ids'][i] for i in train_indices],
    'attention_mask': [tokenized_abstact['attention_mask'][i] for i in train_indices],
}
test_tokens = {
    'input_ids'     : [tokenized_abstact['input_ids'][i] for i in test_indices],
    'attention_mask': [tokenized_abstact['attention_mask'][i] for i in test_indices],
}


class AbstractTextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, i):
        return {
            'input_ids': torch.tensor(self.data['input_ids'][i], dtype=torch.int64).cuda(),
            'labels'   : torch.tensor(self.data['input_ids'][i], dtype=torch.int64).cuda(),
            'attention_mask': torch.tensor(self.data['attention_mask'][i], dtype=torch.int64).cuda()
        }


train_dataset = AbstractTextDataset(train_tokens)
eval_dataset = AbstractTextDataset(test_tokens)


# def preprocess_logits_for_metrics(logits, labels):
#     if isinstance(logits, tuple):
#         # Depending on the model and config, logits may contain extra tensors,
#         # like past_key_values, but logits always come first
#         logits = logits[0]
#     return logits.argmax(dim=-1)

metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    # mlm_probability=0.15,
)

# Initialize our Trainer
trainer = Trainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training

train_result = trainer.train()

metrics = train_result.metrics

metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# Evaluation
metrics = trainer.evaluate()
metrics["eval_samples"] = len(eval_dataset)
try:
    perplexity = math.exp(metrics["eval_loss"])
except OverflowError:
    perplexity = float("inf")
metrics["perplexity"] = perplexity

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

torch.save(model, f'model{os.sep}finetuned-scibert.pt')
