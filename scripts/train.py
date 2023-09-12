# coding: utf-8
import argparse
import os
import torch
import torch.nn as nn
from model_retrend import *
from basic_model import *
from data_loader import *
from torch.utils.data import DataLoader
import random
from tqdm.auto import tqdm
import copy

parser = argparse.ArgumentParser(description='ReTrend Model Training Script')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--model', type=str, default='retrendv0')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda')
keyword_coocurrence = torch.load(f'data{os.sep}security-keyword_coocurrence.pt')
keyword_embeddings = torch.load(f'data{os.sep}security-keywords-embeddings-time-invariant.pt')
# keyword_coocurrence = torch.load(f'data{os.sep}ai-keyword_coocurrence.pt')
# keyword_embeddings = torch.load(f'data{os.sep}ai-keywords-embeddings-time-invariant.pt')
num_node = 10000
seq_length = 4
window = seq_length + 1
data = KeywordCoocurrenceData(keyword_coocurrence, keyword_embeddings, topk=num_node, device=device, window=window)
torch.save(data, f'data{os.sep}security-train-data-{window}.pt')
# data = torch.load(f'data{os.sep}ai-train-data-{window}.pt')
# torch.save(data, f'data{os.sep}security-train-data.pt')
# data: KeywordCoocurrenceData = torch.load(f'data{os.sep}ai-train-data.pt')
# data: KeywordCoocurrenceData = torch.load(f'data{os.sep}train-data-sample.pt')
trainloader = [data[i] for i in range(len(data) - 1)]
trained_keywords = data.selected_keywords
test_data = data[-1]

model = ReTrend_Transformer(num_node, seq_length).cuda()
# model = ReTrend(num_node).cuda()
# model = ReTrend_LSTM(num_node).cuda()
model.init_embed(data.keyword_embeddings)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

# import pdb; pdb.set_trace()
# unshifted_E_idx, unshifted_E_w, shifted_E_idx, shifted_E_w, unshifted_node_cnt, shifted_node_cnt = test_data
# prediction = unshifted_node_cnt[-1]
# prediction = prediction[500 : 2000].float()
# truth = shifted_node_cnt[-1, :][500 : 2000].float()
# keywords = trained_keywords[500 : 2000]
# pred_obj = PredictionResult(prediction, truth, keywords)
@torch.no_grad()
def predict(output):
    return torch.expm1(output).squeeze()

@torch.no_grad()
def evaluate(eval_low=50, eval_high=500):
    model.eval()
    # hidden = model.init_hidden(num_node)
    unshifted_E_idx, unshifted_E_w, shifted_E_idx, shifted_E_w, unshifted_node_cnt, shifted_node_cnt = test_data
    raw_output = model(unshifted_E_idx, unshifted_E_w).squeeze()
    # output, _ = model(test_E_indices, [torch.log1p(e) for e in test_E_weights], hidden)    
    # output, _ = model(hidden)    
    eval_output = raw_output[:, eval_low:eval_high]
    # loss = criterion(output, shifted_node_cnt[:, eval_low:eval_high])
    # loss = criterion(output.squeeze(), torch.log1p(test_node_cnt_true[-1, :]))
    truth_total = [g[eval_low:eval_high].float() for g in shifted_node_cnt]
    print(f'avg spearman {torch.tensor([spearman_correlation(eval_output[i], truth_total[i]) for i in range(seq_length)]).mean()}')
    print(f'avg pearson {torch.tensor([pearson_correlation(torch.log1p(eval_output[i]), torch.log1p(truth_total[i])) for i in range(seq_length)]).mean()}')
    prediction_2018 = predict(raw_output[-2, :eval_high]).float()
    prediction_2019 = predict(raw_output[-1, :eval_high]).float()
    truth_2018 = shifted_node_cnt[-2, :][: eval_high].float()
    truth_2019 = shifted_node_cnt[-1, :][: eval_high].float()
    keywords = trained_keywords[: eval_high]
    pred_obj = PredictionResult(prediction_2018, prediction_2019, truth_2018, truth_2019, keywords)
    last_year_spearman_pred = spearman_correlation(prediction_2019, truth_2019)
    return pred_obj, last_year_spearman_pred


def train(counter):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    # hidden = model.init_hidden(num_node)
    random.shuffle(trainloader)
    for unshifted_E_idx, unshifted_E_w, shifted_E_idx, shifted_E_w, unshifted_node_cnt, shifted_node_cnt in trainloader:
        model.zero_grad()
        # output = model(unshifted_E_idx, unshifted_E_w)
        output = model(unshifted_E_idx, [torch.log1p(e) for e in unshifted_E_w])
        # output, hidden = model(E_indices, [torch.log1p(e) for e in E_weights], hidden)
        # output, hidden = model(hidden)
        # loss = criterion(output.squeeze()[:, :5000], shifted_node_cnt[:, :5000])
        loss = criterion(output.squeeze()[:, :3000], torch.log1p(shifted_node_cnt)[:, :3000])
        # loss = criterion(output.squeeze()[:, 1000:3000], torch.log1p(shifted_node_cnt)[:, 1000:3000])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        counter.update(1)

    cur_loss = total_loss / len(trainloader)
    # lr_scheduler.step(cur_loss)
    print('| epoch {:3d} | loss {:5.2f}'.format(epoch, cur_loss))
    total_loss = 0

best_model = None
best_spearman_pred = -torch.inf
best_pred = None
counter = tqdm(range(args.epochs * len(trainloader)))
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        train(counter)
        next_year_pred, spearman_pred = evaluate()
        if best_spearman_pred < spearman_pred:
            best_spearman_pred = spearman_pred
            best_pred = next_year_pred
            best_model = copy.deepcopy(model)
        print('-' * 89)
        print('| end of epoch {:3d} | 2019 spearman perd {:5.2f} | best spearman pred {:5.2f}'.format(epoch, spearman_pred, best_spearman_pred))
        print('-' * 89)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# torch.save(best_pred, f'data{os.sep}retrend-best-pred-{args.model}-security.pt')
# torch.save(best_model, f'data{os.sep}retrend-best-model-{args.model}-security.pt')
