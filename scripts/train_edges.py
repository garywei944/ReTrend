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
import sklearn.metrics


parser = argparse.ArgumentParser(description='ReTrend Model Training Script')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
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
num_node = 200
seq_length = 4
window = seq_length + 1
data = KeywordCoocurrenceData(keyword_coocurrence, keyword_embeddings, topk=num_node, device=device, window=window)
# torch.save(data, f'data{os.sep}security-train-data.pt')
# data = torch.load(f'data{os.sep}security-train-data.pt')
# data: KeywordCoocurrenceData = torch.load(f'data{os.sep}train-data-sample.pt')
trainloader = [data[i] for i in range(len(data) - 1)]
trained_keywords = data.selected_keywords
test_data = data[-1]

model = ReTrend_Transformer(num_node, seq_length, node_pred=False).cuda()

model.init_embed(data.keyword_embeddings)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
criterion = nn.BCEWithLogitsLoss()

@torch.no_grad()
def predict(output):
    return torch.expm1(output[-1, :]).squeeze()

@torch.no_grad()
def make_binary(x):
    return (x > 0).float()

@torch.no_grad()
def as_binary_adjaceny(E_idx):
    A = torch.zeros(seq_length, num_node, num_node).cuda()
    for ti, row_col in enumerate(E_idx):
        A[ti, row_col[0], row_col[1]] = 1.0
    return A

@torch.no_grad()
def evaluate():
    model.eval()
    unshifted_E_idx, unshifted_E_w, shifted_E_idx, shifted_E_w, _ = test_data
    output = model(unshifted_E_idx, [torch.log1p(e) for e in unshifted_E_w])  
    output = output.squeeze()
    shifted_E = as_binary_adjaceny(shifted_E_idx)
    loss = criterion(output, shifted_E)
    last_year_pred, last_year_true = (output >= 0).float()[-1].view(-1), shifted_E[-1].view(-1)
    acc = 100 * (last_year_pred == last_year_true).sum() / len(last_year_pred)
    precision, recall, f_score, _ = sklearn.metrics.precision_recall_fscore_support(last_year_pred.cpu().numpy(), last_year_true.cpu().numpy(), zero_division=0)
    print(f'acc {acc:.2f}| precision {100 * precision[1]:.2f} | recall {100 * recall[1]:.2f} | F score {100 * f_score[1]:.2f}')
    return loss


def train(counter):
    model.train()
    total_loss = 0.
    random.shuffle(trainloader)
    for unshifted_E_idx, unshifted_E_w, shifted_E_idx, shifted_E_w, _ in trainloader:
        model.zero_grad()
        output = model(unshifted_E_idx, [torch.log1p(e) for e in unshifted_E_w])
        shifted_E = as_binary_adjaceny(shifted_E_idx)
        loss = criterion(output.squeeze(), shifted_E)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        counter.update(1)

    cur_loss = total_loss / len(trainloader)
    print('| epoch {:3d} | loss {:5.3f}'.format(epoch, cur_loss))
    total_loss = 0


best_test_loss = torch.inf
counter = tqdm(range(args.epochs * len(trainloader)))
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        train(counter)
        test_loss = evaluate()
        if best_test_loss > test_loss:
            best_test_loss = test_loss
        print('-' * 89)
        print('| end of epoch {:3d} | test loss {:5.3f} | best test loss {:5.3f}'.format(epoch, test_loss, best_test_loss))
        print('-' * 89)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# torch.save(best_pred, f'data{os.sep}retrend-best-pred-{args.model}.pt')
