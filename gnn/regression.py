import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split


with open('embeddings/user_embeddings.txt') as f:
  uemb = {}
  for line in f.readlines():
    linesplit = line.split()
    uemb[int(linesplit[0])] = np.array(list(map(float, linesplit[1:])))

with open('embeddings/item_embeddings.txt') as f:
  iemb = {}
  for line in f.readlines():
    linesplit = line.split()
    iemb[int(linesplit[0])] = np.array(list(map(float, linesplit[1:])))

dataset = movielens.load_pandas_df(size='100k')
train, test = python_stratified_split(data=dataset, ratio=0.75)

def convert(df):
  X = []
  y = []
  for i, row in df.iterrows():
    X.append(np.concatenate((uemb[int(row['userID'])], iemb[int(row['itemID'])])))
    y.append(row['rating'])
  X = np.array(X)
  y = np.array(y)
  return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

trainX, trainy = convert(train)
testX, testy = convert(test)

model = nn.Sequential(
  nn.Linear(128, 64),
  nn.Sigmoid(),
  nn.Linear(64, 32),
  nn.Sigmoid(),
  nn.Linear(32, 1)
)

trainloader = DataLoader(TensorDataset(trainX, trainy), batch_size=128, shuffle=True)
testloader = DataLoader(TensorDataset(testX, testy), batch_size=128)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.00001)


for epoch in tqdm(range(20)):
  losssum = 0
  losscount = 0
  for X, y in trainloader:
    preds = model(X).view(-1)
    loss = F.mse_loss(preds, y)
    losssum += loss.item()
    losscount += 1
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(f'{epoch}: {losssum/losscount}')

  with torch.no_grad():
    for X, y in testloader:
      preds = model(X)
      loss = F.mse_loss(preds, y)


import pdb; pdb.set_trace()