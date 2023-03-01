'''
Author: Qi7
Date: 2023-03-01 08:51:27
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-03-01 08:53:39
Description: 
'''
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import gc
import torch

def calc_val_loss(model, loss_fn, val_loader):
    with torch.no_grad():
        losses = []
        for X, Y in val_loader:
            preds = model(X)
            loss = loss_fn(preds.ravel(), Y)
            losses.append(loss.item())
        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))

def train_model(model, loss_fn, optimizer, train_loader, val_loader, epochs=10):
    for i in range(1, epochs+1):
        losses = []
        for X, Y in tqdm(train_loader):
            Y_preds = model(X)

            loss = loss_fn(Y_preds.ravel(), Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
        calc_val_loss(model, loss_fn, val_loader)