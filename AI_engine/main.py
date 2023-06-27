import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from pathlib import Path

import influxdb_client
import tsai
#from tsai.all import *

import re
import pytz
from datetime import datetime

import enum
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
#from data_loader import SiameseNetworkDataset
#from models import LSTM
#from customized_criterion import ContrastiveLoss
from torchinfo import summary
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

import load_data

x = load_data.load('X.npy')
y = load_data.load('y.npy')

# normalization on input data x
x = (x - x.mean(axis=0)) / x.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

#training_set = np.concatenate((X_train, y_train.reshape(-1,1)) ,axis=1)
#test_set = np.concatenate((X_test, y_test.reshape(-1,1)) ,axis=1)

X_train = np.transpose(X_train,(0,2,1))
X_test = np.transpose(X_test,(0,2,1))

#y_train = np.expand_dims(y_train,axis=1)
#y_test = np.expand_dims(y_test,axis=1)

training_set = waveformDataset(X_train, y_train)
test_set = waveformDataset(X_test, y_test)