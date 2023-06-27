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

p = Path('.')
datapath = p / "test_data/"