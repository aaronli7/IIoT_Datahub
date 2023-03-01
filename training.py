'''
Author: Qi7
Date: 2023-02-28 22:07:04
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-02-28 23:03:50
Description: 
'''
from utils.sliding_window import sliding_windows
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_df = pd.read_csv("dataset/cpu_temp_pi4.csv")
data_df = data_df.set_index('timestamp')
# print(data_df.head())

feature_cols = ['cpu_temperature'] # only one features in this case
X = data_df[feature_cols].values.flatten() #numpy array -> list

X_organized, Y_organized = sliding_windows(X.flatten(), 30, 3, 0)

