'''
Author: Qi7
Date: 2023-02-28 22:07:04
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-03-01 09:48:26
Description: 
'''
from utils.sliding_window import sliding_windows
import matplotlib.pyplot as plt
from utils.dataloader import MyLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn 
from torch.nn import functional as F
from torch.optim import Adam
from AI_engine.models import LSTMRegressor
from AI_engine.training import *
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error

saved_npy = "dataset/cpu_temp_pi4.npy"
data_df = pd.read_csv("dataset/cpu_temp_pi4.csv")
data_df = data_df.set_index('timestamp')
# print(data_df.head())

feature_cols = ['cpu_temperature'] # only one features in this case
X = data_df[feature_cols].values.flatten() #numpy array -> list

X_organized, Y_organized = sliding_windows(X.flatten(), 30, 1, 0)

npy_data = np.concatenate((X_organized, np.expand_dims(Y_organized, axis=1)), axis = 1)


# save the dataset as npy file
# print(npy_data.shape)
# with open(saved_npy, 'wb') as f:
#     np.save(f, npy_data)

# construct the dimenstion to LSTM format

X_organized = X_organized.reshape(X_organized.shape + (1,)).astype(np.float32)
Y_organized = Y_organized.reshape(Y_organized.shape + (1,)).astype(np.float32)


X_train, X_test, y_train, y_test = train_test_split(X_organized, Y_organized, test_size=0.25, shuffle=False)

# normalization here ...

train_dataset = MyLoader(X_train, y_train)
test_dataset = MyLoader(X_test, y_test)

train_loader = DataLoader(train_dataset, shuffle=False, batch_size=32)
test_loader  = DataLoader(test_dataset,  shuffle=False, batch_size=32)

# training hyperparameters
epochs = 20
learning_rate = 1e-3

loss_fn = nn.MSELoss()
lstm_regressor = LSTMRegressor(n_features=1)
optimizer = Adam(lstm_regressor.parameters(), lr=learning_rate)

# training process
train_model(
    model=lstm_regressor,
    loss_fn=loss_fn,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=epochs
)


torch.save(lstm_regressor.state_dict(), "lstm.pt")

# Model class must be defined somewhere
# lstm_regressor.load_state_dict(torch.load("lstm.pt"))
lstm_regressor.eval()

test_preds = lstm_regressor(torch.from_numpy(X_test))
print(test_preds[:5])

print("Test  MSE : {:.2f}".format(mean_squared_error(test_preds.detach().numpy().squeeze(), y_test.squeeze())))
print("Test  R^2 Score : {:.2f}".format(r2_score(test_preds.detach().numpy().squeeze(), y_test.squeeze())))

plt.figure(figsize=(10,8))
plt.plot(test_preds.detach().numpy().squeeze())
plt.plot(y_test.squeeze())
plt.show()