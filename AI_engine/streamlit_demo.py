import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from pathlib import Path

import re
import pytz
from datetime import datetime

import enum
#import os
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay, auc, roc_curve

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import joblib

import load_data
import sk_classifier_builder as skb

p = Path('.')

save_eval_path = p / "AI_engine/evaluation_results"
save_model_path = p / "AI_engine/saved_models"
datapath = p / "AI_engine/test_data/"

data = np.load(datapath / "synthetic_dataset.npy")

#print("shape of  data is ",data.shape)

x = data[:, :data.shape[1]-1]  # data
y = data[:, -1] # label

#print("shape of x is ",x.shape)
#print("shape of y is ",y.shape)

# normalization on input data x
x = (x - x.mean(axis=0)) / x.std(axis=0)

# Use line below with PV_Data Only
#x = np.delete(x, 799999, 1)  # delete second column of C

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

#training_set = np.concatenate((X_train, y_train.reshape(-1,1)) ,axis=1)
#test_set = np.concatenate((X_test, y_test.reshape(-1,1)) ,axis=1)

#print(x)
#print(X_train)

decision_tree = skb.pipeBuild_DecisionTreeClassifier(max_depth=[3, 5, 10])
random_forest = skb.pipeBuild_RandomForestClassifier(n_estimators=[5,10], max_depth=[3, 5, 10],max_features=[1])

names = ['Decision Tree','Random Forest']
pipes = [decision_tree,random_forest]


titles = []
for t in names:
    tn = t + ' Train'
    ts = t + ' Test'
    titles.append(tn)
    titles.append(ts)


fig1 = make_subplots(rows=2,cols=1,subplot_titles = ['Training Data','Testing Data'])

#"""
i = 1

x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5

# Plot Training Set
#fig1.append_trace(go.Scatter(x = X_train[:, 0],y = X_train[:, 1],),row=i,col=1)

# Plot Testing Set
#fig1.append_trace(go.Scatter(x = X_test[:, 0],y = X_test[:, 1],),row=i,col=1)

# iterate over classifiers
for j in range(len(names)):
    
    grid_search = GridSearchCV(pipes[j][0], pipes[j][1], cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    score = grid_search.score(X_test, y_test)
    print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
    print(grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_estimator(grid_search, X_test, y_test, xticks_rotation="vertical")

    #fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    #roc_auc = auc(fpr, tpr)
    #RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name=grid_search)

plt.tight_layout()
plt.show()
#fig1.show()
#"""