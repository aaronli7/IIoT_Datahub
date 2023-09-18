from math import radians
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from pathlib import Path

#import streamlit as st

import re
import pytz
from datetime import datetime

import enum
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn import cluster, datasets, mixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay, auc, roc_curve, roc_auc_score


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import joblib

import load_data as ld
import sk_clustering_builder as skcl
import sk_classifier_metrics as skm

import sk_novelty_builder as skn


#st.title('Streamlit Demo')

p = Path('.')

save_eval_path = p / "AI_engine/evaluation_results/"
save_model_path = p / "AI_engine/saved_models/"
datapath = p / "AI_engine/test_data/"

#data = np.load("C:/Users/steph/OneDrive/Documents/GitHub/IIoT_Datahub/AI_engine/test_data/synthetic_dataset.npy")
#data = np.load(datapath / "synthetic_dataset.npy")
data = ld.selectFileAndLoad()


#"""
print("shape of  data is ",data.shape)

x = data[:, :data.shape[1]-1]  # data
y = data[:, -1] # label

n_classes = int(np.amax(y)+1)
print("number of classes is ",n_classes)

#print("shape of x is ",x.shape)
#print("shape of y is ",y.shape)

# normalization on input data x
x = (x - x.mean(axis=0)) / x.std(axis=0)

# Use line below with PV_Data Only
#x = np.delete(x, 799999, 1)  # delete second column of C

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

#print(x)
#print(X_train)

#SK LEARN
kmeans = skcl.pipeBuild_KMeans(n_clusters=[n_classes])
affprop = skcl.pipeBuild_AffinityPropagation()

#TS LEARN
kernelKmeans = skcl.pipeBuild_KernelKMeans(n_clusters=[n_classes])
tskmeans = skcl.pipeBuild_TimeSeriesKMeans(n_clusters=[n_classes])

# Run All
#names = ['K Means','Kernel K Means','TS K Means','Affinity Propigation']
#pipes = [kmeans,kernelKmeans,tskmeans,affprop]

# Run One
names=['Affinity Propigation']
pipes=[affprop]

titles = []
for t in names:
    tn = t + ' Train'
    ts = t + ' Test'
    titles.append(tn)
    titles.append(ts)


#x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
#y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5

samples = np.arange(len(X_train[0,:]))
#print("samples: ",samples)

# Plot Training Set
plt.plot(X_train[0,:]) 
plt.plot(X_train[1,:]) 
plt.plot(X_train[2,:]) 
#fig1 = px.scatter(x = samples,y = X_train[0,:],title="Sample Data Entry")
#st.plotly_chart(fig1)
plt.show()

# Plot Testing Set
#fig1.append_trace(go.Scatter(x = X_test[:, 0],y = X_test[:, 1],),row=i,col=1)

#fig2, ax = plt.subplots(1,len(names))

# iterate over classifiers
for j in range(len(names)):
    
    grid_search = GridSearchCV(estimator=pipes[j][0], param_grid=pipes[j][1], scoring='neg_mean_squared_error',cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    score = grid_search.score(X_test, y_test)
    print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
    print(grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_estimator(grid_search, X_test, y_test, xticks_rotation="vertical")
    
    #if j == 0:
    #    fig_0 = skb.plot_cv_results(grid_search.cv_results_, 'decision__criterion', 'decision__max_depth')
    #elif j == 1:
    #    fig_1 = skb.plot_cv_results(grid_search.cv_results_, 'random__criterion', 'random__max_depth')

    #fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    #roc_auc = auc(fpr, tpr)
    #RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name=grid_search)

plt.tight_layout()
#st.pyplot(plt)
plt.show()
#fig2.show()
#"""