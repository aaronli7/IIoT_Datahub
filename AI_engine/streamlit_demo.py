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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, RocCurveDisplay, auc, roc_curve, roc_auc_score

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import joblib

import load_data
import sk_classifier_builder as skb
import sk_classifier_metrics as skm

#st.title('Streamlit Demo')

p = Path('.')

save_eval_path = p / "AI_engine/evaluation_results/"
save_model_path = p / "AI_engine/saved_models/"
datapath = p / "AI_engine/test_data/"

data = np.load("C:/Users/steph/OneDrive/Documents/GitHub/IIoT_Datahub/AI_engine/test_data/synthetic_dataset.npy")
#data = np.load(datapath / "synthetic_dataset.npy")

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

#print(x)
#print(X_train)

decision_tree = skb.pipeBuild_DecisionTreeClassifier(criterion=['gini','entropy'],max_depth=[5, 10])
random_forest = skb.pipeBuild_RandomForestClassifier(criterion=['gini','entropy'],n_estimators=[10], max_depth=[3, 5, 10],max_features=[1])
knn = skb.pipeBuild_KNeighborsClassifier(n_neighbors=[3,5],weights=['uniform'],algorithm=['auto'],leaf_size=[20,30])
gauss = skb.pipeBuild_GaussianProcessClassifier(max_iter_predict=[100],multi_class=['one_vs_rest'])
ada = skb.pipeBuild_AdaBoostClassifier(estimator=[DecisionTreeClassifier()],n_estimators=[50],learning_rate=[1.0])
gnb = skb.pipeBuild_GaussianNB(priors=[None],var_smoothing=[1.0e-9])
qda = skb.pipeBuild_QuadraticDiscriminantAnalysis(priors=[None],reg_param=[0.0],store_covariance=[False],tol=[1.0e-4])
svc = skb.pipeBuild_SVC(C=[1.0],kernel=['rbf'],degree=[3],gamma=['scale'],tol=[1.0e-3],random_state=None)
mlp = skb.pipeBuild_MLPClassifier(hidden_layer_sizes=[(100,)],activation=['relu'],solver=['adam'],alpha=[0.0001],batch_size=['auto'],learning_rate=['constant'],random_state=None)
nusvc = skb.pipeBuild_NuSVC(nu=[0.5],kernel=['rbf'],degree=[3],gamma=['scale'],tol=[1.0e-3],random_state=None)

names = ['Decision Tree','Random Forest','KNN','Gaussian','AdaBoost','GaussianNB','QDA','SVC','MLP','NuSVC']
pipes = [decision_tree,random_forest,knn,gauss,ada,gnb,qda,svc,mlp,nusvc]

#names=['NuSVC']
#pipes=[nusvc]

titles = []
for t in names:
    tn = t + ' Train'
    ts = t + ' Test'
    titles.append(tn)
    titles.append(ts)

#"""
#x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
#y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5

samples = np.arange(len(X_train[0,:]))
#print("samples: ",samples)

# Plot Training Set
#fig1 = px.scatter(x = samples,y = X_train[0,:],title="Sample Data Entry")
#st.plotly_chart(fig1)
#fig1.show()

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