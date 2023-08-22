import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from pathlib import Path

import streamlit as st

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
import filter_builder as fltr

st.title('Filter Demo')

p = Path('.')

save_eval_path = p / "AI_engine/evaluation_results/"
save_model_path = p / "AI_engine/saved_models/"
datapath = p / "AI_engine/test_data/"


def filter_plot(filter_name,sig,t,fs,lowcut,highcut,orders=[5],max_rip=5,min_attn=5):
  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.signal import freqz

  # Plot the frequency response for a few different orders.
  plt.figure(1)
  plt.clf()
  for order in orders:
    if filter_name == 'butter_bp':
      b, a = fltr.butter_bandpass(lowcut, highcut, fs, order=order)
    elif filter_name == 'butter_hp':
      b, a = fltr.butter_highpass(lowcut, fs, order=order)
    elif filter_name == 'butter_lp':
      b, a = fltr.butter_lowpass(highcut, fs, order=order)
    elif filter_name == 'cheby1_bp':
      b, a = fltr.cheby1_bandpass(max_rip=max_rip, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    elif filter_name == 'cheby1_hp':
      b, a = fltr.cheby1_highpass(max_rip, lowcut, fs, order=order)
    elif filter_name == 'cheby1_lp':
      b, a = fltr.cheby1_lowpass(max_rip, highcut, fs, order=order)
    elif filter_name == 'cheby2_bp':
      b, a = fltr.cheby2_bandpass(min_attn=min_attn, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    elif filter_name == 'cheby2_hp':
      b, a = fltr.cheby2_highpass(min_attn, lowcut, fs, order=order)
    elif filter_name == 'cheby2_lp':
      b, a = fltr.cheby2_lowpass(min_attn, highcut, fs, order=order)
    elif filter_name == 'butter_bs':
      b, a = fltr.butter_bandstop(lowcut, highcut, fs, order=order)
    elif filter_name == 'cheby1_bs':
      b, a = fltr.cheby1_bandstop(max_rip=max_rip, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    elif filter_name == 'cheby2_bs':
      b, a = fltr.cheby2_bandstop(min_attn=min_attn, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    elif filter_name == 'bessel_bp':
      b, a = fltr.bessel_bandpass(lowcut, highcut, fs, order=order)
    elif filter_name == 'bessel_hp':
      b, a = fltr.bessel_highpass(lowcut, fs, order=order)
    elif filter_name == 'bessel_lp':
      b, a = fltr.bessel_lowpass(highcut, fs, order=order)
    elif filter_name == 'bessel_bs':
      b, a = fltr.bessel_bandstop(lowcut, highcut, fs, order=order)
    else:
      b, a = fltr.butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

  plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],'--', label='sqrt(0.5)')
  plt.title(filter_name + " frequency response")
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Gain')
  plt.grid(True)
  plt.legend(loc='best')

  # Filter a noisy signal.
  #f0 = 20.0
  x = sig
  plt.figure(2)
  plt.clf()
  plt.plot(t, x, label='Noisy signal')

  # Butterworth filters
  if filter_name == 'butter_bp':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Filtered signal (%g Hz)' % f0
    y = fltr.butter_bandpass_filter(x, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'butter_bs':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Blocked signal (%g Hz)' % f0
    y = fltr.butter_bandstop_filter(x, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'butter_hp':
    f0 = int(lowcut)
    label='Signals above (%g Hz)' % f0
    y = fltr.butter_highpass_filter(x, lowcut, fs, order=orders[-1])
  elif filter_name == 'butter_lp':
    f0 = int(highcut)
    label='Signals below (%g Hz)' % f0
    y = fltr.butter_lowpass_filter(x, highcut, fs, order=orders[-1])
  # Cheby1 filters
  elif filter_name == 'cheby1_bp':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Filtered signal (%g Hz)' % f0
    y = fltr.cheby1_bandpass_filter(x, max_rip,lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'cheby1_bs':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Blocked signal (%g Hz)' % f0
    y = fltr.cheby1_bandstop_filter(x, max_rip, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'cheby1_hp':
    f0 = int(lowcut)
    label='Signals above (%g Hz)' % f0
    y = fltr.cheby1_highpass_filter(x, max_rip,lowcut, fs, order=orders[-1])
  elif filter_name == 'cheby1_lp':
    f0 = int(highcut)
    label='Signals below (%g Hz)' % f0
    y = fltr.cheby1_lowpass_filter(x, max_rip, highcut, fs, order=orders[-1])
  # Cheby2 filters
  elif filter_name == 'cheby2_bp':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Filtered signal (%g Hz)' % f0
    y = fltr.cheby2_bandpass_filter(x, min_attn, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'cheby2_bs':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Blocked signal (%g Hz)' % f0
    y = fltr.cheby2_bandstop_filter(x, min_attn, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'cheby2_hp':
    f0 = int(lowcut)
    label='Signals above (%g Hz)' % f0
    y = fltr.cheby2_highpass_filter(x, min_attn, lowcut, fs, order=orders[-1])
  elif filter_name == 'cheby2_lp':
    f0 = int(highcut)
    label='Signals below (%g Hz)' % f0
    y = fltr.cheby2_lowpass_filter(x, min_attn, highcut, fs, order=orders[-1])
  # Bessel filters
  if filter_name == 'bessel_bp':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Filtered signal (%g Hz)' % f0
    y = fltr.bessel_bandpass_filter(x, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'bessel_bs':
    f0 = int((highcut-lowcut)/2+lowcut)
    label='Blocked signal (%g Hz)' % f0
    y = fltr.bessel_bandstop_filter(x, lowcut, highcut, fs, order=orders[-1])
  elif filter_name == 'bessel_hp':
    f0 = int(lowcut)
    label='Signals above (%g Hz)' % f0
    y = fltr.bessel_highpass_filter(x, lowcut, fs, order=orders[-1])
  elif filter_name == 'bessel_lp':
    f0 = int(highcut)
    label='Signals below (%g Hz)' % f0
    y = fltr.bessel_lowpass_filter(x, highcut, fs, order=orders[-1])
  plt.plot(t, y, label=label)
  plt.title(filter_name + " filtered signal")
  plt.xlabel('time (seconds)')
  plt.grid(True)
  plt.axis('tight')
  plt.legend(loc='upper left')

  plt.show()


# Sample rate and desired cutoff frequencies (in Hz).
fs = 60.0
lowcut = 15.0
highcut = 25.0
orders = [3,6,9]

# Filter a noisy signal.
T = 1
nsamples = 100 #int(T * fs)
t = np.linspace(0, T, nsamples, endpoint=False)
sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t) + np.sin(2*np.pi*30*t) # 10 Hz + 20 Hz + 30 Hz

#High Pass (butter)
filter_plot('butter_hp',sig,t,fs,lowcut,highcut,orders)

# High Pass (cheby1)
filter_plot('cheby1_hp',sig,t,fs,lowcut,highcut,orders)

# High Pass (cheby2)
filter_plot('cheby2_hp',sig,t,fs,lowcut,highcut,orders)

#High Pass (bessel)
filter_plot('bessel_hp',sig,t,fs,lowcut,highcut,orders)

#Low Pass (butter)
filter_plot('butter_lp',sig,t,fs,lowcut,highcut,orders)

# Low Pass (cheby1)
filter_plot('cheby1_lp',sig,t,fs,lowcut,highcut,orders)

# Low Pass (cheby2)
filter_plot('cheby2_lp',sig,t,fs,lowcut,highcut,orders)

#Low Pass (bessel)
filter_plot('bessel_lp',sig,t,fs,lowcut,highcut,orders)

#Band Pass (butter)
filter_plot('butter_bp',sig,t,fs,lowcut,highcut,orders)

# Band Pass (cheby1)
filter_plot('cheby1_bp',sig,t,fs,lowcut,highcut,orders)

# Band Pass (cheby2)
filter_plot('cheby2_bp',sig,t,fs,lowcut,highcut,orders)

#Band Pass (bessel)
filter_plot('bessel_bp',sig,t,fs,lowcut,highcut,orders)

#Band Stop (butter)
filter_plot('butter_bs',sig,t,fs,lowcut,highcut,orders)

# Band Stop (cheby1)
filter_plot('cheby1_bs',sig,t,fs,lowcut,highcut,orders)

# Band Stop (cheby2)
filter_plot('cheby2_bs',sig,t,fs,lowcut,highcut,orders)

#Band Stop (bessel)
filter_plot('bessel_bs',sig,t,fs,lowcut,highcut,orders)

"""
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

butter_hp = fltr.butter_highpass_filter()
butter_lp = fltr.butter_lowpass_filter()
butter_bp = fltr.butter_bandpass_filter()
butter_bs = fltr.butter_bandstop_filter()
cheby_1 = fltr.cheby1_filter()
cheby_2 = fltr.cheby2_filter()

decision_tree = skb.pipeBuild_DecisionTreeClassifier(criterion=['gini','entropy'],max_depth=[5, 10])
random_forest = skb.pipeBuild_RandomForestClassifier(criterion=['gini','entropy'],n_estimators=[10], max_depth=[3, 5, 10],max_features=[1])

filters = []
names = ['Decision Tree','Random Forest']
pipes = [decision_tree,random_forest]

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
fig1 = px.scatter(x = samples,y = X_train[0,:],title="Sample Data Entry")
st.plotly_chart(fig1)

# Plot Testing Set
#fig1.append_trace(go.Scatter(x = X_test[:, 0],y = X_test[:, 1],),row=i,col=1)

fig2, ax = plt.subplots(1,len(names))

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
st.pyplot(plt)
#plt.show()
#fig1.show()
#"""