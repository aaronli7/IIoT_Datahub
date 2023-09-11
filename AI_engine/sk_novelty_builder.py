import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import re
import pytz
from datetime import datetime

import enum
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

from anomaly_detection import sst_class as sst

# All inputs execpt random_state should be lists of values, even if only one value

## NOVELTY DETECTION

# ONCE CLASS SVM
def pipeBuild_OneClassSVM(kernel=['rbf'],degree=[3], gamma=['scale'], coef0=[0.0], tol=[1.0e-3],
                 nu=[0.5], shrinking=[True], cache_size=[200], verbose=[False],max_iter=[-1]):
  classifier = OneClassSVM()
  pipeline = Pipeline(steps=[('1svm', classifier)])
  params = [{
        '1svm__kernel': kernel,
        '1svm__degree': degree,
        '1svm__gamma': gamma,
        '1svm__coef0': coef0,
        '1svm__tol': tol,
        '1svm__nu': nu,
        '1svm__shrinking': shrinking,
        '1svm__cache_size': cache_size,
        '1svm__verbose': verbose,
        '1svm__max_iter': max_iter,
    }]
  return pipeline, params

# SGC ONCE CLASS SVM
def pipeBuild_SGDOneClassSVM(nu=[0.5],fit_intercept=[True], max_iter=[1000], tol=[1.0e-3],
                 shuffle=[True],verbose=[False],random_state=None,learning_rate=['optimal'],
                 eta0=[0.0],power_t=[0.5],warm_start=[False],average=[False]):
  classifier = SGDOneClassSVM(random_state=random_state)
  pipeline = Pipeline(steps=[('sgd1svm', classifier)])
  params = [{
        'sgd1svm__nu': nu,
        'sgd1svm__fit_intercept': fit_intercept,
        'sgd1svm__max_iter': max_iter,        
        'sgd1svm__tol': tol,        
        'sgd1svm__shuffle': shuffle,
        'sgd1svm__verbose': verbose,
        'sgd1svm__learning_rate': learning_rate,
        'sgd1svm__eta0': eta0,
        'sgd1svm__power_t': power_t,
        'sgd1svm__warm_start': warm_start,
        'sgd1svm__average': average,
    }]
  return pipeline, params

# SST ANOMALY DETECTOR
def pipeBuild_SstDetector(win_length,threshold=[0.5], order=[None], n_components=[5],lag=[None],
                 is_scaled=[False], use_lanczos=[True], rank_lanczos=[None], eps=[1e-3]):
  detector = sst.SstDetector(win_length=win_length,threshold=threshold,order=order,n_components=n_components,
                               lag=lag,is_scaled=is_scaled,use_lanczos=use_lanczos,rank_lanczos=rank_lanczos,eps=eps)
  pipeline = Pipeline(steps=[('sst', detector)])
  params = [{
        'sst__threshold': threshold,
        'sst__order': order,
        'sst__n_components': n_components,
        'sst__lag': lag,
        'sst__is_scaled': is_scaled,
        'sst__use_lanczos': use_lanczos,
        'sst__rank_lanczos': rank_lanczos,
        'sst__eps': eps,
    }]
  return pipeline, params


# NOVELTY OR OUTLIER DETECTION

# Local Outlier Factor
    # Set novelty to True for novelty detection, otherwise False is outlier detection
def pipeBuild_LocalOutlierFactor(n_neighbors=[20],algorithm=['auto'], leaf_size=[30], metric=['minkowski'],
                 p=[2],verbose=[False],metric_params=[None],contamination=['auto'],
                 novelty=[False],n_jobs=[None]):
  detector = LocalOutlierFactor()
  pipeline = Pipeline(steps=[('lof', detector)])
  params = [{
        'lof__n_neighbors': n_neighbors,
        'lof__algorithm': algorithm,
        'lof__leaf_size': leaf_size,        
        'lof__metric': metric,        
        'lof__p': p,
        'lof__metric_params': metric_params,
        'lof__contamination': contamination,
        'lof__novelty': novelty,
        'lof__n_jobs': n_jobs,
    }]
  return pipeline, params

## OUTLIER DETECTION

# Elliptic Envelope
def pipeBuild_EllipticEnvelope(store_precision=[True],assume_centered=[False], support_fraction=[None], 
                               contamination=[0.1], random_state=None):
  detector = EllipticEnvelope(random_state=random_state)
  pipeline = Pipeline(steps=[('ellenv', detector)])
  params = [{
        'ellenv__store_precision': store_precision,
        'ellenv__assume_centered': assume_centered,
        'ellenv__support_fraction': support_fraction,        
        'ellenv__contamination': contamination,        
    }]
  return pipeline, params

# Isolation Forest
def pipeBuild_IsolationForest(n_estimators=[100],max_samples=['auto'], contamination=['auto'], 
                               max_features=[1.0], bootstrap=[False], n_jobs=[None], random_state=None,
                               verbose=[0],warm_start=[False]):
  detector = IsolationForest(random_state=random_state)
  pipeline = Pipeline(steps=[('isofrst', detector)])
  params = [{
        'isofrst__n_estimators': n_estimators,
        'isofrst__max_samples': max_samples,
        'isofrst__contamination': contamination, 
        'isofrst__max_features': max_features,
        'isofrst__bootstrap': bootstrap,
        'isofrst__n_jobs': n_jobs, 
        'isofrst__verbose': verbose, 
        'isofrst__warm_start': warm_start,
       
    }]
  return pipeline, params