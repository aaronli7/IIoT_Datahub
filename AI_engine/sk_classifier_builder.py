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
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import BernoulliRBM
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV

# All inputs execpt random_state should be lists of values, even if only one value

# DECISION TREE CLASSIFIER
def pipeBuild_DecisionTreeClassifier(criterion=['gini'],splitter=['best'], max_depth=[None],random_state=None,):
  classifier = DecisionTreeClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('decision', classifier)])
  #pipeline = make_pipeline(classifier)
  params = [{
        'decision__criterion': criterion,
        'decision__splitter': splitter,
        'decision__max_depth': max_depth,
    }]
  return pipeline, params

# RANDOME FOREST CLASSIFIER
def pipeBuild_RandomForestClassifier(n_estimators=[100],criterion=['gini'],max_depth=[None],max_features=['sqrt'],random_state=None):
  classifier = RandomForestClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('random', classifier)])
  
  params = [{
      'random__n_estimators': n_estimators,
      'random__criterion': criterion,
      'random__max_depth': max_depth,
      'random__max_features': max_features,
  }]
  return pipeline, params

# K NEAREST NEIGHBORS CLASSIFIER
def pipeBuild_KNeighborsClassifier(n_neighbors=[100],weights=['uniform'],algorithm=['auto'],leaf_size=[30]):
  classifier = KNeighborsClassifier()
  pipeline = Pipeline(steps=[('knn', classifier)])
  
  params = [{
      'knn__n_neighbors': n_neighbors,
      'knn__weights': weights,
      'knn__algorithm': algorithm,
      'knn__leaf_size': leaf_size,
  }]
  return pipeline, params

# GAUSSIAN PROCESS CLASSIFIER
def pipeBuild_GaussianProcessClassifier(max_iter_predict=[100],multi_class=['one_vs_rest'],random_state=None):
  classifier = GaussianProcessClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('gaussian', classifier)])
  
  params = [{
      'gaussian__max_iter_predict': max_iter_predict,
      'gaussian__multi_class': multi_class,
  }]
  return pipeline, params

# ADA BOOST CLASSIFIER
def pipeBuild_AdaBoostClassifier(estimator=[DecisionTreeClassifier()],n_estimators=[50],learning_rate=[1.0],random_state=None):
  classifier = AdaBoostClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('ada', classifier)])
  
  params = [{
      'ada__estimator': estimator,
      'ada__n_estimators': n_estimators,
      'ada__learning_rate': learning_rate,
  }]
  return pipeline, params

# GAUSSIAN NAIVE BAYES CLASSIFIER
def pipeBuild_GaussianNB(priors=[None],var_smoothing=[1.0e-9]):
  classifier = GaussianNB()
  pipeline = Pipeline(steps=[('gnb', classifier)])
  
  params = [{
      'gnb__priors': priors, # Array of Arrays if not default
      'gnb__var_smoothing': var_smoothing,
  }]
  return pipeline, params

# QUADRATIC DISCRIMINANT ANALYSIS
def pipeBuild_QuadraticDiscriminantAnalysis(priors=[None],reg_param=[0.0],store_covariance=[False],tol=[1.0e-4]):
  classifier = QuadraticDiscriminantAnalysis()
  pipeline = Pipeline(steps=[('qda', classifier)])
  
  params = [{
      'qda__priors': priors, # Array of Arrays if not default
      'qda__reg_param': reg_param,
      'qda__store_covariance': store_covariance,
      'qda__tol': tol,
  }]
  return pipeline, params

# SUPPORT VECTOR CLASSIFIER
def pipeBuild_SVC(C=[1.0],kernel=['rbf'],degree=[3],gamma=['scale'],tol=[1.0e-3],random_state=None):
  classifier = SVC(random_state=random_state)
  pipeline = Pipeline(steps=[('svc', classifier)])
  
  params = [{
      'svc__C': C,
      'svc__kernel': kernel,
      'svc__degree': degree,
      'svc__gamma': gamma,
      'svc__tol': tol,
  }]
  return pipeline, params

# MULTI-LAYER PERCEPTRON CLASSIFIER
def pipeBuild_MLPClassifier(hidden_layer_sizes=[(100,)],activation=['relu'],solver=['adam'],alpha=[0.0001],batch_size=['auto'],learning_rate=['constant'],random_state=None):
  classifier = MLPClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('mlp', classifier)])
  
  params = [{
      'mlp__hidden_layer_sizes': hidden_layer_sizes,
      'mlp__activation': activation,
      'mlp__solver': solver,
      'mlp__alpha': alpha,
      'mlp__batch_size': batch_size,
      'mlp__learning_rate': learning_rate,
  }]
  return pipeline, params

# NU-SUPPORT VECTOR CLASSIFIER
def pipeBuild_NuSVC(nu=[0.5],kernel=['rbf'],degree=[3],gamma=['scale'],tol=[1.0e-3],random_state=None):
  classifier = NuSVC(random_state=random_state)
  pipeline = Pipeline(steps=[('nusvc', classifier)])
  
  params = [{
      'nusvc__nu': nu,
      'nusvc__kernel': kernel,
      'nusvc__degree': degree,
      'nusvc__gamma': gamma,
      'nusvc__tol': tol,
  }]
  return pipeline, params