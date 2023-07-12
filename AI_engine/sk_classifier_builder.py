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
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
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

# All inputs execpt random_state should be lists of values, even if only one value
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

# All inputs execpt random_state should be lists of values, even if only one value
def pipeBuild_RandomForestClassifier(n_estimators=[100],criterion=['gini'],max_depth=[None],max_features=['sqrt'],random_state=None):
  classifier = RandomForestClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('random', classifier)])
  #pipeline = make_pipeline(classifier)
  
  params = [{
      'random__n_estimators': n_estimators,
      'random__criterion': criterion,
      'random__max_depth': max_depth,
      'random__max_features': max_features,
  }]
  return pipeline, params