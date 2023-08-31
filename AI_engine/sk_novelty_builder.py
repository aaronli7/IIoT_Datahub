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

from anomaly_detection import sst_class as sst

# All inputs execpt random_state should be lists of values, even if only one value

# SST ANOMALY DETECTOR
def pipeBuild_SstDetector(win_length,threshold=[0.5], n_components=[5], order=[None], lag=[None],
                 is_scaled=[False], use_lanczos=[True], rank_lanczos=[None], eps=[1e-3]):
  classifier = sst.SstAnomalyDetector(win_length=win_length,threshold=threshold)
  pipeline = Pipeline(steps=[('sst', classifier)])
  #pipeline = make_pipeline(classifier)
  params = [{
        'sst__threshold': threshold,
        'sst__n_components': n_components,
        'sst__order': order,
        'sst__lag': lag,
        'sst__is_scaled': is_scaled,
        'sst__use_lanczos': use_lanczos,
        'sst__rank_lanczos': rank_lanczos,
        'sst__eps': eps,
    }]
  return pipeline, params