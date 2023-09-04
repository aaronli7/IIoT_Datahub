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
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier, BallTree, KDTree
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.neural_network import BernoulliRBM

# All inputs execpt random_state should be lists of values, even if only one value

# DECISION TREE CLASSIFIER
def pipeBuild_DecisionTreeClassifier(criterion=['gini'],splitter=['best'], max_depth=[None],random_state=None):
  classifier = DecisionTreeClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('decision', classifier)])
  #pipeline = make_pipeline(classifier)
  params = [{
        'decision__criterion': criterion,
        'decision__splitter': splitter,
        'decision__max_depth': max_depth,
    }]
  return pipeline, params

# RANDOM FOREST CLASSIFIER
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

# NEAREST CENTROID CLASSIFIER
def pipeBuild_NearestCentroid(metric=['euclidean'],shrink_threshold=[None]):
  classifier = NearestCentroid()
  pipeline = Pipeline(steps=[('nc', classifier)])
  
  params = [{
      'nc__metric': metric,
      'nc__shrink_threshold': shrink_threshold,
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

# LINEAR DISCRIMINANT ANALYSIS
def pipeBuild_LinearDiscriminantAnalysis(solver=['svd'],shrinkage=[None],priors=[None],n_components=[None],store_covariance=[False],tol=[1.0e-4],covariance_estimator=[None]):
  classifier = LinearDiscriminantAnalysis()
  pipeline = Pipeline(steps=[('lda', classifier)])
  
  params = [{
      'lda__solver': solver,
      'lda__shrinkage': shrinkage,
      'lda__priors': priors, # Array of Arrays if not default
      'lda__n_components': n_components,
      'lda__store_covariance': store_covariance,
      'lda__tol': tol,
      'lda__covariance_estimator': covariance_estimator,
  }]
  return pipeline, params

# STOCHASTIC GRADIENT DESCENT
def pipeBuild_SGDClassifier(loss=['hinge'],penalty=['l2'],alpha=[0.0001],l1_ratio=[0.15],fit_intercept=[True],max_iter=[1000],tol=[1.0e-3],epsilon=[0.1],random_state=None,learning_rate=['optimal']):
  classifier = SGDClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('sgd', classifier)])
  
  params = [{
      'sgd__loss': loss,
      'sgd__penalty': penalty,
      'sgd__alpha': alpha,
      'sgd__l1_ratio': l1_ratio,
      'sgd__fit_intercept': fit_intercept,
      'sgd__max_iter': max_iter,
      'sgd__tol': tol,
      'sgd__epsilon': epsilon,
      'sgd__learning_rate': learning_rate,
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

# PASSIVE AGGRESSIVE CLASSIFIER
def pipeBuild_PassiveAggressiveClassifier(C=[1.0],fit_intercept=[True],max_iter=[1000],tol=[1.0e-3],early_stopping=[False],n_iter_no_change=[5],loss=['hinge'],random_state=None):
  classifier = PassiveAggressiveClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('pac', classifier)])
  
  params = [{
      'pac__C': C,
      'pac__fit_intercept': fit_intercept,
      'pac__max_iter': max_iter,
      'pac__tol': tol,
      'pac__early_stopping': early_stopping,
      'pac__n_iter_no_change': n_iter_no_change,
      'pac__loss': loss,
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

# BAGGING CLASSIFIER
def pipeBuild_BaggingClassifier(estimator=[DecisionTreeClassifier()],n_estimators=[10],max_samples=[1.0],max_features=[1.0],random_state=None):
  classifier = BaggingClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('bag', classifier)])
  
  params = [{
      'bag__estimator': estimator,
      'bag__n_estimators': n_estimators,
      'bag__max_samples': max_samples,
      'bag__max_features': max_features,
  }]
  return pipeline, params

# EXTRA TREES CLASSIFIER
def pipeBuild_ExtraTreesClassifier(n_estimators=[100],criterion=['gini'],max_depth=[None],min_samples_split=[2],min_samples_leaf=[1],max_features=['sqrt'],random_state=None):
  classifier = ExtraTreesClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('extra', classifier)])
  
  params = [{
      'extra__n_estimators': n_estimators,
      'extra__criterion': criterion,
      'extra__max_depth': max_depth,
      'extra__min_samples_split': min_samples_split,
      'extra__min_samples_leaf': min_samples_leaf,
      'extra__max_features': max_features,
  }]
  return pipeline, params

# RADIUS NEAREST NEIGHBORS
def pipeBuild_RadiusNeighborsClassifier(radius=[1.0],weights=['uniform'],algorithm=['auto'],leaf_size=[30],p=[2],metric=['minkowski']):
  classifier = RadiusNeighborsClassifier()
  pipeline = Pipeline(steps=[('radiusNN', classifier)])
  
  params = [{
      'radiusNN__radius': radius,
      'radiusNN__weights': weights,
      'radiusNN__algorithm': algorithm,
      'radiusNN__leaf_size': leaf_size,
      'radiusNN__p': p,
      'radiusNN__metric': metric,
  }]
  return pipeline, params


# GRADIENT TREE BOOSTING
def pipeBuild_GradientBoostingClassifier(loss=['log_loss'],learning_rate=[0.1],n_estimators=[100],subsample=[1.0],criterion=['friedman_mse'],min_samples_split=[2],min_samples_leaf=[1],max_depth=[3],random_state=None):
  classifier = GradientBoostingClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('gb', classifier)])
  
  params = [{
      'gb__loss': loss,
      'gb__learning_rate': learning_rate,
      'gb__n_estimators': n_estimators,
      'gb__subsample': subsample,
      'gb__criterion': criterion,      
      'gb__min_samples_split': min_samples_split,
      'gb__min_samples_leaf': min_samples_leaf,
      'gb__max_depth': max_depth,    
  }]
  return pipeline, params

# HISTOGRAM GRADIENT BOOSTING
def pipeBuild_HistGradientBoostingClassifier(loss=['log_loss'],learning_rate=[0.1],max_iter=[100],max_leaf_nodes=[31],max_depth=[3],min_samples_leaf=[20],l2_regularization=[0],max_bins=[255],random_state=None):
  classifier = HistGradientBoostingClassifier(random_state=random_state)
  pipeline = Pipeline(steps=[('hgb', classifier)])
  
  params = [{
      'hgb__loss': loss,
      'hgb__learning_rate': learning_rate,
      'hgb__max_iter': max_iter,
      'hgb__max_leaf_nodes': max_leaf_nodes, 
      'hgb__max_depth': max_depth,
      'hgb__min_samples_leaf': min_samples_leaf,
      'hgb__l2_regularization': l2_regularization,
      'hgb__max_bins': max_bins,    
  }]
  return pipeline, params

# BOURNOULLI NAIVE BAYES CLASSIFIER
def pipeBuild_BernoulliNB(alpha=[1.0],force_alpha=[True],binarize=[0.0],fit_prior=[True],class_prior=[None]):
  classifier = BernoulliNB()
  pipeline = Pipeline(steps=[('mnb', classifier)])
  
  params = [{
      'mnb__alpha': alpha,
      'mnb__binarize': binarize,
      'mnb__fit_prior': fit_prior,
      'mnb__fit_prior': fit_prior,
      'mnb__class_prior': class_prior,
  }]
  return pipeline, params


"""
# KD TREE
def pipeBuild_KDTree(X,leaf_size=[40],metric=['minkowski']):
  classifier = KDTree(X)
  pipeline = Pipeline(steps=[('kdt', classifier)])
  
  params = [{
      'kdt__leaf_size': leaf_size,
      'kdt__metric': metric,
  }]
  return pipeline, params

# BALL TREE
def pipeBuild_BallTree(X,leaf_size=[40],metric=['minkowski']):
  classifier = BallTree(X)
  pipeline = Pipeline(steps=[('ball', classifier)])
  
  params = [{
      'ball__leaf_size': leaf_size,
      'ball__metric': metric,
  }]
  return pipeline, params

# COMPLEMENT NAIVE BAYES CLASSIFIER
def pipeBuild_ComplementNB(alpha=[1.0],force_alpha=[True],fit_prior=[True],class_prior=[None],norm=[False]):
  classifier = ComplementNB()
  pipeline = Pipeline(steps=[('cnb', classifier)])
  
  params = [{
      'cnb__alpha': alpha,
      'cnb__force_alpha': force_alpha,
      'cnb__fit_prior': fit_prior,
      'cnb__class_prior': class_prior,
      'cnb__norm': norm,
  }]
  return pipeline, params

# STACKING CLASSIFIER - Not Working, same issue as Voting Classifier
def pipeBuild_StackingClassifier(estimators=[('dt_def', DecisionTreeClassifier()),('knn_def', KNeighborsClassifier()),],final_estimator=[None],cv=[None],stack_method=['auto']):
  classifier = StackingClassifier(estimators=estimators)
  pipeline = Pipeline(steps=[('stack', classifier)])
  
  params = [{
      'stack__estimators': estimators,
      'stack__final_estimator': final_estimator,
      'stack__cv': cv,
      'stack__stack_method': stack_method,
  }]
  return pipeline, params

# VOTING CLASSIFIER - NOT WORKING

def pipeBuild_VotingClassifier(estimators=[[DecisionTreeClassifier()],[KNeighborsClassifier()]],voting=['hard'],weights=[None]):
  classifier = VotingClassifier(estimators=estimators)
  pipeline = Pipeline(steps=[('vote', classifier)])
  
  params = [{
      'vote__estimators': estimators,
      'vote__voting': voting,
      'vote__weights': weights,
  }]
  return pipeline, params
#"""