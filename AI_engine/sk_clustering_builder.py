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
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, BisectingKMeans, DBSCAN, FeatureAgglomeration, KMeans, MeanShift, MiniBatchKMeans, OPTICS, SpectralClustering
#from sklearn.cluster import HDBSCAN

from tslearn.clustering import KernelKMeans, KShape, TimeSeriesKMeans

# All inputs execpt random_state should be lists of values, even if only one value

# K MEANS
def pipeBuild_KMeans(n_clusters=[8],init=['k-means++'], n_init=[10],max_iter=[300],tol=[1.0e4],verbose=[0],
                     random_state=None,copy_x=[True],algorithm=['lloyd']):
  clusterer = KMeans(random_state=random_state)
  pipeline = Pipeline(steps=[('kmeans', clusterer)])
  params = [{
        'kmeans__n_clusters': n_clusters,
        'kmeans__init': init,
        'kmeans__n_init': n_init,
        'kmeans__max_iter': max_iter,
        'kmeans__tol': tol,
        'kmeans__verbose': verbose,
        'kmeans__copy_x': copy_x,
        'kmeans__algorithm': algorithm,
    }]
  return pipeline, params

# BISECTING K MEANS
def pipeBuild_BisectingKMeans(n_clusters=[8], *, init=['random'], n_init=[1], random_state=None, max_iter=[300], 
                              verbose=[0], tol=[0.0001], copy_x=[True], algorithm=['lloyd'], 
                              bisecting_strategy=['biggest_inertia']):
  clusterer = BisectingKMeans(random_state=random_state)
  pipeline = Pipeline(steps=[('bikmeans', clusterer)])
  params = [{
        'bikmeans__n_clusters': n_clusters,
        'bikmeans__init': init,
        'bikmeans__max_iter': max_iter,
        'bikmeans__copy_x': copy_x,
        'bikmeans__verbose': verbose,
        'bikmeans__algorithm': algorithm,        
        'bikmeans__tol': tol,
        'bikmeans__bisecting_strategy': bisecting_strategy,       
        'bikmeans__n_init': n_init,
    }]
  return pipeline, params

# MINI BATCH K MEANS
def pipeBuild_MiniBatchKMeans(n_clusters=[8], *, init=['k-means++'], max_iter=[100], batch_size=[1024], verbose=[0], 
                     compute_labels=[True], random_state=None, tol=[0.0], max_no_improvement=[10], 
                     init_size=[None], n_init=['warn'], reassignment_ratio=[0.01]):
  clusterer = MiniBatchKMeans(random_state=random_state)
  pipeline = Pipeline(steps=[('mbkmeans', clusterer)])
  params = [{
        'mbkmeans__n_clusters': n_clusters,
        'mbkmeans__init': init,
        'mbkmeans__max_iter': max_iter,
        'mbkmeans__batch_size': batch_size,
        'mbkmeans__verbose': verbose,
        'mbkmeans__compute_labels': compute_labels,        
        'mbkmeans__tol': tol,
        'mbkmeans__max_no_improvement': max_no_improvement,
        'mbkmeans__init_size': init_size,        
        'mbkmeans__n_init': n_init,
        'mbkmeans__reassignment_ratio': reassignment_ratio,
    }]
  return pipeline, params

# KERNEL K MEANS
def pipeBuild_KernelKMeans(n_clusters=[3], kernel=['gak'], max_iter=[50], tol=[1e-06], n_init=[1], 
                     kernel_params=[None], n_jobs=[None], verbose=[0], random_state=None):
  clusterer = KernelKMeans(random_state=random_state)
  pipeline = Pipeline(steps=[('kernelkmeans', clusterer)])
  params = [{
        'kernelkmeans__n_clusters': n_clusters,
        'kernelkmeans__kernel': kernel,
        'kernelkmeans__max_iter': max_iter,
        'kernelkmeans__tol': tol,
        'kernelkmeans__n_init': n_init,
        'kernelkmeans__kernel_params': kernel_params,
        'kernelkmeans__n_jobs': n_jobs,
        'kernelkmeans__verbose': verbose,
    }]
  return pipeline, params

# TIME SERIES K MEANS
def pipeBuild_TimeSeriesKMeans(n_clusters=[3], max_iter=[50], tol=[1e-06], n_init=[1], metric=['euclidean'], 
                               max_iter_barycenter=[100], metric_params=[None], n_jobs=[None], 
                               dtw_inertia=[False], verbose=[0], random_state=None, init=['k-means++']):
  clusterer = TimeSeriesKMeans(random_state=random_state)
  pipeline = Pipeline(steps=[('tskmeans', clusterer)])
  params = [{
        'tskmeans__n_clusters': n_clusters,        
        'tskmeans__max_iter': max_iter,
        'tskmeans__tol': tol,
        'tskmeans__n_init': n_init,
        'tskmeans__metric': metric,
        'tskmeans__max_iter_barycenter': max_iter_barycenter,
        'tskmeans__metric_params': metric_params,
        'tskmeans__n_jobs': n_jobs,
        'tskmeans__dtw_inertia': dtw_inertia,
        'tskmeans__verbose': verbose,
        'tskmeans__init': init,
    }]
  return pipeline, params

# K SHAPE
def pipeBuild_KShape(n_clusters=[3], max_iter=[100], tol=[1e-06], n_init=[1], verbose=[False], 
                     random_state=None, init=['random']):
  clusterer = KShape(random_state=random_state)
  pipeline = Pipeline(steps=[('kshape', clusterer)])
  params = [{
        'kshape__n_clusters': n_clusters,        
        'kshape__max_iter': max_iter,
        'kshape__tol': tol,
        'kshape__n_init': n_init,
        'kshape__verbose': verbose,
        'kshape__init': init,
    }]
  return pipeline, params

# DBSCAN
def pipeBuild_DBSCAN(eps=[0.5], min_samples=[5], metric=['euclidean'], metric_params=[None], 
                     algorithm=['auto'], leaf_size=[30], p=[None], n_jobs=[None]):
  clusterer = DBSCAN()
  pipeline = Pipeline(steps=[('dbscan', clusterer)])
  params = [{
        'dbscan__eps': eps,        
        'dbscan__min_samples': min_samples,
        'dbscan__metric': metric,
        'dbscan__metric_params': metric_params,
        'dbscan__algorithm': algorithm,
        'dbscan__leaf_size': leaf_size,
        'dbscan__p': p,
        'dbscan__n_jobs': n_jobs,
    }]
  return pipeline, params

# AFFINITY PROPAGATION
def pipeBuild_AffinityPropagation(damping=[0.5], max_iter=[200], convergence_iter=[15], copy=[True], 
                                  preference=[None], affinity=['euclidean'], verbose=[False], random_state=None):
  clusterer = AffinityPropagation(random_state=random_state)
  pipeline = Pipeline(steps=[('affprop', clusterer)])
  params = [{
        'affprop__damping': damping,
        'affprop__max_iter': max_iter,
        'affprop__convergence_iter': convergence_iter,
        'affprop__copy': copy,        
        'affprop__preference': preference,
        'affprop__affinity': affinity,
        'affprop__verbose': verbose,
    }]
  return pipeline, params

# MEAN SHIFT
def pipeBuild_MeanShift(bandwidth=[None], seeds=[None], bin_seeding=[False], min_bin_freq=[1], cluster_all=[True], 
                        n_jobs=[None], max_iter=[300]):
  clusterer = MeanShift()
  pipeline = Pipeline(steps=[('meanshift', clusterer)])
  params = [{
        'meanshift__bandwidth': bandwidth,        
        'meanshift__seeds': seeds,
        'meanshift__bin_seeding': bin_seeding,        
        'meanshift__min_bin_freq': min_bin_freq,
        'meanshift__cluster_all': cluster_all,
        'meanshift__n_jobs': n_jobs,
        'meanshift__max_iter': max_iter,
    }]
  return pipeline, params

# SPECTRAL CLUSTERING
def pipeBuild_SpectralClustering(n_clusters=[8], eigen_solver=[None], n_components=[None], random_state=None, 
                                 n_init=[10], gamma=[1.0], affinity=['rbf'], n_neighbors=[10], 
                                 eigen_tol=['auto'], assign_labels=['kmeans'], degree=[3], coef0=[1], 
                                 kernel_params=[None], n_jobs=[None], verbose=[False]):
  clusterer = SpectralClustering(random_state=random_state)
  pipeline = Pipeline(steps=[('specclust', clusterer)])
  params = [{
        'specclust__n_clusters': n_clusters,        
        'specclust__eigen_solver': eigen_solver,
        'specclust__n_components': n_components,        
        'specclust__n_init': n_init,
        'specclust__gamma': gamma,
        'specclust__affinity': affinity,
        'specclust__n_neighbors': n_neighbors,
        'specclust__eigen_tol': eigen_tol,
        'specclust__assign_labels': assign_labels,
        'specclust__degree': degree,
        'specclust__coef0': coef0,
        'specclust__kernel_params': kernel_params,
        'specclust__n_jobs': n_jobs,
        'specclust__verbose': verbose,
    }]
  return pipeline, params

# AGGLOMERATIVE CLUSTERING
def pipeBuild_AgglomerativeClustering(n_clusters=[2], affinity=['deprecated'], metric=[None], memory=[None], 
                                      connectivity=[None], compute_full_tree=['auto'], linkage=['ward'], 
                                      distance_threshold=[None], compute_distances=[False]):
  clusterer = AgglomerativeClustering()
  pipeline = Pipeline(steps=[('aggclust', clusterer)])
  params = [{
        'aggclust__n_clusters': n_clusters,
        'aggclust__affinity': affinity,
        'aggclust__metric': metric,
        'aggclust__memory': memory,
        'aggclust__connectivity': connectivity,
        'aggclust__compute_full_tree': compute_full_tree,
        'aggclust__linkage': linkage,
        'aggclust__distance_threshold': distance_threshold,
        'aggclust__compute_distances': compute_distances,
    }]
  return pipeline, params

# FEATURE AGGLOMERATION
def pipeBuild_FeatureAgglomeration(n_clusters=[2], affinity=['deprecated'], metric=[None], memory=[None], 
                                      connectivity=[None], compute_full_tree=['auto'], linkage=['ward'], 
                                      pooling_func=[np.mean], distance_threshold=[None], 
                                      compute_distances=[False]):
  clusterer = FeatureAgglomeration()
  pipeline = Pipeline(steps=[('featagg', clusterer)])
  params = [{
        'featagg__n_clusters': n_clusters,
        'featagg__affinity': affinity,
        'featagg__metric': metric,
        'featagg__memory': memory,
        'featagg__connectivity': connectivity,
        'featagg__compute_full_tree': compute_full_tree,
        'featagg__linkage': linkage,
        'featagg__pooling_func': pooling_func,
        'featagg__distance_threshold': distance_threshold,
        'featagg__compute_distances': compute_distances,
    }]
  return pipeline, params

# OPTICS
def pipeBuild_OPTICS(min_samples=[5], max_eps=[np.inf], metric=['minkowski'], p=[2], metric_params=[None], 
                     cluster_method=['xi'], eps=[None], xi=[0.05], predecessor_correction=[True], 
                     min_cluster_size=[None], algorithm=['auto'], leaf_size=[30], memory=[None], n_jobs=[None]):
  clusterer = OPTICS()
  pipeline = Pipeline(steps=[('optics', clusterer)])
  params = [{
        'optics__min_samples': min_samples,
        'optics__max_eps': max_eps,
        'optics__metric': metric,
        'optics__p': p,
        'optics__metric_params': metric_params,
        'optics__cluster_method': cluster_method,
        'optics__eps': eps,
        'optics__xi': xi,
        'optics__predecessor_correction': predecessor_correction,
        'optics__min_cluster_size': min_cluster_size,
        'optics__algorithm': algorithm,
        'optics__leaf_size': leaf_size,
        'optics__memory': memory,
        'optics__n_jobs': n_jobs,
    }]
  return pipeline, params

"""
# HDBSCAN
def pipeBuild_HDBSCAN(min_cluster_size=[5], min_samples=[None], cluster_selection_epsilon=[0.0], 
                      max_cluster_size=[None], metric=['euclidean'], metric_params=[None], alpha=[1.0], 
                      algorithm=['auto'], leaf_size=[40], n_jobs=[None], cluster_selection_method=['eom'], 
                      allow_single_cluster=[False], store_centers=[None], copy=[False]):
  clusterer = HDBSCAN()
  pipeline = Pipeline(steps=[('hdbscan', clusterer)])
  params = [{
        'hdbscan__min_cluster_size': min_cluster_size,        
        'hdbscan__min_samples': min_samples,
        'hdbscan__cluster_selection_epsilon': cluster_selection_epsilon,
        'hdbscan__max_cluster_size': max_cluster_size,
        'hdbscan__metric': metric,
        'hdbscan__metric_params': metric_params,
        'hdbscan__alpha': alpha,
        'hdbscan__algorithm': algorithm,
        'hdbscan__leaf_size': leaf_size,        
        'hdbscan__n_jobs': n_jobs,
        'hdbscan__cluster_selection_method': cluster_selection_method,
        'hdbscan__allow_single_cluster': allow_single_cluster,
        'hdbscan__store_centers': store_centers,
        'hdbscan__copy': copy,
    }]
  return pipeline, params
#"""