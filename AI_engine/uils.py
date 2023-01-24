'''
Author: Qi7
Date: 2023-01-18 09:48:56
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-01-24 09:45:27
Description: slicing window for the array.
'''
import torch
import numpy as np

def extract_windows(array, clearing_time_index, max_time, sub_window_size):
    """
    Rolling sub-window. For-loop is been used, which is bad in terms of the performance.
    """
    examples = []
    start = clearing_time_index + 1 - sub_window_size + 1
    
    for i in range(max_time+1):
        example = array[start+i:start+sub_window_size+i]
        examples.append(np.expand_dims(example, 0))
    
    return np.vstack(examples)


def extract_windows_vectorized(array, clearing_time_index, max_time, sub_window_size):
    start = clearing_time_index + 1 - sub_window_size + 1
    
    sub_windows = (
        start +
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time + 1), 0).T
    )
    
    return array[sub_windows]

def vectorized_stride_v1(array, clearing_time_index, max_time, sub_window_size,
                         stride_size):
    start = clearing_time_index + 1 - sub_window_size + 1
    
    sub_windows = (
        start + 
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time + 1), 0).T
    )
    
    # Fancy indexing to select every V rows.
    return array[sub_windows[::stride_size]]


def vectorized_stride_v2(array, clearing_time_index, max_time, sub_window_size,
                         stride_size):
    start = clearing_time_index + 1 - sub_window_size + 1
    
    sub_windows = (
        start + 
        np.expand_dims(np.arange(sub_window_size), 0) +
        # Create a rightmost vector as [0, V, 2V, ...].
        np.expand_dims(np.arange(max_time + 1, step=stride_size), 0).T
    )
    
    return array[sub_windows]