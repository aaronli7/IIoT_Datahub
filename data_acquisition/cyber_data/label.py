'''
Author: Qi7
Date: 2023-04-04 16:58:46
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-12 17:04:44
Description: 
'''
import pandas as pd

filename_normal = 'data_acquisition/cyber_data/cyber_normal.csv'
filename_attack_1 = 'data_acquisition/cyber_data/cyber_attack_x1_speedup.csv'
filename_attack_2 = 'data_acquisition/cyber_data/cyber_attack_x2_speedup.csv'
filename_attack_3 = 'data_acquisition/cyber_data/cyber_attack_motors_spike.csv'

df_normal = pd.read_csv(filename_normal)
df_attack_1 = pd.read_csv(filename_attack_1)
df_attack_2 = pd.read_csv(filename_attack_2)
df_attack_3 = pd.read_csv(filename_attack_3)


num_samples = df_normal.shape[0]
labels = ['normal'] * num_samples
df_normal['class_1'] = labels
df_normal['class_2'] = labels

num_samples = df_attack_1.shape[0]
labels = ['attack'] * num_samples
df_attack_1['class_1'] = labels
labels = ['x1_speedup'] * num_samples
df_attack_1['class_2'] = labels

num_samples = df_attack_2.shape[0]
labels = ['attack'] * num_samples
df_attack_2['class_1'] = labels
labels = ['x2_speedup'] * num_samples
df_attack_2['class_2'] = labels

num_samples = df_attack_3.shape[0]
labels = ['attack'] * num_samples
df_attack_3['class_1'] = labels
labels = ['motors_spike'] * num_samples
df_attack_3['class_2'] = labels



df = pd.concat([df_normal, df_attack_1, df_attack_2, df_attack_3], axis=0, ignore_index=True, sort=False)
df = df.drop("Unnamed: 0", axis=1)

df.to_csv("cyber_final.csv")