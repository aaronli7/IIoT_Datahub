'''
Author: Qi7
Date: 2023-04-04 19:53:35
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-04 21:54:12
Description: 
'''
#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

df = pd.read_csv('cyber_final.csv')


#%%
df.head()
# %%

class_encoder = LabelEncoder()
df['class_1'] = class_encoder.fit_transform(df['class_1'])
df['class_2'] = class_encoder.fit_transform(df['class_2'])
# %%