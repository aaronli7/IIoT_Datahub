'''
Author: Qi7
Date: 2023-02-08 09:40:57
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-02-08 09:54:27
Description: 
'''
# An quick example of showing how to do the online analysis with Pi's systematic data

import numpy as np
import math
import matplotlib.pyplot as plt
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import sys
sys.path.append('../')
from AI_engine.uils import sliding_windows # import helper function from AI engine module

token = "S-Ohm_ZbcIZfp5_2BgXnSTInDGtjfwJi3uXhij1DR3U1b_83GWkv5CBrHqKjBD5LtRmKYmJ-s5fDGqH7AZ_gLw=="
bucket = "testbed"
org = "lab711"
url="http://sensorwebdata.engr.uga.edu:8086"

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)

# Query script
start = "2023-02-07T01:00:00Z"
stop = "2023-02-08T07:30:00Z"
measurement = "cpu-total"
field = "usage_idle"
status = "normal" # to prompt the data's label
if status == "normal":
    label = 0
else:
    label = 1

query_api = client.query_api()
query = f'from(bucket:"{bucket}")\
|> range(start: {start}, stop: {stop})\
|> filter(fn:(r) => r._measurement == "{measurement}")\
|> filter(fn:(r) => r._field == "{field}")'


result = query_api.query(org=org, query=query)
results = []
for table in result:
    for record in table.records:
        # results.append((record.get_field(), record.get_value()))
        results.append(record.get_value())


x = sliding_windows(results, 20, 2)

# Adding the label information to the matrix
Y = [label] * x.shape[0]
Y = np.array(Y)
Y = Y.reshape((-1,1))
npy_data = np.concatenate((x, Y), axis = 1)

print(npy_data.shape)

# print(npy_data)
# with open('abnormal.npy', 'wb') as f:
#     np.save(f, npy_data)