'''
Author: Qi7
Date: 2023-04-12 11:23:24
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-12 11:59:49
Description: query script for collecting the systematic data from Raspberry Pi.
'''
import numpy as np
import pytz
import matplotlib.pyplot as plt
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS


time_format = "%Y-%m-%d %H:%M:%S"
tz_NY = pytz.timezone("America/New_York")

#influxdb config
token = "0ML4vBa-81dGKI3_wD-ReiSRdLggdJPXKoTKLPITBcOZXl8MJh7W8wFSkNUNM_uPS9mJpzvBxUKfKgie0dHiow=="
org = "lab711"
bucket = "rasp-pi"
url = "sensorwebdata.engr.uga.edu:8086"

# Query script
start = "2023-04-06T18:05:00Z"
stop = "2023-04-06T18:10:00Z"
measurement = "temp" #net, cpu
tag = "pi1testbed"  #tag name: host

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)

query_api = client.query_api()
query = f'from(bucket:"{bucket}")\
|> range(start: {start}, stop: {stop})\
|> filter(fn:(r) => r._measurement == "{measurement}")\
|> filter(fn: (r) => r["host"] == "{tag}")\
|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'
# pivot() align multiple fields by time


# results = query_api.query_csv(query) # to csv iterator

pd_results = query_api.query_data_frame(query) 
# pd_results = query_api.query_data_frame(query)[1] # to pandas dataframe, the [0] is not relevant (only for network data)
pd_results = pd_results.drop(columns=['result', 'table', '_start', '_stop', '_measurement']) # drop the unrelated columns

pd_results.to_csv('temp_pi_1.csv')