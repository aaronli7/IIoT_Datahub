'''
Author: Qi7
Date: 2023-04-04 15:58:55
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-04 16:54:52
Description: 
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
bucket = "testbed"
url = "sensorwebdata.engr.uga.edu:8086"

# Query script
start = "2023-04-03T18:05:00Z"
stop = "2023-04-03T19:05:00Z"
measurement = "NI_Waveform"
tag = "lab711"

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)

query_api = client.query_api()
query = f'from(bucket:"{bucket}")\
|> range(start: {start}, stop: {stop})\
|> filter(fn:(r) => r._measurement == "{measurement}")\
|> filter(fn: (r) => r["location"] == "{tag}")\
|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'
# pivot() align multiple fields by time


# results = query_api.query_csv(query) # to csv iterator

pd_results = query_api.query_data_frame(query) # to pandas dataframe
pd_results = pd_results.drop(columns=['result', 'table', '_start', '_stop', '_measurement']) # drop the unrelated columns

pd_results.to_csv('physical_normal.csv')