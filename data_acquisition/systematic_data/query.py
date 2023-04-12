'''
Author: Qi7
Date: 2023-04-12 11:23:24
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-04-12 17:32:20
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

# normal:
# start = "2023-04-12T17:45:00Z"
# stop = "2023-04-12T18:45:00Z"

# x1:
# start = "2023-04-12T18:50:00Z"
# stop = "2023-04-12T19:20:00Z"

# x2:
# start = "2023-04-12T19:25:00Z"
# stop = "2023-04-12T19:55:00Z"

# spike:
# start = "2023-04-12T20:00:00Z"
# stop = "2023-04-12T20:30:00Z"

# Query script
start = "2023-04-12T20:00:00Z"
stop = "2023-04-12T20:30:00Z"
measurement = "mem" #net, cpu, processes, diskio, system, mem
tag = "pi1testbed"  #tag name: host
# name = "mmcblk0" # for diskio. add this to query:|> filter(fn: (r) => r["name"] == "{name}")\
# interface = "wlan0" # for network data. add this to query:|> filter(fn: (r) => r["interface"] == "{interface}")\

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)

tags = ["pi1testbed","pi2testbed","pi3testbed","pi4testbed"]

for tag in tags:
    query_api = client.query_api()
    query = f'from(bucket:"{bucket}")\
    |> range(start: {start}, stop: {stop})\
    |> filter(fn:(r) => r._measurement == "{measurement}")\
    |> filter(fn: (r) => r["host"] == "{tag}")\
    |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'
    # pivot() align multiple fields by time


    # results = query_api.query_csv(query) # to csv iterator

    # pd_results = query_api.query_data_frame(query) 
    pd_results = query_api.query_data_frame(query) # to pandas dataframe, the [0] is not relevant (only for network data)
    pd_results = pd_results.drop(columns=['result', 'table', '_start', '_stop', '_measurement']) # drop the unrelated columns

    # pd_results.to_csv(f'{measurement}_{tag}_normal.csv')
    # pd_results.to_csv(f'{measurement}_{tag}_attack_x1_speedup.csv')
    # pd_results.to_csv(f'{measurement}_{tag}_attack_x2_speedup.csv')
    pd_results.to_csv(f'{measurement}_{tag}_attack_motors_spike.csv')