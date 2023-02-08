'''
Author: Qi7
Date: 2023-01-24 20:39:08
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-02-07 20:08:49
Description: uploading motor data to jay's influxdbcloud
'''
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime, time
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

dataset = np.load('motor_data.npy')
timestamp = datetime.datetime.now().timestamp()

bucket = "uga test"
org = "james.hill@agingaircraft.us"
token = "F18iTk0we6pA4iixSlWR4TMFJv0so4R1HWSWCE6cHBzBgOvUvquA5jEI06n8hMkkZ5pPz_sFbLSUidB3kwFyBw=="
url="https://us-east-1-1.aws.cloud2.influxdata.com"

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org,
    debug=False
)
# write script
write_api = client.write_api(write_options=SYNCHRONOUS)
timestamp = datetime.datetime.now().timestamp()
step = 0.00005 # 50000 nanosecond

for i in range(dataset.shape[0]):
# for i in range(10):
    timestamp += 0.1
    p = influxdb_client.Point("motor_data_5M").tag("label", dataset[i][26]).field("feature_0", dataset[i][0]).field("feature_1", dataset[i][1]).field("feature_2", dataset[i][2]).field("feature_3", dataset[i][3]).field("feature_4", dataset[i][4]).field("feature_5", dataset[i][5]).field("feature_6", dataset[i][6]).field("feature_7", dataset[i][7]).field("feature_8", dataset[i][8]).field("feature_9", dataset[i][9]).field("feature_10", dataset[i][10]).field("feature_11", dataset[i][11]).field("feature_12", dataset[i][12]).field("feature_13", dataset[i][13]).field("feature_14", dataset[i][14]).field("feature_15", dataset[i][15]).field("feature_16", dataset[i][16]).field("feature_17", dataset[i][17]).field("feature_18", dataset[i][18]).field("feature_19", dataset[i][19]).field("feature_20", dataset[i][20]).field("feature_21", dataset[i][21]).field("feature_22", dataset[i][22]).field("feature_23", dataset[i][23]).field("feature_24", dataset[i][24]).field("feature_25", dataset[i][25])
    write_api.write(bucket=bucket, org=org, record=p)
    time.sleep(step)