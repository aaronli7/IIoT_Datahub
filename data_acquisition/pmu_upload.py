import socket
import json
import datetime, time
from typing import Sized
import pytz
import json
import warnings
import pandas as pd
from waveform_to_PMU import feature_extract
import numpy as np
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS

warnings.filterwarnings("ignore")

# helper function
def df_int_to_float(df):
    for i in range(len(df)):
        for j in range(len(df[i])):
            df[i][j] = float(df[i][j])
    return df

time_format = "%Y-%m-%d %H:%M:%S"
tz_NY = pytz.timezone("America/New_York")

#influxdb config
token = "0ML4vBa-81dGKI3_wD-ReiSRdLggdJPXKoTKLPITBcOZXl8MJh7W8wFSkNUNM_uPS9mJpzvBxUKfKgie0dHiow=="
org = "lab711"
bucket = "testbed"
url = "sensorwebdata.engr.uga.edu:8086"
measurement = "NI_Waveform"

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)

write_api = client.write_api(write_options=SYNCHRONOUS)

# socket config for receiving the data from NI device
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("localhost", 8089))
    s.listen(1)
    conn, addr = s.accept()
    cnt = 0
    with conn:
        print(f"Socket connected by {addr}")
        while True:
            data = conn.recv(6*2000*4*8)  # 6 channels * 2khz * 4bytes * 8bit
            if not data:
                break

            data_array = json.loads(data.decode())
            value_list = data_array
            value_list = df_int_to_float(value_list)
            value_list = np.array(value_list)

            features = feature_extract(value_list, f_s=2500)

            currentTime = datetime.datetime.now(tz_NY)
            timestamp = int(currentTime.timestamp() * 1000000000)
            p = influxdb_client.Point(measurement) \
                .tag("location", "lab711") \
                .field("sensor1_DC_mag", features[0][0]) \
                .field("sensor1_DC_freq", features[1][0]) \
                .field("sensor1_DC_angle", features[2][0]) \
                .field("sensor1_DC_thd", features[3][0]) \
                .field("sensor1_AC_mag", features[4][0]) \
                .field("sensor1_AC_freq", features[5][0]) \
                .field("sensor1_AC_angle", features[6][0]) \
                .field("sensor1_AC_thd", features[7][0]) \
                .field("sensor2_DC_mag", features[8][0]) \
                .field("sensor2_DC_freq", features[9][0]) \
                .field("sensor2_DC_angle", features[10][0]) \
                .field("sensor2_DC_thd", features[11][0]) \
                .field("sensor2_AC_mag", features[12][0]) \
                .field("sensor2_AC_freq", features[13][0]) \
                .field("sensor2_AC_angle", features[14][0]) \
                .field("sensor2_AC_thd", features[15][0]) \
                .field("sensor3_DC_mag", features[16][0]) \
                .field("sensor3_DC_freq", features[17][0]) \
                .field("sensor3_DC_angle", features[18][0]) \
                .field("sensor3_DC_thd", features[19][0]) \
                .field("sensor3_AC_mag", features[20][0]) \
                .field("sensor3_AC_freq", features[21][0]) \
                .field("sensor3_AC_angle", features[22][0]) \
                .field("sensor3_AC_thd", features[23][0]) \
                .time(timestamp)
            write_api.write(bucket=bucket, org=org, record=p)

print("Connection closed.")
