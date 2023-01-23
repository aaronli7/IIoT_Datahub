'''
Author: Qi7
Date: 2023-01-23 16:15:49
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-01-23 16:56:47
Description: query the data from influxDB
'''
import numpy as np
import math
import matplotlib.pyplot as plt
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

token = "sSDR3urw9jxgsqq4q45MkUHZ4pqloQuKt_8MNTPoz8mEu4Nx4TRKXApZBTR-4QIz0XHcWrykWWm__9eoW9QLQQ=="
bucket = "theBucket"
org = "sevenSun"
url="http://localhost:8086"

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)

# Query script
start = "2023-01-23T23:27:40Z"
stop = "2023-01-23T23:27:41Z"
measurement = "waveform"
field = "current"
status = "normal"

query_api = client.query_api()
query = f'from(bucket:"{bucket}")\
|> range(start: {start}, stop: {stop})\
|> filter(fn:(r) => r._measurement == "{measurement}")\
|> filter(fn:(r) => r.status == "{status}")\
|> filter(fn:(r) => r._field == "{field}")'

result = query_api.query(org=org, query=query)
results = []
plot_values = []
for table in result:
    for record in table.records:
        results.append((record.get_field(), record.get_value()))
        plot_values.append(record.get_value())

plt.plot(plot_values)
plt.show()