'''
Author: Qi7
Date: 2023-02-28 20:13:02
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-02-28 22:14:22
Description: query the database and make a csv files
'''
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import configparser
import csv

config = configparser.ConfigParser()
config.read('configs/database.ini')

token = config['influxdb']['token']
url = config['influxdb']['url']
org = config['influxdb']['org']

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)

output_file = "dataset/cpu_temp_pi4.csv"
bucket = "rasp-pi"
host = "pi4testbed" #tag name
start = "2023-02-27T19:45:00Z"
stop = "2023-02-28T20:30:00Z"
measurement = "temp"
field = "temp"

query_api = client.query_api()
query = f'from(bucket:"{bucket}")\
        |> range(start: {start}, stop: {stop})\
        |> filter(fn:(r) => r._measurement == "{measurement}")\
        |> filter(fn:(r) => r.host == "{host}")\
        |> filter(fn:(r) => r._field == "{field}")'

result = query_api.query(org=org, query=query)
results = []
timestamps = []

# modify here to make the dataset format
for table in result:
    for record in table.records:
        # results.append((record.get_field(), record.get_value()))
        results.append(record.get_value())
        timestamps.append(record.get_time().timestamp() * 1000000000) # to standard timestamp


# save to csv file
header = ['timestamp', 'cpu_temperature']
with open(output_file, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(len(results)):
        writer.writerow([timestamps[i], results[i]])