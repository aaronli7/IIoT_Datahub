'''
Author: Qi7
Date: 2023-02-28 16:53:35
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-02-28 20:10:24
Description: 
'''
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import matplotlib.pyplot as plt
import numpy as np
import math
import sys, os, configparser
sys.path.append('../')
# from AI_engine.uils import sliding_windowss

# config = configparser.ConfigParser()
# config.read('../../database.ini')

# token = config['influxdb']['token']
# host = config['influxdb']['host']
# org = config['influxdb']['org']
# bucket = config['influxdb']['bucket']

print(os.getcwd())