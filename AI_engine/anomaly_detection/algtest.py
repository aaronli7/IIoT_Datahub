'''
Author: Qi7
Date: 2023-03-02 11:10:23
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-03-02 11:42:44
Description: 
'''
from util import *
from datetime import datetime
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
import time
import sys, os
import warnings
import algorithm as alg
import algorithm_detect as detect

warnings.filterwarnings("ignore")

#influxdb config
token = "0ML4vBa-81dGKI3_wD-ReiSRdLggdJPXKoTKLPITBcOZXl8MJh7W8wFSkNUNM_uPS9mJpzvBxUKfKgie0dHiow=="
org = "lab711"
bucket = "testbed"
url = "sensorwebdata.engr.uga.edu:8086"
measurement = "detection_results"

debug = True
verbose = True

src = {'ip': url, 'org':org,'token':token,'bucket':bucket}
dest = {'ip': url, 'org':org,'token':token,'bucket':bucket}

