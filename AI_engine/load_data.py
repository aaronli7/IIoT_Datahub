import sys
import numpy as np
import pandas as pd
from pandas import read_csv
from pathlib import Path

p = Path('.')
datapath = p / "test_data/"

def load(data_filename):
  data_file = np.load(datapath / data_filename)
  return data_file

