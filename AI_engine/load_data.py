import sys
import numpy as np
import pandas as pd
from pandas import read_csv
from pathlib import Path

def load_data(data_path, data_filename):
  data_file = np.load(data_path / data_filename)
  return data_file