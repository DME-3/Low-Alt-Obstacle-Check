import datetime
import paramiko
import json
import pandas as pd
import numpy as np
import os
import sys
from glob import glob





if __name__ == "__main__":
  try:
      arg1 = sys.argv[1]
  except IndexError:
      print('Usage: ' + os.path.basename(__file__) + ' <path with dataframes json files to process>')
      sys.exit(1)

