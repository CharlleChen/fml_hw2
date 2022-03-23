from libsvm.svmutil import *

import numpy as np
import pandas as pd

input_name = "abalone.data"
output_name = "abalone_formatted.txt"

with open(input_name, 'r') as f:
  lines = f.readlines()

ret = ''

d = {
  'M': "1:1 2:0 3:0 ",
  'F': "1:0 2:1 3:0 ",
  'I': "1:0 2:0 3:1 "
}

for l in lines:
  items = l.split(',')
  
  if int(items[-1]) <= 9:
    ret += '+1 '
  else:
    ret += '-1 '
  
  features = [float(i) for i in items[1:-1]]

  ret += d[items[0]]

  idx = 4
  for feat in features:
    ret += f"{idx}:{feat} "
    idx += 1
  ret += '\n'
with open(output_name, 'w') as f:
  f.write(ret)


    