from cmath import exp
from libsvm.svmutil import *
from IPython import embed as e
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import re
import os, select


input_name = "abalone_formatted.txt"
output_name = ['train.txt', 'text.txt']

y, x = svm_read_problem(input_name, return_scipy = True)
train_X = x[:3133]
train_y = y[:3133]
test_X = x[3133:]
test_y = y[3133:]

scale_param = csr_find_scale_param(train_X, lower=0)
scaled_train_X = csr_scale(train_X, scale_param)
scaled_test_X = csr_scale(test_X, scale_param)

# shuffle train data
shuffle_index = np.arange(3133)
np.random.shuffle(shuffle_index)
scaled_train_X = scaled_train_X[shuffle_index, :]
train_y = train_y[shuffle_index]

# 5 folds
kf = KFold(n_splits=5)
pipe_out, pipe_in = os.pipe()

def experiment(d, c):
  errs = []
  test_errs = []
  svs = []
  bsvs = []
  for train_index, test_index in kf.split(scaled_train_X):
    X_train, X_test = scaled_train_X[train_index], scaled_train_X[test_index]
    y_train, y_test = train_y[train_index], train_y[test_index]

    prob  = svm_problem(y_train, X_train)
    # print(f'-t 1 -d {d} -c {c} -q')
    param = svm_parameter(f'-t 1 -d {d} -c {c}')
    stdout = os.dup(1)
    os.dup2(pipe_in, 1)
    m = svm_train(prob, param)
    # check if we have more to read from the pipe
    def more_data():
      r, _, _ = select.select([pipe_out], [], [], 0)
      return bool(r)

    # read the whole pipe
    def read_pipe():
      out = ''
      while more_data():
        out += os.read(pipe_out, 1024).decode("utf-8") 
      return out
    os.dup2(stdout,1)
    output = read_pipe()
    p = re.compile("nSV.*\,")
    nsv = int(p.search(output)[0][6:-1])
    p = re.compile("nBSV.*\n")
    nbsv = int(p.search(output)[0][7:-1])
     
    svs.append(nsv)
    
    bsvs.append(nbsv)
    preds = svm_predict(y_test, X_test, m, "-q")
    acc = np.sum(preds[0] == y_test) / len(preds[0])

    preds = svm_predict(test_y, scaled_test_X, m, "-q")
    test_acc = np.sum(preds[0] == test_y) / len(preds[0])

    errs.append(1-acc)
    test_errs.append(1-test_acc)

  m_svs = np.mean(svs)
  m_bsvs = np.mean(bsvs)

  means = np.mean(errs)
  std = np.std(errs)

  test_means = np.mean(test_errs)
  test_std = np.std(test_errs)

  return means, std, [m_svs, m_svs-m_bsvs, test_means, test_std]

def test_for_d(d, ax):
  global best_error
  global best_pair
  print(f"starting d: {d}")

  stds, means, cs = [], [], []

  for k in range(-K, K + 1):
    c = 3 ** k
    cs.append(c)
    print(f"c: {c}")

    m_, s_, _ = experiment(d, c)
    if m_ < best_error:
      best_error = m_
      best_pair = [d, c]

    stds.append(s_)
    means.append(m_)

  stds = np.array(stds)
  means = np.array(means)
  cs = np.array(cs)

  ax.plot(np.log(cs) / np.log(10), means, 'k')
  ax.plot(np.log(cs) / np.log(10), means + stds, ':k')
  ax.plot(np.log(cs) / np.log(10), means - stds, ':k')

  ax.grid()
  ax.set_title(f"d = {d}")

# Part 3
best_error = 1
best_pair = [0, 0]
K = 10
fig, axs = plt.subplots(2, 3)
for d in range(5):
  test_for_d(d + 1, axs[d//3, d%3])
fig.text(0.5, 0.04, 'log_10(C)', ha='center', va='center')
fig.text(0.06, 0.5, 'Classification error', ha='center', va='center', rotation='vertical')
plt.show()

print(f"Best error {best_error} achieved by the pair {best_pair}")


# Part 4
best_c = 59049

ds, means, stds, nsvs, nsvs_on_m, test_means, test_stds = [], [], [], [], [], [], []
for d in range(5):
  d += 1
  ds.append(d)
  m_, s_, [nsv_, nsv_m_, t_m_, t_s_] = experiment(d, best_c)
  means.append(m_)
  stds.append(s_)
  test_means.append(t_m_)
  test_stds.append(t_s_)
  nsvs.append(nsv_)
  nsvs_on_m.append(nsv_m_)

means = np.array(means)
stds = np.array(stds)
test_means = np.array(test_means)
test_stds = np.array(test_stds)
nsvs = np.array(nsvs)
nsvs_on_m = np.array(nsvs_on_m)


fig, axs = plt.subplots(2,2)
axs[0, 0].plot(ds, means, 'k')
axs[0, 0].plot(ds, means + stds, 'k:')
axs[0, 0].plot(ds, means - stds, 'k:')

axs[0, 0].grid()
axs[0, 0].set_ylabel("cross validation error")

axs[0, 1].plot(ds, test_means, 'k')
axs[0, 1].plot(ds, test_means + test_stds, 'k:')
axs[0, 1].plot(ds, test_means - test_stds, 'k:')

axs[0, 1].grid()
axs[0, 1].set_ylabel("test error")

axs[1, 0].plot(ds, nsvs, 'k')
axs[1, 0].grid()
axs[1, 0].set_ylabel("number of support vectors")

axs[1, 1].plot(ds, nsvs_on_m, 'k')
axs[1, 1].grid()
axs[1, 1].set_ylabel("number of support vectors on margin hyperplane")

fig.text(0.5, 0.04, 'd', ha='center', va='center')

plt.show()

# Part 5
best_C = 59049
best_d = 2

def experiment_5(d, c):
  errs = []
  test_errs = []
  for train_index, test_index in kf.split(scaled_train_X):
    X_train, X_test = scaled_train_X[train_index], scaled_train_X[test_index]
    y_train, y_test = train_y[train_index], train_y[test_index]

    prob  = svm_problem(y_train, X_train)
    # print(f'-t 1 -d {d} -c {c} -q')
    param = svm_parameter(f'-t 1 -d {d} -c {c} -q')
    m = svm_train(prob, param)
    # check if we have more to read from the pipe
   
    preds = svm_predict(y_test, X_test, m, "")
    acc = np.sum(preds[0] == y_test) / len(preds[0])

    preds = svm_predict(test_y, scaled_test_X, m, "")
    test_acc = np.sum(preds[0] == test_y) / len(preds[0])

    errs.append(1-acc)
    test_errs.append(1-test_acc)
  return [errs, test_errs]

d = experiment_5(best_d, best_C)
fig, axs = plt.subplots(1, 2)
xs = ['fold{}'.format(i+1) for i in range(5)]
axs[0].scatter(xs, d[0])
axs[0].set_title("cross validation")
axs[1].scatter(xs, d[1])
axs[1].set_title("test")

fig.text(0.5, 0.04, 'Validation fold #', ha='center', va='center')
fig.text(0.06, 0.5, 'Classification error', ha='center', va='center', rotation='vertical')

plt.show()