# from joblib import dump, load
import numpy as np
import time
from cuml import ForestInference as FIL

from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import treelite
import treelite.runtime
import xgboost as xgb

from numba import cuda
import cudf
import copy 
import csv
from pathlib import Path

from datasets import prepare_dataset

repeat = 3

def simulate_data(m, n, k=2, random_state=None, classification=True):
    if classification:
        features, labels = make_classification(n_samples=m, n_features=n,
                                                n_informative=int(n/5), n_classes=k,
                                                random_state=random_state, shuffle=True)
    else:
        features, labels = make_regression(n_samples=m, n_features=n,
                                            n_informative=int(n/5), n_targets=1,
                                            random_state=random_state, shuffle=True)
    return features.astype(np.float32), labels.astype(np.float32)

model_path = "./models/"
data_path = "./data/"
# result_path = "./"

# control the test cases 
n_trees = 100
max_depth = 5
dataset = "higgs"
dataset_row = 11000000
n_cols = 28

print("===========================================")
print("Benchmark Starts")

print("===========================================")
print("    Preparing dataset " + dataset)
data = prepare_dataset(data_path, dataset, dataset_row)

# to numpy array 
X_train = data.X_train.to_numpy(np.float32)
X_test = data.X_test.to_numpy(np.float32)
y_train = data.y_train.to_numpy(np.float32)
y_test = data.y_test.to_numpy(np.float32)

X_test_g = cuda.to_device(np.ascontiguousarray(X_test[:1000000, :]))
X_test_c = X_test[:1000000, :]
y_test_t = y_test[:1000000]


"""
print("===>Training XGB - D: %d, T: %d, C: %d" % (max_depth, n_trees, n_cols))

dtrain = xgb.DMatrix(X_train, label=y_train, silent=False)

# instantiate params
params = {}
# general params
general_params = {'silent': 0}
params.update(general_params)
# learning task params
learning_task_params = {}
learning_task_params['eval_metric'] = 'error'
# predict 0 or 1 instead of probability 
learning_task_params['objective'] = 'binary:logistic'
learning_task_params['max_depth'] = max_depth
learning_task_params['base_score'] = 0.5
# use GPU training to save time 
learning_task_params['tree_method'] = 'gpu_hist'
learning_task_params['n_gpus'] = 1
learning_task_params['gpu_id'] = 0
params.update(learning_task_params)    

start_xgb = time.time()
xgb_tree = xgb.train(params, dtrain, n_trees)
stop_xgb = time.time()

print("    XGboost training time: ", stop_xgb - start_xgb)

if save_load: 
    xgb_tree.save_model(model_path+'xgb_D'+str(max_depth)+'_T'+str(n_trees)+'_C'+str(n_cols)+'.model')
"""

xgb_tree = xgb.Booster()
xgb_tree.load_model(model_path+'xgb_D'+str(max_depth)+'_T'+str(n_trees)+'_C'+str(n_cols)+'.model') 

xgb_tree.set_param({'predictor': 'gpu_predictor'})
xgb_tree.set_param({'n_gpus': '1'})
xgb_tree.set_param({'gpu_id': '0'})
X_test_g_cudf = cudf.DataFrame.from_gpu_matrix(X_test_g)
# X_test_g_cudf = cudf.DataFrame.from_pandas(data.X_test[:1000000])
dtest = xgb.DMatrix(X_test_g_cudf, silent=False)

each_run = []
for i in range(repeat):
    _ = xgb_tree.predict(dtest)
    start_xgb = time.time()
    xgb_preds_gpu = xgb_tree.predict(dtest)
    stop_xgb = time.time()
    each_run.append((stop_xgb - start_xgb) * 1000)

print(each_run)
xgb_acc_gpu = accuracy_score(xgb_preds_gpu > 0.5, y_test_t)
print("    XGboost GPU testing time: ", min(each_run), " XGboost GPU acc: ", xgb_acc_gpu)

"""
fm = {}
algos = ['NAIVE']

for algo in algos:
    fm[algo] = FIL.load(model_path+'xgb_D'+str(max_depth)+'_T'+str(n_trees)+'_C'+str(n_cols)+'.model',
                        algo=algo, output_class=True, threshold=0.50)

for algo in algos:
    each_run = []
    for run in range(repeat):
        _ = fm[algo].predict(X_test_g)
        start_fil = time.time()
        fil_preds = fm[algo].predict(X_test_g)
        stop_fil = time.time()

        fil_acc = accuracy_score(fil_preds, y_test_t)
        each_run.append((stop_fil - start_fil) * 1000)

    print(each_run)
    print("    FIL %s testing time: " % algo, min(each_run), " FIL %s acc: " % algo, fil_acc)
"""

