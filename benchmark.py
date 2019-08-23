from joblib import dump, load
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
result_path = "./results/"

def train_xgb(max_depth, n_trees, n_cols, X_train, y_train):
    print("===>Training XGB - D: %d, T: %d, C: %d" % (max_depth, n_trees, n_cols))

    if Path(model_path+'xgb_D'+str(max_depth)+'_T'+str(n_trees)+'_C'+str(n_cols)+'.model').is_file():
        print("    Model exist, exiting")
        return 
    
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

    xgb_tree.save_model(model_path+'xgb_D'+str(max_depth)+'_T'+str(n_trees)+'_C'+str(n_cols)+'.model')

def train_all(max_depth, n_trees, n_cols, X_train, y_train):
    train_xgb(max_depth, n_trees, n_cols, X_train, y_train)

def test_all(max_depth, n_trees, n_cols, test_rows, test_models, X_test, y_test, dataset):
    xgb_tree = xgb.Booster()
    xgb_tree.load_model(model_path+'xgb_D'+str(max_depth)+'_T'+str(n_trees)+'_C'+str(n_cols)+'.model') 

    tl_model = treelite.Model.from_xgboost(xgb_tree)
    toolchain = 'gcc'
    tl_model.export_lib(toolchain=toolchain, libpath=model_path +'treelite_D'+str(max_depth)+'_T'+str(n_trees)+'_C' + str(n_cols)+'.so', params={'parallel_comp': 40}, verbose=False)
    tl_predictor = treelite.runtime.Predictor(model_path +'treelite_D'+str(max_depth)+'_T'+str(n_trees)+'_C' + str(n_cols)+'.so', verbose=False)

    fm = {}
    algos = ['NAIVE', 'TREE_REORG', 'BATCH_TREE_REORG']

    for algo in algos:
        fm[algo] = FIL.load(model_path+'xgb_D'+str(max_depth)+'_T'+str(n_trees)+'_C'+str(n_cols)+'.model',
                            algo=algo, output_class=True, threshold=0.50)


    for n_rows in test_rows:
        print("===>Testing - D: %d, T: %d, C: %d, R: %d" % (max_depth, n_trees, n_cols, n_rows))

        X_test_g = cuda.to_device(np.ascontiguousarray(X_test[:n_rows, :]))
        X_test_c = X_test[:n_rows, :]
        y_test_t = y_test[:n_rows]

        write_csv = []
        common_csv = []
        common_csv.append(dataset)
        common_csv.append(max_depth)
        common_csv.append(n_trees)
        common_csv.append(n_cols)
        common_csv.append(n_rows)

        # Test XGB CPU
        if 'xgb_cpu' in test_models:
            write_csv_xgb_cpu = copy.deepcopy(common_csv)
            write_csv_xgb_cpu.append("xgb_cpu")

            xgb_tree.set_param({'predictor': 'cpu_predictor'})
            xgb_tree.set_param({'n_gpus': '0'})

            dtest = xgb.DMatrix(X_test_c, silent=False)
            each_run = []        

            for run in range(repeat):

                _ = xgb_tree.predict(dtest)
                start_xgb = time.time()
                xgb_preds_cpu = xgb_tree.predict(dtest)
                stop_xgb = time.time()

                xgb_acc_cpu = accuracy_score(xgb_preds_cpu > 0.5, y_test_t)
                each_run.append((stop_xgb - start_xgb) * 1000)

            write_csv_xgb_cpu.append(min(each_run))
            write_csv_xgb_cpu.append(xgb_acc_cpu)
            write_csv.append(write_csv_xgb_cpu)
            print("    XGboost CPU testing time: ", min(each_run), " XGboost CPU acc: ", xgb_acc_cpu)

        # Test XGB GPU 
        if 'xgb_gpu' in test_models:
            write_csv_xgb_gpu = copy.deepcopy(common_csv)
            write_csv_xgb_gpu.append("xgb_gpu")

            xgb_tree.set_param({'predictor': 'gpu_predictor'})
            xgb_tree.set_param({'n_gpus': '1'})
            xgb_tree.set_param({'gpu_id': '0'})

            dtest = xgb.DMatrix(X_test_c, silent=False)
            each_run = []

            for run in range(repeat):

                _ = xgb_tree.predict(dtest)
                start_xgb = time.time()
                xgb_preds_gpu = xgb_tree.predict(dtest)
                stop_xgb = time.time()

                xgb_acc_gpu = accuracy_score(xgb_preds_gpu > 0.5, y_test_t)
                each_run.append((stop_xgb - start_xgb) * 1000)

            write_csv_xgb_gpu.append(min(each_run))
            write_csv_xgb_gpu.append(xgb_acc_gpu)
            write_csv.append(write_csv_xgb_gpu)
            print("    XGboost GPU testing time: ", min(each_run), " XGboost GPU acc: ", xgb_acc_gpu)

        # Test Treelite
        if 'treelite' in test_models:
            write_csv_tl = copy.deepcopy(common_csv)
            write_csv_tl.append("treelite")

            tl_batch = treelite.runtime.Batch.from_npy2d(X_test_c)

            each_run = []        

            for run in range(repeat):

                _ = tl_predictor.predict(tl_batch)
                start_tl = time.time()
                tl_pred = tl_predictor.predict(tl_batch)
                stop_tl = time.time()

                tl_acc = accuracy_score(tl_pred > 0.5, y_test_t)
                each_run.append((stop_tl - start_tl) * 1000)

            write_csv_tl.append(min(each_run))
            write_csv_tl.append(tl_acc)    
            write_csv.append(write_csv_tl)
            print("    Treelite CPU testing time: ", min(each_run), " Treelite CPU acc: ", tl_acc)

        # Test FIL 
        if 'fil' in test_models:
            for algo in algos:
                write_csv_fil = copy.deepcopy(common_csv)
                write_csv_fil.append("fil_" + str(algo))

                each_run = []
                for run in range(repeat):
                    _ = fm[algo].predict(X_test_g)
                    start_fil = time.time()
                    fil_preds = fm[algo].predict(X_test_g)
                    stop_fil = time.time()

                    fil_acc = accuracy_score(fil_preds, y_test_t)
                    each_run.append((stop_fil - start_fil) * 1000)

                write_csv_fil.append(min(each_run))
                write_csv_fil.append(fil_acc)
                write_csv.append(write_csv_fil)
                print("    FIL %s testing time: " % algo, min(each_run), " FIL %s acc: " % algo, fil_acc)

        with open(result_path + dataset + '.csv', 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for i in range(len(write_csv)):
                wr.writerow(write_csv[i])

if __name__ == '__main__':

    # control the test cases 
    test_trees = [100, 250, 500, 750, 1000]
    test_depth = [5, 6, 7, 8]
    test_models = ['xgb_cpu', 'fil', 'treelite', 'xgb_gpu']
    dataset = "epsilon"
    test_rows = [100, 1000, 10000, 100000, 1000000]
    test_cols = [1024]
    dataset_row = 0

    if dataset == "higgs":
        # 11M 
        dataset_row = 11000000
        test_cols = [28]
    elif dataset == "airline":
        # 115M 
        dataset_row = 115000000
        test_cols = [13]
    elif dataset == "bosch":
        # 1.184M
        dataset_row = 1184000
        test_cols = [968]
    elif dataset == "epsilon":
        # 500K 
        dataset_row = 500000
        test_cols = [2000]

    header_csv = ["dataset", "depth", "n_trees", "n_cols", "n_rows", "predictor", "time", "acc"]
    with open(result_path + dataset + '.csv', 'a', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(header_csv)

    print("===========================================")
    print("Benchmark Starts")

    print("===========================================")
    print("    Preparing dataset " + dataset)
    data = prepare_dataset(data_path, dataset, dataset_row)

    if dataset == 'epsilon':
        X_train = data.X_train.astype(np.float32)
        X_test = data.X_test.astype(np.float32)
        y_train = data.y_train.astype(np.float32)
        y_test = data.y_test.astype(np.float32)
    # to numpy array 
    else:
        X_train = data.X_train.to_numpy(np.float32)
        X_test = data.X_test.to_numpy(np.float32)
        y_train = data.y_train.to_numpy(np.float32)
        y_test = data.y_test.to_numpy(np.float32)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    for n_cols in test_cols:
        for n_trees in test_trees:
            for max_depth in test_depth:
                train_xgb(max_depth, n_trees, n_cols, X_train, y_train)

                print("===========================================")
                test_all(max_depth, n_trees, n_cols, test_rows, test_models, X_test, y_test, dataset)
