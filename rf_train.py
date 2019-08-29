# Setup a simple logger
import logging
logging.basicConfig(format="%(asctime)s - %(process)d - %(message)s",
                    level=logging.INFO)

from cuml import RandomForestClassifier as cuRF
from cuml.dask.ensemble import RandomForestClassifier as cuRFC_mg

from sklearn.ensemble import RandomForestClassifier as sklRF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from urllib.request import urlretrieve
import gzip
import os.path
import time
from numba import cuda

# Distributed stuff
from dask.distributed import Client, wait, progress
from dask_cuda import LocalCUDACluster, DGX

from datasets import prepare_dataset

# BASE_PATH = '/gpfs/fs1/jzedlewski/data'
# USE_HIGGS = False
data_path = './data/'

# from contextlib import contextmanager

# @contextmanager
# def timed(name):
#     t0 = time.time()
#     yield
#     t1 = time.time()
#     print("%32s: %6.3f" %(name, t1 - t0))

# def download_higgs(compressed_filepath, decompressed_filepath):
#     higgs_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
#     if not os.path.isfile(compressed_filepath):
#         urlretrieve(higgs_url, compressed_filepath)
#     if not os.path.isfile(decompressed_filepath):
#         cf = gzip.GzipFile(compressed_filepath)
#         with open(decompressed_filepath, 'wb') as df:
#             df.write(cf.read())


# def make_data(use_higgs, n_rows, random_state=None):
#     import cudf
#     import dask_cudf

#     col_names = ['label'] + ["col-{}".format(i) for i in range(2, 30)] # Assign column names
#     dtypes_ls = ['int32'] + ['float32' for _ in range(2, 30)] # Assign dtypes to each column

#     compressed_filepath = os.path.join(BASE_PATH, 'HIGGS.csv.gz') # Set this as path for gzipped Higgs data file, if you already have
#     decompressed_filepath = os.path.join(BASE_PATH, 'HIGGS.csv') # Set this as path for decompressed Higgs data file, if you already have
#     if use_higgs:
#         download_higgs(compressed_filepath, decompressed_filepath)
#         data = cudf.read_csv(decompressed_filepath, names=col_names, dtype=dtypes_ls)
#         if n_rows < data.shape[0]:
#             data = data[:n_rows,:]
#         # XXX testme
#     else:
#         from sklearn import datasets
#         print("Random state: ", random_state)
#         X_in, y_in = datasets.make_classification(n_features=28, n_samples=n_rows, random_state=random_state)
#         data_pd = pd.DataFrame(np.hstack((y_in[:,np.newaxis], X_in)), columns=col_names)
#         data = cudf.DataFrame.from_pandas(data_pd)
#         data['label'] = data['label'].astype(np.int32)
#     return data

def fit_sklearn(cu_rf_params, X_train, y_train, X_test, y_test):
    print("--- Basic sklearn ---")
    sk_params = cu_rf_params.copy()
    sk_params = cu_rf_params.copy()
    sk_params['n_jobs'] = -1
    del sk_params['n_bins']
    del sk_params['split_algo']
    if 'n_streams' in sk_params:
        del sk_params['n_streams']

    print("SKL params: ", str(sk_params))
    from sklearn.ensemble import RandomForestClassifier as skRFC

    rfc = skRFC(**sk_params)
    t0 = time.time()
    rfc.fit(X_train, y_train)
    fit_time = time.time() - t0

    predicted = rfc.predict(X_test)
    acc_score_skl = accuracy_score(predicted, y_test)

    return fit_time, acc_score_skl

def compare_rf_runtimes(X_train, y_train, X_test, y_test, 
                        train_size, n_estimators, n_workers,
                        max_depth=16,
                        n_streams=4,
                        scheduler_address=None,
                        run_cuml=True, run_sklearn=True, skip_predict=False,
                        explicit_persist=True, random_state=None):
    # print("--- Begin cluster setup ---")
    # if scheduler_address is None:
    #     cluster = LocalCUDACluster(threads_per_worker=1, n_workers=n_workers)
    #     c = Client(cluster)
    # else:
    #     print("--- Reuse exising scheduler: %s ---" % scheduler_address)
    #     cluster = None
    #     c = Client(scheduler_address)

    import cudf, dask_cudf

    # workers = c.has_what().keys()
    # if scheduler_address and n_workers == 0:
    #     print("Defaulting to full set of %d workers" % len(workers))
    #     n_workers = len(workers)

    # print("--- All Workers: ", len(workers), "Subset to: ", n_workers, " --- ")
    # print("All workers: ", workers)
    # workers = [k for k in list(workers)[:n_workers]]
    # print("Post workers: ", workers)

    # test_size = 1000
    # data = make_data(USE_HIGGS, train_size + test_size, random_state=random_state)

    # Train / test splits
    # X, y = data[data.columns.difference(['label'])].as_matrix(), data['label'].to_array() # Separate data into X and y
    # del data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    print("X train: ", X_train.shape, "Y train: ", y_train.shape,
          "X test: ", X_test.shape, "Y test: ", y_test.shape)

    #
    # Set params
    #
    cu_rf_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'n_bins': 16,
        'split_algo': 1,
        'n_streams': n_streams
    }

    if run_cuml:
        print("--- Basic cuML ---")
        sg_params = cu_rf_params.copy()
        print("CUML params: ", str(sg_params))
        cu_rf = cuRF(**sg_params)
        print("--- Basic model: \n\n---", str(cu_rf.get_params()))
        X_train_g = cuda.to_device(np.ascontiguousarray(X_train))
        X_train_df = cudf.DataFrame.from_gpu_matrix(X_train_g)
        y_train_g = cuda.to_device(np.ascontiguousarray(y_train))
        y_train_df = cudf.Series(y_train_g)
        t0 = time.time()
        cu_rf.fit(X_train, y_train)
        sg_fit_time = time.time() - t0

        print("cuML Fit RF in ", sg_fit_time)
        cu_rf_sg_predicted = cu_rf.predict(X_test)
        acc_score_cuml = accuracy_score(cu_rf_sg_predicted, y_test)
        print("Total fit + predict SG: ", time.time() - t0)
    else:
        sg_fit_time = 0.0
        acc_score_cuml = 0.0

    if run_sklearn:
        skl_fit_time, acc_score_skl = fit_sklearn(cu_rf_params, X_train, y_train, X_test, y_test)
    else:
        skl_fit_time = 0.0
        acc_score_skl = 0.0

    # X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
    # X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_workers)

    # y_cudf = np.array(pd.DataFrame(y_train).values)
    # y_cudf = y_cudf[:, 0]
    # y_cudf = cudf.Series(y_cudf)
    # y_train_df = dask_cudf.from_cudf(y_cudf, npartitions=n_workers)

    # if explicit_persist:
    #     X_train_df, y_train_df = c.persist([X_train_df, y_train_df],
    #                                        workers={X_train_df: workers,
    #                                                 y_train_df: workers})
    #     print("Persisted X and y to all workers explicitly")
    # else:
    #     X_train_df = X_train_df.persist()
    #     y_train_df = y_train_df.persist()
    #     print("Standard persist for X and y")

    # print("Wait for data distribution: ")
    # progress(X_train_df, y_train_df)

    # print("--- Begin fit ---")
    # print("Dask params: ", str(cu_rf_params))
    # cu_rf_mg = cuRFC_mg(**cu_rf_params, workers=workers)
    # t1 = time.time()
    # cu_rf_mg.fit(X_train_df, y_train_df)
    # wait(cu_rf_mg)
    # wait(cu_rf_mg.rfs)
    # mg_fit_time = time.time() - t1
    # print("Fit dask in ", time.time() - t1)
    # if skip_predict:
    #     acc_score_dask = 0.0
    # else:
    #     cu_rf_mg_predicted = cu_rf_mg.predict(X_test)
    #     acc_score_dask = accuracy_score(cu_rf_mg_predicted, y_test)
    # print("Fit and predict combo MG: ", time.time() - t1)

    # if cluster:
    #     cluster.close()

    return dict(size=train_size,
                n_estimators=n_estimators,
                n_streams=n_streams,
                max_depth=max_depth,
                sg_fit_time=sg_fit_time,
                # mg_fit_time=mg_fit_time,
                skl_fit_time=skl_fit_time,
                acc_score_skl=acc_score_skl,
                acc_score_cuml=acc_score_cuml,
                # acc_score_dask=acc_score_dask,
                n_workers=n_workers)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", "-n", type=int, default=100000)
    parser.add_argument("--max-depth", "-d", type=str, default="8,12,16")
    parser.add_argument("--scheduler", "-s", type=str, default=None)
    parser.add_argument("--skip-cuml", action="store_true", default=False)
    parser.add_argument("--skip-skl", action="store_true", default=False)
    parser.add_argument("--skip-predict", action="store_true", default=False)
    parser.add_argument("--implicit-persist", action="store_true", default=False)
    parser.add_argument("--workers", type=str, default="1")
    parser.add_argument("--estimators", type=str, default="100,500,1000")
    parser.add_argument("--n-streams", type=str, default="8,10")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="higgs")
    args = parser.parse_args()

    pd.set_option('display.max_colwidth', 200)

    results = []
    n_samples = args.n_samples
    scheduler_address = args.scheduler
    dataset = args.dataset
    explicit_persist = (not args.implicit_persist)
    worker_array = list(map(int, args.workers.split(",")))
    estimator_array = list(map(int, args.estimators.split(",")))
    streams_array = list(map(int, args.n_streams.split(",")))
    depth_array = list(map(int, args.max_depth.split(",")))

    print("Will run with worker counts: ", worker_array)
    print("Will run with estimator counts: ", estimator_array)
    print("Will run with depth counts: ", depth_array)
    print("Will run on dataset: ", dataset)

    if dataset == "higgs":
        # 11M 
        dataset_row = 11000000
        dataset_col = 28
    elif dataset == "airline":
        # 115M 
        dataset_row = 115000000
        dataset_col = 13
    elif dataset == "bosch":
        # 1.184M
        dataset_row = 1184000
        dataset_col = 968
    elif dataset == "epsilon":
        # 500K 
        dataset_row = 500000
        dataset_col = 2000
    else:
        print("Error Not Supported")

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

    # by pass n_samples 
    n_samples = X_train.shape[0]
    print("Will run number of rows: ", n_samples)

    for n_estimators in estimator_array:
        for n_workers in worker_array:
            for n_streams in streams_array:
                for max_depth in depth_array:
                    try:
                        results.append(compare_rf_runtimes(X_train, y_train, X_test, y_test, 
                                                        n_samples, n_estimators=n_estimators,
                                                        max_depth=max_depth,
                                                        n_workers=n_workers,
                                                        n_streams=n_streams,
                                                        run_cuml=(not args.skip_cuml),
                                                        run_sklearn=(not args.skip_skl),
                                                        scheduler_address=scheduler_address,
                                                        skip_predict=args.skip_predict,
                                                        random_state=args.random_state,
                                                        explicit_persist=explicit_persist))
                    except Exception as e:
                        print("Failed to fit for %d workers, %d estimators" % (n_workers, n_estimators))
                        print(e)
                        print("------------------")
                        raise e
                    pd.set_option('display.max_colwidth', 300)
                    df = pd.DataFrame(results)
                    print(df.to_string())

