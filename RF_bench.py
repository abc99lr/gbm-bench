
# coding: utf-8

# # Random Forest Bench
# 

# In[1]:


import cudf
import cuml
import sklearn as skl 
from cuml import RandomForestClassifier as cuRF
from sklearn.ensemble import RandomForestClassifier as sklRF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from urllib.request import urlretrieve
import gzip
from cuml import ForestInference
import time
from numba import cuda


# In[2]:


print("cudf version: ", cudf.__version__)
print("cuml version: ", cuml.__version__)
print("skl version: ", skl.__version__)


# ## Main benchmarking function

# In[3]:


def start_bench(run_cuml, run_skl, skip_test, estimator_array, stream_array, depth_array, csv_path, X_train, y_train, X_train_np, y_train_np, X_test_np, y_test_np):
    results = []
    for n_estimators in estimator_array:
        for n_streams in stream_array:
            for max_depth in depth_array:
                # cuml Random Forest params
                cu_rf_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'n_bins': 16,
                    'split_algo': 1,
                    'n_streams': n_streams
                }

                cu_fit_time = 0
                skl_fit_time = 0
                acc_score_cuml = 0
                acc_score_skl = 0

                if run_cuml:
                    print("====>cuml====")
                    cuml_params = cu_rf_params.copy()
                    print("    cuml params: ", str(cuml_params))
                    cu_rf = cuRF(**cu_rf_params)
                    print("    cuml model: ", str(cu_rf.get_params()))

                    t0 = time.time()
                    cu_rf.fit(X_train, y_train)
                    cu_fit_time = time.time() - t0

                    print("    cuml fits RF: ", cu_fit_time)

                    if not skip_test:
                        # use a subset of test data to inference 
                        cu_rf_predicted = cu_rf.predict(X_test_np[:1000, :])
                        acc_score_cuml = accuracy_score(cu_rf_predicted, y_test_np[:1000])
                        print("    cuml total time: ", time.time() - t0)
                        print("    cuml acc: ", acc_score_cuml)

                if run_skl and n_streams == 8:
                    print("====>sklearn====")
                    sk_params = cu_rf_params.copy()
                    print("    skl params: ", str(cuml_params))

                    sk_params['n_jobs'] = -1
                    del sk_params['n_bins']
                    del sk_params['split_algo']
                    if 'n_streams' in sk_params:
                        del sk_params['n_streams']                
                    rfc = sklRF(**sk_params)

                    t0 = time.time()
                    rfc.fit(X_train_np, y_train_np)
                    skl_fit_time = time.time() - t0

                    print("    skl fits RF: ", skl_fit_time)
                    
                    if not skip_test:
                        skl_predicted = rfc.predict(X_test_np[:1000, :])
                        acc_score_skl = accuracy_score(skl_predicted, y_test_np[:1000])                
                        print("    skl total time: ", time.time() - t0)
                        print("    skl acc: ", acc_score_skl)

                pd.set_option('display.max_colwidth', 300)
                results.append(dict(cu_fit_time=cu_fit_time, acc_score_cuml=acc_score_cuml, skl_fit_time=skl_fit_time, acc_score_skl=acc_score_skl))
                df = pd.DataFrame(results)
                print(df.to_string())
                df.to_csv(csv_path, mode='a')


# ## Higgs

"""
# In[4]:


def download_higgs(compressed_filepath, decompressed_filepath):
    higgs_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
    if not os.path.isfile(compressed_filepath):
        urlretrieve(higgs_url, compressed_filepath)
    if not os.path.isfile(decompressed_filepath):
        cf = gzip.GzipFile(compressed_filepath)
        with open(decompressed_filepath, 'wb') as df:
            df.write(cf.read())


# In[5]:


# compressed_filepath = '../gbm-bench/data/HIGGS.csv.gz' # Set this as path for gzipped Higgs data file, if you already have
decompressed_filepath = '../gbm-bench/data/HIGGS.csv' # Set this as path for decompressed Higgs data file, if you already have
# download_higgs(compressed_filepath, decompressed_filepath)

col_names = ['label'] + ["col-{}".format(i) for i in range(2, 30)] # Assign column names
dtypes_ls = ['int32'] + ['float32' for _ in range(2, 30)] # Assign dtypes to each column
data = cudf.read_csv(decompressed_filepath, names=col_names, dtype=dtypes_ls)

y_cudf = data['label']
X_cudf = data.drop('label')
from cuml.preprocessing.model_selection import train_test_split
# train_size is the ratio of the entire dataset to be split into training data
X_train, X_test, y_train, y_test = train_test_split(X_cudf, y_cudf, train_size=0.80)

print("Shape of the training data : ", X_train.shape)
print("Shape of the ground truth data used for training : ", y_train.shape)
print("Shape of the testing data : ", X_test.shape)
print("Shape of the ground truth data used for testing : ",y_test.shape)

X_train_np = X_train.as_matrix()
y_train_np = y_train.to_array()
X_test_np = X_test.as_matrix()
y_test_np = y_test.to_array()


# In[6]:


data.head().to_pandas()


# In[7]:


estimator_array = [100, 500, 1000]
stream_array = [8, 10]
depth_array = [8, 12, 16]
run_cuml = True
run_skl = True 
skip_test = True
csv_path = './rf_bench_results/rf_bench_higgs.csv'

start_bench(run_cuml, run_skl, skip_test, estimator_array, stream_array, depth_array, csv_path, X_train, y_train, X_train_np, y_train_np, X_test_np, y_test_np)


# ## Airline

# In[7]:


from datasets import prepare_dataset

data = prepare_dataset('../gbm-bench/data/', 'airline', 115000000)

data.X_train = data.X_train.astype(np.float32)
data.X_test = data.X_test.astype(np.float32)
data.y_train = data.y_train.astype(np.int32)
data.y_test = data.y_test.astype(np.int32)

X_train_np = data.X_train.to_numpy()
X_test_np = data.X_test.to_numpy()
y_train_np = data.y_train.to_numpy()
y_test_np = data.y_test.to_numpy()

X_train_df = cudf.from_pandas(data.X_train)
y_train_df = cudf.from_pandas(data.y_train)

print("NP Shape of the training data : ", X_train_np.shape)
print("NP Shape of the ground truth data used for training : ", y_train_np.shape)
print("NP Shape of the testing data : ", X_test_np.shape)
print("NP Shape of the ground truth data used for testing : ", y_test_np.shape)
print("DF Shape of the training data : ", X_train_df.shape)
print("DF Shape of the ground truth data used for training : ", y_train_df.shape)


# In[10]:


estimator_array = [100, 500, 1000]
stream_array = [8, 10]
depth_array = [8, 12, 16]
run_cuml = True
run_skl = False 
skip_test = False
csv_path = './rf_bench_results/rf_bench_airline.csv'

# CudaAPIError: [1] Call to cuMemcpyHtoD results in CUDA_ERROR_INVALID_VALUE
# got this error even with numpy array with input 
start_bench(run_cuml, run_skl, skip_test, estimator_array, stream_array, depth_array, csv_path, X_train_df, y_train_df, X_train_np, X_test_np, X_test_np, y_test_np)

# ## Epsilon

# In[5]:


from datasets import prepare_dataset

data = prepare_dataset('../gbm-bench/data/', 'epsilon', 500000)


# In[19]:


col_names = ["col-{}".format(i) for i in range(0, 2000)] # Assign column names
X_train_df = pd.DataFrame(data.X_train, columns=col_names, dtype=np.float32)
X_train_df_gpu = cudf.from_pandas(X_train_df)

y_train_df = pd.DataFrame(data.y_train, columns=['label'], dtype=np.int32)
y_train_df_gpu = cudf.from_pandas(y_train_df)

print("NP Shape of the training data : ", data.X_train.shape)
print("NP Shape of the ground truth data used for training : ", data.y_train.shape)
print("NP Shape of the testing data : ", data.X_test.shape)
print("NP Shape of the ground truth data used for testing : ", data.y_test.shape)
print("DF Shape of the training data : ", X_train_df_gpu.shape)
print("DF Shape of the ground truth data used for training : ", y_train_df_gpu.shape)


# In[23]:


estimator_array = [100, 500, 1000]
stream_array = [8, 10]
depth_array = [8, 12, 16]
run_cuml = True
run_skl = True 
skip_test = False
csv_path = './rf_bench_results/rf_bench_epsilon.csv'

start_bench(run_cuml, run_skl, skip_test, estimator_array, stream_array, depth_array, csv_path, X_train_df_gpu, y_train_df_gpu, data.X_train, data.y_train, data.X_test, data.y_test)
"""

# ## Bosch

# In[16]:


from datasets import prepare_dataset

data = prepare_dataset('../gbm-bench/data/', 'bosch', 1184000)

data.X_train = data.X_train.astype(np.float32)
data.X_test = data.X_test.astype(np.float32)
data.y_train = data.y_train.astype(np.int32)
data.y_test = data.y_test.astype(np.int32)

X_train_np = data.X_train.to_numpy()
X_test_np = data.X_test.to_numpy()
y_train_np = data.y_train.to_numpy()
y_test_np = data.y_test.to_numpy()

# OOM 
X_train_df = cudf.from_pandas(data.X_train)
y_train_df = cudf.from_pandas(data.y_train)
X_train_df = X_train_df.fillna(0)
y_train_df = y_train_df.fillna(0)

print("NP Shape of the training data : ", X_train_np.shape)
print("NP Shape of the ground truth data used for training : ", y_train_np.shape)
print("NP Shape of the testing data : ", X_test_np.shape)
print("NP Shape of the ground truth data used for testing : ", y_test_np.shape)
print("DF Shape of the training data : ", X_train_df.shape)
print("DF Shape of the ground truth data used for training : ", y_train_df.shape)


# In[23]:


# X_train contains NaN 
print(np.where(np.isnan(X_train_np)))
# fill with zero 
# otherwise sklearn has error: ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
X_train_np = np.nan_to_num(X_train_np)

# X_test contains NaN 
print(np.where(np.isnan(X_test_np)))
# fill with zero 
# otherwise sklearn has error: ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
X_test_np = np.nan_to_num(X_test_np)


# In[26]:


estimator_array = [100, 500, 1000]
stream_array = [8, 10]
depth_array = [8, 12]
run_cuml = True
run_skl = True 
skip_test = False
csv_path = './rf_bench_results/rf_bench_bosch.csv'

# use X_train_np, y_train_np for cuml as GPU data frame OOM 
# RuntimeError: Exception occured! file=/gpfs/fs1/rlan/rf_bench/cuml/cpp/src/decisiontree/quantile/quantile.cuh line=110: FAIL: call='cudaGetLastError()'. Reason:out of memory
start_bench(run_cuml, run_skl, skip_test, estimator_array, stream_array, depth_array, csv_path, X_train_df, y_train_df, X_train_np, y_train_np, X_test_np, y_test_np)

