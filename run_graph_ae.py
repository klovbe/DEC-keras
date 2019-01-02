import os
import codecs
import subprocess
from time import time

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import metrics
import csv
import time


# data_dir = "F:/project/new_data/cluster/filter_data/"
# # data_dir = "F:/project/new_data/filter_samp/"
# # data_dir = "F:/project/new_data/ercc/"
# python_path = "C:/Users/klovbe/Anaconda3/envs/tensorflow_gpu/python.exe"
# script_path = "F:/project/DEC-keras/pure_ae.py"
# done_path = "F:/project/DEC-keras/pure_ae/done_train.txt"
done_model_names = ['zeiselercc',"melanoma"]
data_dir = "/home/xysmlx/data/run_data"
# data_dir = "F:/project/new_data/filter_samp/"
# data_dir = "F:/project/new_data/ercc/"
# data_dir = "F:/project/new_data/cluster/filter_data/new/"
python_path = "/home/xysmlx/anaconda3/bin/python"
script_path = "/home/xysmlx/python_project/DEC-keras/graph_ae.py"
done_path = "/home/xysmlx/python_project/DEC-keras/done_train.txt"
date_now = "19_1_2"
# date_now = "rpkm_new_0731"
run_log_path = "./graph_ae/{}/run_logs".format(date_now)
out_path = "./graph_ae/{}/out".format(date_now)


name_list = os.listdir(data_dir)
# name_list_new = []
# for file in name_list:
#     if "count.csv" in file:
#         name_list_new.append(file)
# name_list = name_list_new
alpha_list = [0, 0.001, 0.01]

for i, file in enumerate(name_list):
    path = os.path.join(data_dir, file)
    name = file.split(".")[0].split("_")[0]
    data_type = file.split(".")[0].split("_")[1]
    for alpha in alpha_list:
        run_log_path_name = os.path.join(run_log_path, name)
        run_log_path_name = os.path.join(run_log_path_name, str(alpha))
        out_dir = os.path.join(out_path, name)
        out_dir = os.path.join(out_dir, str(alpha))
        if os.path.exists(run_log_path_name) is False:
            os.makedirs(run_log_path_name)
        if os.path.exists(out_dir) is False:
            os.makedirs(out_dir)
        if name in done_model_names:
          print("{} has been trained before".format(name))
          continue
        par_dict = {
        "python": python_path,
        "script_path": script_path,
        "model_name": name,
        "data_type" :data_type,
        "datapath":path,
        "outDir": out_dir,
        "run_log_path_name": run_log_path_name,
        "alpha": alpha
        }
        cmd = "python {script_path} --name={model_name}  --train_datapath={datapath} --data_type={data_type} " \
            " --outDir={outDir}  --alpha={alpha} > {run_log_path_name}/{model_name}.log 2>&1".format(
        **par_dict
            )
        print("running {}...".format(cmd))
        ret = subprocess.check_call(cmd, shell=True, cwd="/home/xysmlx/python_project/DEC-keras")
        print(ret)