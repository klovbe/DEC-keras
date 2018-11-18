# #encoding=utf-8
#
# import os
# import codecs
# import subprocess
# from time import time
#
# import numpy as np
# import pandas as pd
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import LabelEncoder
# import metrics
# import csv
#
#
# data_dir = "F:/project/Gan/data"
# python_path = "C:/ProgramData/Anaconda/python.exe"
# script_path = "F:/project/autoencoder-rmd/train.py"
# done_path = "F:/project/autoencoder-rmd/done_train.txt"
# done_model_names = []
#
# # name_list = os.listdir(data_dir)
# gamma_list = [0.0, 0.001, 0.005, 0.01, 0.1, 0.5, 1.0, 8.0, 20.0]
# # name_list = ["h_pollen.train", "h_kolod.train", "h_brain.train", "h_usoskin.train"]
# # name_list = ["h_human1.train", "h_human2.train", "h_human3.train", "h_human4.train", "h_mouse1.train", "h_mouse2.train", "h_zelsel.train"]
# # name_list = ["h_human1.train", "h_zelsel.train"]
# # name_list = ["h_kolod.train"]
# name_list= ["h_usoskin_fgene_scale.train","h_kolod_scale.train","h_brain_fgene_scale.train","h_zelsel_fgene_scale.train"]
#
#
# for file in name_list:
#     path = os.path.join(data_dir, file)
#     name = file.split(".")[0].split("_")[1]
#     if os.path.exists("./run_logs/each_dataset/scale") is False:
#         os.mkdir("./run_logs/each_dataset/scale")
#     if os.path.exists("./run_logs/each_dataset/scale/{}".format(name)) is False:
#         os.mkdir("./run_logs/each_dataset/scale/{}".format(name))
#     for gamma in gamma_list:
#         outDir = "./each_dataset/scale/{}/{}".format(name,gamma)
#         if name in done_model_names:
#           print("{} has been trained before".format(name))
#           continue
#         namep = "scale" + name
#         par_dict = {
#         "python": python_path,
#         "script_path": script_path,
#         "model_name": namep,
#         "datapath":path,
#         "outDir": outDir,
#         "gamma":gamma
#         }
#         cmd = "python {script_path} --model_name={model_name}  --train_datapath={datapath} " \
#             " --outDir={outDir} --epoch=1000 --batch_size=128 --gamma={gamma} > ./run_logs/each_dataset/dim1/{model_name}/{gamma}.log 2>&1".format(
#         **par_dict
#             )
#         print("running {}...".format(cmd))
#         ret = subprocess.check_call(cmd, shell=True, cwd="F:/project/autoencoder-rmd")
#         print(ret)
#

#encoding=utf-8

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


data_dir = "/home/xysmlx/data/data_new"
# data_dir = "F:/project/new_data/filter_samp/"
# data_dir = "F:/project/new_data/ercc/"
# data_dir = "F:/project/new_data/cluster/filter_data/new/"
python_path = "/home/xysmlx/anaconda3/bin/python"
script_path = "/home/xysmlx/python_project/DEC-keras/sparse_idec.py"
done_path = "/home/xysmlx/python_project/DEC-keras/done_train.txt"
done_model_names = ["macosko", "shekhar"]
# date_now = time.strftime("%d_%m")
date_now ="0812_lr0.0005"
# date_now = "rpkm_new_0731"
run_log_path = "./{}/run_logs".format(date_now)
out_path = "./{}/out".format(date_now)
if os.path.exists(out_path) is False:
    os.makedirs(out_path)
if os.path.exists(run_log_path) is False:
    os.makedirs(run_log_path)

name_list = os.listdir(data_dir)
print(name_list)
name_list_new = []
for file in name_list:
    if ".csv" in file:
        name_list_new.append(file)
name_list = name_list_new
print(name_list)
early_stopping = [5, 5, 10, 10, 5, 5, 10, 10, 3, 5, 5, 3, 5, 10, 5, 10]
nclusters = [14, 4, 6, 9, 6, 5, 4, 3, 12, 5, 4, 6, 5, 4, 6, 9]
nclusters = [14, 5, 6, 9, 10, 5, 4, 3, 12, 5, 11, 19, 5, 11, 7, 9]
update_interval = [2, 10, 10, 10, 10, 10, 5, 10, 1, 10, 10, 1, 10, 10, 10, 5 ]
# early_stopping = [5, 3, 5, 5]
# early_stopping = [5, 5, 5, 5, 5, 5]
# early_stopping = [10, 10, 10, 10, 10, 10, 10]
for i, file in enumerate(name_list):
    path = os.path.join(data_dir, file)
    name = file.split(".")[0].split("_")[0]
    data_type = file.split(".")[0].split("_")[1]
    run_log_path_name = os.path.join(run_log_path, name)
    if os.path.exists(run_log_path_name) is False:
        os.makedirs(run_log_path_name)
    if name in done_model_names:
      print("{} has been trained before".format(name))
      continue
    par_dict = {
    "python": python_path,
    "script_path": script_path,
    "model_name": name,
    "data_type" :data_type,
    "early_stopping": early_stopping[i],
    "datapath":path,
    "outDir": out_path,
    "run_log_path_name":run_log_path_name,
    "update_interval" : update_interval[i],
    "n_clusters":nclusters[i]
    }
    cmd = "python {script_path} --model_name={model_name}  --train_datapath={datapath} --data_type={data_type} --early_stopping={early_stopping}" \
        " --outDir={outDir} --epoch=500 --batch_size=256 --gamma=0.1 --trans=True --gene_scale=True --update_interval={update_interval} --n_clusters={n_clusters} > {run_log_path_name}/{model_name}.log 2>&1".format(
    **par_dict
        )
    print("running {}...".format(cmd))
    ret = subprocess.check_call(cmd, shell=True, cwd="/home/xysmlx/python_project/DEC-keras")
    print(ret)