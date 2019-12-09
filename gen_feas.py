#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: quincy qiang 
@license: Apache Licence 
@file: gen_feas.py 
@time: 2019/12/09
@software: PyCharm 
"""
# !/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincyqiang
@license: Apache Licence
@file: gen_feas.py
@time: 2019-11-16 11:52
@description:
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import *
from numpy import random
from pathlib import Path
import gc
import datetime
from sklearn.utils import shuffle
from utils import *
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

start_time = time.time()

X_train = pd.read_pickle('tmp/X_train.pickle')
X_valid = pd.read_pickle('tmp/X_valid.pickle')
X_test = pd.read_pickle('tmp/X_test.pickle')
X_train_2 = pd.read_pickle('tmp/X_train_2.pickle')

no_features = ['id', 'target', 'ts', 'guid', 'deviceid', 'newsid', 'timestamp', 'ID', 'fold'] + \
              ['id', 'target', 'timestamp', 'ts', 'isTest', 'day',
               'lat_mode', 'lng_mode', 'abtarget', 'applist_key',
               'applist_weight', 'tag_key', 'tag_weight', 'outertag_key', 'tag', 'outertag',
               'outertag_weight', 'newsid', 'datetime']
features = [fea for fea in X_train.columns if fea not in no_features]

end_time = time.time()
print("生成特征耗时：", end_time - start_time)


def load_data():
    return X_train, X_train_2, X_valid, X_test, no_features, features
