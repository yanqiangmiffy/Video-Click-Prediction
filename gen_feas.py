#!/usr/bin/env python
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

df = pd.read_pickle('data/tmp.pickle')
# 构造历史特征 分别统计前一天 guid deviceid 的相关信息
# 8 9 10 11
history_9 = df[df['day'] == 8]
history_10 = df[df['day'] == 9]
history_11 = df[df['day'] == 10]
history_12 = df[df['day'] == 11]
del df
gc.collect()
# 61326
# 64766
# 66547
# 41933
# 42546
print(len(set(history_9['deviceid'])))
print(len(set(history_10['deviceid'])))
print(len(set(history_11['deviceid'])))
print(len(set(history_12['deviceid'])))
print(len(set(history_9['deviceid']) & set(history_10['deviceid'])))
print(len(set(history_10['deviceid']) & set(history_11['deviceid'])))
print(len(set(history_11['deviceid']) & set(history_12['deviceid'])))

# 61277
# 64284
# 66286
# 41796
# 42347

print(len(set(history_9['guid'])))
print(len(set(history_10['guid'])))
print(len(set(history_11['guid'])))
print(len(set(history_12['guid'])))
print(len(set(history_9['guid']) & set(history_10['guid'])))
print(len(set(history_10['guid']) & set(history_11['guid'])))
print(len(set(history_11['guid']) & set(history_12['guid'])))

# 640066
# 631547
# 658787
# 345742
# 350542

print(len(set(history_9['newsid'])))
print(len(set(history_10['newsid'])))
print(len(set(history_11['newsid'])))
print(len(set(history_12['newsid'])))
print(len(set(history_9['newsid']) & set(history_10['newsid'])))
print(len(set(history_10['newsid']) & set(history_11['newsid'])))
print(len(set(history_11['newsid']) & set(history_12['newsid'])))


# deviceid guid timestamp ts 时间特征
def get_history_visit_time(data1, date2):
    data1 = data1.sort_values(['ts', 'timestamp'])
    data1['timestamp_ts'] = data1['timestamp'] - data1['ts']
    data1_tmp = data1[data1['target'] == 1].copy()
    del data1
    for col in ['deviceid', 'guid']:
        for ts in ['timestamp_ts']:
            f_tmp = data1_tmp.groupby([col], as_index=False)[ts].agg({
                '{}_{}_max'.format(col, ts): 'max',
                '{}_{}_mean'.format(col, ts): 'mean',
                '{}_{}_min'.format(col, ts): 'min',
                '{}_{}_median'.format(col, ts): 'median'
            })
        date2 = pd.merge(date2, f_tmp, on=[col], how='left', copy=False)

    return date2


history_10 = get_history_visit_time(history_9, history_10)
history_11 = get_history_visit_time(history_10, history_11)
history_12 = get_history_visit_time(history_11, history_12)

data = pd.concat([history_10, history_11], axis=0, sort=False, ignore_index=True)
data = pd.concat([data, history_12], axis=0, sort=False, ignore_index=True)
del history_9, history_10, history_11, history_12

data = data.sort_values('ts')
data['ts_next'] = data.groupby(['deviceid'])['ts'].shift(-1)
data['ts_next_ts'] = data['ts_next'] - data['ts']

# 当前一天内的特征 leak
for col in [['deviceid'], ['guid'], ['newsid']]:
    print(col)
    data['{}_days_count'.format('_'.join(col))] = data.groupby(['day'] + col)['id'].transform('count')

print('train and predict')
X_train = data[data['flag'].isin([9])]
X_valid = data[data['flag'].isin([10])]
X_test = data[data['flag'].isin([11])]
X_train_2 = data[data['flag'].isin([9, 10])]

X_train = reduce_mem_usage(X_train)
X_valid = reduce_mem_usage(X_valid)
X_test = reduce_mem_usage(X_test)
X_train_2 = reduce_mem_usage(X_train_2)

X_train.to_pickle('tmp/X_train.pickle')
X_valid.to_pickle('tmp/X_valid.pickle')
X_test.to_pickle('tmp/X_test.pickle')
X_train_2.to_pickle('tmp/X_train_2.pickle')

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
