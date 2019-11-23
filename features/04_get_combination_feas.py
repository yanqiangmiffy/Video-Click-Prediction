#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 04_get_combination_feas.py 
@time: 2019-11-23 00:54
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
from utils import *


def get_time_str(x):
    dateArray = datetime.datetime.utcfromtimestamp(x)
    otherStyleTime = dateArray.strftime('%Y-%m-%d %H:%M:%S')
    return otherStyleTime


def preprocess_ts(df):
    """
    时间特征
    :param df:
    :return:
    """
    df["hour"] = df["ts"].dt.hour
    df["day"] = df["ts"].dt.day
    # df["weekend"] = df["ts"].dt.weekday
    # df["month"] = df["ts"].dt.month
    df["dayofweek"] = df["ts"].dt.dayofweek


# 加载数据
root = Path('./data/')
train_df = pd.read_csv(root / 'train.csv')
train_df['target'] = train_df['target'].astype(int)
test_df = pd.read_csv(root / 'test.csv')
test_df['target'] = 0

# 将时间戳转为datetime
train_df['ts'] = train_df['ts'].apply(lambda x: get_time_str(x / 1000))
test_df['ts'] = test_df['ts'].apply(lambda x: get_time_str(x / 1000))
train_df['ts'] = pd.to_datetime(train_df['ts'])
test_df['ts'] = pd.to_datetime(test_df['ts'])

df = pd.concat([train_df, test_df], sort=False, axis=0)
preprocess_ts(df)

print('添加组合特征...')
combination_cols = []
df['deviceid_newsid'] = (df['deviceid'].astype(str) + df['newsid'].astype(str)).astype('category')
df['guid_newsid'] = (df['guid'].astype(str) + df['newsid'].astype(str)).astype('category')
df['pos_newsid'] = (df['pos'].astype(str) + df['newsid'].astype(str)).astype('category')
df['device_vendor_newsid'] = (df['device_vendor'].astype(str) + df['newsid'].astype(str)).astype('category')
df['lng_newsid'] = (df['lng'].astype(str) + df['newsid'].astype(str)).astype('category')
df['hour_newsid'] = (df['hour'].astype(str) + df['newsid'].astype(str)).astype('category')
df['dayofweek_newsid'] = (df['dayofweek'].astype(str) + df['newsid'].astype(str)).astype('category')

df['dayofweek_hour'] = (df['dayofweek'].astype(str) + df['hour'].astype(str)).astype('category')

df['netmodel_hour'] = (df['netmodel'].astype(str) + df['hour'].astype(str)).astype('category')
df['netmodel_dayofweek'] = (df['netmodel'].astype(str) + df['dayofweek'].astype(str)).astype('category')

combination_cols.extend([
    'deviceid_newsid', 'guid_newsid',
    'pos_newsid', 'device_vendor_newsid',
    'lng_newsid', 'hour_newsid',
    'dayofweek_newsid', 'dayofweek_hour',
    'netmodel_hour', 'netmodel_dayofweek'
])

for col in combination_cols:
    print(col)
    df['{}_count'.format(col)] = df.groupby(col)['id'].transform('count')
    del df[col]
    gc.collect()
