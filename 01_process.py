#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: quincy qiang
@license: Apache Licence
@file: 01_process.py
@time: 2019/12/08
@software: PyCharm
"""
import gc
import pickle
import pandas as pd
import numpy as np
import time


def csv2pkl():
    tmp = pd.read_csv('data/train.csv')
    tmp.to_pickle('data/train.pickle')

    tmp = pd.read_csv('data/test.csv')
    tmp.to_pickle('data/test.pickle')

    tmp = pd.read_csv('data/app.csv')
    tmp.to_pickle('data/app.pickle')

    tmp = pd.read_csv('data/user.csv')
    tmp.to_pickle('data/user.pickle')

    tmp = pd.read_csv('data/sample.csv')
    tmp.to_pickle('data/sample.pickle')


# csv2pkl()

df_train = pd.read_pickle('data/train.pickle')
df_test = pd.read_pickle('data/test.pickle')
df_app = pd.read_pickle('data/app.pickle')
df_user = pd.read_pickle('data/user.pickle')
df_sample = pd.read_pickle('data/sample.pickle')


# 时间格式转化 ts
def time_data2(time_sj):
    data_sj = time.localtime(time_sj / 1000)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", data_sj)
    return time_str


# 时间处理
df_train['datetime'] = df_train['ts'].apply(time_data2)
df_test['datetime'] = df_test['ts'].apply(time_data2)
df_train['datetime'] = pd.to_datetime(df_train['datetime'])
df_test['datetime'] = pd.to_datetime(df_test['datetime'])

df_train['day'] = df_train['datetime'].dt.day
df_test['day'] = df_test['datetime'].dt.day
df_train['flag'] = df_train['day']
df_test['flag'] = 11

df = pd.concat([df_train, df_test], axis=0, sort=False)
del df_train, df_test

df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
# 缺失值填充
df['guid'] = df['guid'].fillna('abc')