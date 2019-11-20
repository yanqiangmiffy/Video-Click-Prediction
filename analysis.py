#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: analysis.py 
@time: 2019-11-21 01:42
@description:
"""
import pandas as pd
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

start_time = time.time()


def get_time_str(x):
    dateArray = datetime.datetime.utcfromtimestamp(x)
    otherStyleTime = dateArray.strftime('%Y-%m-%d %H:%M:%S')
    return otherStyleTime


train=pd.read_csv('data/train.csv')
# 将时间戳转为datetime
train['ts'] = train['ts'].apply(lambda x: get_time_str(x / 1000))
train['ts'] = pd.to_datetime(train['ts'])
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

preprocess_ts(train)
train0=train[train['target']==0][:100000]
train1=train[train['target']==1]
train0.to_csv('tmp/train0.csv',index=None)
train1.to_csv('tmp/train1.csv',index=None)