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
from tqdm import tqdm
from sklearn.preprocessing import *
from numpy import random
from pathlib import Path
import gc

# 辅助函数
def statics():
    stats = []
    for col in train_df.columns:
        stats.append((col, train_df[col].nunique(), train_df[col].isnull().sum() * 100 / train_df.shape[0],
                      train_df[col].value_counts(normalize=True, dropna=False).values[0] * 100, train_df[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                            'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('Unique_values', ascending=False, inplace=True)
    stats_df.to_excel('stats_train.xlsx', index=None)


# 加载数据
root = Path('./data/')
train_df = pd.read_feather(root / 'train.feather')[:2000]
train_df['target'] = train_df['target'].astype(int)
test_df = pd.read_feather(root / 'test.feather')
print(train_df.shape)
print(test_df.shape)

app_df = pd.read_feather(root / 'app.feather')
user_df = pd.read_feather(root / 'user.feather')


def preprocess(df):
    df["hour"] = df["ts"].dt.hour
    #     df["day"] = df["timestamp"].dt.day
    df["weekend"] = df["ts"].dt.weekday
    df["month"] = df["ts"].dt.month
    df["dayofweek"] = df["ts"].dt.dayofweek


df = pd.concat([train_df, test_df], sort=False, axis=0)
preprocess(df)

cate_cols = [ 'device_vendor', 'app_version', 'osversion', 'netmodel']
df=pd.get_dummies(df,columns=cate_cols)
for col in ['device_version']:
    lb = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = df[col].fillna('999')
    df[col] = lb.fit_transform(df[col])

no_features = ['id', 'target','ts','guid', 'deviceid', 'newsid','deviceid',]
features = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(train_df)], df[len(train_df):]
df.head(100).to_csv('tmp/df.csv', index=None)
print("df shape",df.shape)

del df
gc.collect()

print(features)
print(train['target'].value_counts())
print("train_df shape",train_df.shape)
print("test_df shape",test_df.shape)
def load_data():
    return train, test, no_features, features
