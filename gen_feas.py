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

root = Path('./data/')

train_df = pd.read_feather(root / 'train.feather')
test_df = pd.read_feather(root / 'test.feather')
print(train_df.shape)
print(test_df.shape)
app_df = pd.read_feather(root / 'app.feather')
user_df = pd.read_feather(root / 'user.feather')

df = pd.concat([train_df, test_df], sort=False, axis=0)

print(train_df.shape[0] - train_df.count())
print(test_df.shape[0] - test_df.count())
print(train_df['ts'].head(1))

stats = []
for col in train_df.columns:
    stats.append((col, train_df[col].nunique(), train_df[col].isnull().sum() * 100 / train_df.shape[0],
                  train_df[col].value_counts(normalize=True, dropna=False).values[0] * 100, train_df[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                        'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Unique_values', ascending=False, inplace=True)
stats_df.to_excel('stats_train.xlsx', index=None)

cate_cols=['deviceid','guid','device_version','device_vendor','app_version','osversion','netmodel']

for col in cate_cols:
    lb=LabelEncoder()
    df[col]=df[col].astype(str)
    df[col]=df[col].fillna('999')
    df[col]=lb.fit_transform(df[col])
# df['ts']=df['ts'].astype('float32')

train_df['date'] = train_df['ts'].dt.date
no_features = ['id', 'target','ts']

features = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(train_df)], df[len(train_df):]

print(train.target.value_counts())
def load_data():
    return train, test, no_features, features
