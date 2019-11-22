#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 08_deepctr.py 
@time: 2019-11-22 07:55
@description:
"""

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
# """ 运行DeepFM """

path = 'data/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# 测试
train = train[0:10000]
test = test[0:10000]

# print(train.head())
data = pd.concat([train, test], ignore_index=True, sort=False)
# print(data.head()) timestamp：代表改用户点击改视频的时间戳，如果未点击则为NULL。 deviceid：用户的设备id。 newsid：视频的id。 guid：用户的注册id。 pos：视频推荐位置。
# app_version：app版本。 device_vendor：设备厂商。 netmodel：网络类型。 osversion：操作系统版本。 lng：经度。 lat：维度。 device_version：设备版本。
# ts：视频暴光给用户的时间戳。 id,target,timestamp,deviceid,newsid,guid,pos,app_version,device_vendor,netmodel,osversion,lng,lat,
# device_version,ts id,deviceid,newsid,guid,pos,app_version,device_vendor,netmodel,osversion,lng,lat,device_version,
# ts 单值类别特征
fix_len_category_columns = ['app_version', 'device_vendor', 'netmodel', 'osversion', 'device_version']
# 数值特征
fix_len_number_columns = ['timestamp', 'lng', 'lat', 'ts']

target = ['target']

data[fix_len_category_columns] = data[fix_len_category_columns].fillna('-1', )
data[fix_len_number_columns] = data[fix_len_number_columns].fillna(0, )

for feat in fix_len_category_columns:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

mms = MinMaxScaler(feature_range=(0, 1))
data[fix_len_number_columns] = mms.fit_transform(data[fix_len_number_columns])

fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                          for feat in fix_len_category_columns] + [DenseFeat(feat, 1, ) for feat in
                                                                   fix_len_number_columns]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

train = data[~data['target'].isnull()]
X = {name: train[name] for name in feature_names}.values
y = train['target'].values

test = data[data['target'].isnull()]
test_model_input = {name: test[name] for name in feature_names}

y_pred_final = np.zeros(len(test))

skf = StratifiedKFold(n_splits=5, random_state=1314, shuffle=True)
for k, train_index, valid_index in skf.split(X, y):
    print("n。{}_th fold".format(k))
    X_train, X_valid, y_train, y_valid = X[train_index], X[valid_index], y[train_index], y[valid_index]

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )
    model.fit(X_train, y_train,
              validation_data=[X_valid, y_valid],
              verbose=1,
              batch_size=256,
              epochs=10,
              validation_split=0.2
              )

    pred_ans = model.predict(test_model_input, 8192)
    y_pred_final += pred_ans.reshape(pred_ans.shape[0])

test['target'] = y_pred_final/skf.n_splits
test[['id', 'target']].to_csv('result/deepctr.csv', index=None)
