#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: lgb.py 
@time: 2019-11-16 10:40
@description:
"""
import gc
import os
import random
import sys
import time

from tqdm import tqdm_notebook as tqdm
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py

# py.init_notebook_mode(connected=False)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import *

from utils import *

sample_submission = pd.read_feather('data/sample_submission.feather')


def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True


from gen_feas import load_data

train, test, no_features, features = load_data()
print(features)
X = train[features].values
y = train['target'].astype('int32')
test_data = test[features].values
print(X.shape)
# %%

# 训练
# 采取分层采样
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sklearn.metrics import roc_auc_score

print("start：********************************")
start = time.time()

N = 5
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2018)

auc_cv = []
y_pred_all_l1 = np.zeros(test.shape[0])

for k, (train_in, test_in) in enumerate(skf.split(X, y)):
    X_train, X_valid, y_train, y_valid = X[train_in], X[test_in], \
                                         y[train_in], y[test_in]

    # 数据结构
    lgb_train = lgb.Dataset(X_train, y_train, params={'verbose': -1})
    lgb_eval = lgb.Dataset(X_valid, y_valid, params={'verbose': -1}, reference=lgb_train)

    # 设置参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'verbose': -1,
        "nthread": -1
        # 'lambda_l1':0.25,
        # 'lambda_l2':0.5,
        # 'scale_pos_weight':10.0/1.0, #14309.0 / 691.0, #不设置
        # 'num_threads':4,
    }
    print('................Start training..........................')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    valid_sets=(lgb_train, lgb_eval),
                    early_stopping_rounds=100,
                    verbose_eval=500,
                    feval=evalerror,
                    feature_name=features,

                    )

    print('................Start predict .........................')
    # 预测
    y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    # 评估
    tmp_auc = roc_auc_score(y_valid, y_pred)
    auc_cv.append(tmp_auc)
    print("f1_score", tmp_auc)

    # test
    y_pred_all_l1 += gbm.predict(test_data, num_iteration=gbm.best_iteration)

    del gbm,y_pred
    gc.collect()

# K交叉验证的平均分数
print('the cv information:')
print('cv f1 mean score', np.mean(auc_cv))

end = time.time()
print("......................run with time: ", (end - start) / 60.0)
print("over:*********************************")

r = y_pred_all_l1 / skf.n_splits
sample_submission['target'] = r
sample_submission.to_csv('result/lgb_prob.csv', index=False, sep=",")

sample_submission['target'] = [1 if x > 0.50 else 0 for x in r]
print(sample_submission['target'].value_counts())
sample_submission.to_csv('result/lgb_result.csv', index=False)
