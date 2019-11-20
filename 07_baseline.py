#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 07_baseline.py 
@time: 2019-11-19 19:21
@description:
"""
import gc
import math
import json
import time
import numpy as np
import pandas as pd
import multiprocessing
import lightgbm as lgb
from tqdm import tqdm
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import warnings
import os
import datetime

warnings.filterwarnings('ignore')

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
data = train.append(test).reset_index(drop=True)
def get_time_str(x):
    dateArray = datetime.datetime.utcfromtimestamp(x)
    otherStyleTime = dateArray.strftime('%Y-%m-%d %H:%M:%S')
    return otherStyleTime
data['ts'] = data['ts'].apply(lambda x:get_time_str(x/1000))
data['ts'] = pd.to_datetime(data['ts'])

def preprocess(df):
    df["hour"] = df["ts"].dt.hour
    #     df["day"] = df["timestamp"].dt.day
    # df["weekend"] = df["ts"].dt.weekday
    # df["month"] = df["ts"].dt.month
    df["dayofweek"] = df["ts"].dt.dayofweek
preprocess(data)
# def get_combination_fea(df):
#     """
#     添加组合特征
#     :return:
#     """
#     print('添加组合特征...')
#     combination_cols = []
#     df['netmodel_device_vendor'] =( df['netmodel'].astype(str) + df['device_vendor'].astype(str)).astype('category')
#     df['netmodel_pos'] = (df['netmodel'].astype(str) + df['pos'].astype(str)).astype('category')
#     df['dayofweek_hour'] = (df['dayofweek'].astype(str) + df['hour'].astype(str)).astype('category')
#     df['netmodel_hour'] = (df['netmodel'].astype(str) + df['hour'].astype(str)).astype('category')
#     df['netmodel_dayofweek'] = (df['netmodel'].astype(str) + df['dayofweek'].astype(str)).astype('category')
#
#     combination_cols.extend([
#         # 'deviceid_newsid', 'guid_newsid',
#         # 'pos_newsid', 'device_vendor_newsid',
#         'netmodel_device_vendor',
#         'netmodel_pos', 'dayofweek_hour',
#         'netmodel_hour', 'netmodel_dayofweek'
#     ])
#
#     for col in combination_cols:
#         print(col)
#         df['{}_count'.format(col)] = df.groupby(col)['id'].transform('count')
#         del df[col]
#         gc.collect()
#     return df


# 提取app个数特征
app = pd.read_csv("data/app.csv")
app['applist'] = app['applist'].apply(lambda x: str(x)[1:-2])
app['applist'] = app['applist'].apply(lambda x: str(x).replace(' ', '|'))
app = app.groupby('deviceid')['applist'].apply(lambda x: '|'.join(x)).reset_index()
app['app_len'] = app['applist'].apply(lambda x: len(x.split('|')))
data = data.merge(app[['deviceid', 'app_len']], how='left', on='deviceid')

# data=get_combination_fea(data)

del app

user = pd.read_csv("data/user.csv")
user = user.drop_duplicates('deviceid')
data = data.merge(user[['deviceid', 'level', 'personidentification', 'followscore', 'personalscore', 'gender']],
                  how='left', on='deviceid')

print("get_tag_fea....")
user['outertag'] = user['outertag'].astype(str)
grouped_df = user.groupby('deviceid').agg({'outertag': '|'.join})
grouped_df.columns = ['deviceid_' + 'outertag']

# 最受欢迎的50个outertag
all_outertag = {}
for x in user.outertag.astype(str):
    tags = x.split('|')
    if tags[0] != 'nan':
        for tag in tags:
            tmp = tag.split(':')
            if len(tmp) == 2:
                if tmp[0] in all_outertag:
                    all_outertag[tmp[0]] += float(tmp[1])
                else:
                    all_outertag[tmp[0]] = 0
                    all_outertag[tmp[0]] += float(tmp[1])
top_outertag = {}
for tag, score in sorted(all_outertag.items(), key=lambda item: item[1], reverse=True)[:10]:
    top_outertag[tag] = score

for tag in top_outertag:
    grouped_df[tag] = grouped_df['deviceid_outertag'].apply(lambda x: top_outertag[tag] if tag in x else 0)

del top_outertag, all_outertag
gc.collect()

# 最受欢迎的100个tag
all_tag = {}
for x in user.tag.astype(str):
    tags = x.split('|')
    if tags[0] != 'nan':
        for tag in tags:
            tmp = tag.split(':')
            if len(tmp) == 2:
                if tmp[0] in all_tag:
                    all_tag[tmp[0]] += float(tmp[1])
                else:
                    all_tag[tmp[0]] = 0
                    all_tag[tmp[0]] += float(tmp[1])
top_tag = {}
for tag, score in sorted(all_tag.items(), key=lambda item: item[1], reverse=True)[:20]:
    top_tag[tag] = score

for tag in top_tag:
    grouped_df[tag] = grouped_df['deviceid_outertag'].apply(lambda x: top_tag[tag] if tag in x else 0)
del grouped_df['deviceid_outertag']
del top_tag, all_tag
gc.collect()

data = data.merge(grouped_df,
                  how='left', on='deviceid')

del user

# 类别特征count特征
cat_list = [i for i in train.columns if i not in ['id', 'lat', 'lng', 'target', 'timestamp', 'ts']] + ['level']
for i in tqdm(cat_list):
    data['{}_count'.format(i)] = data.groupby(['{}'.format(i)])['id'].transform('count')

# 类别特征五折转化率特征
data['ID'] = data.index
data['fold'] = data['ID'] % 5
data.loc[data.target.isnull(), 'fold'] = 5
target_feat = []
for i in tqdm(cat_list):
    target_feat.extend([i + '_mean_last_1'])
    data[i + '_mean_last_1'] = None
    for fold in range(6):
        data.loc[data['fold'] == fold, i + '_mean_last_1'] = data[data['fold'] == fold][i].map(
            data[(data['fold'] != fold) & (data['fold'] != 5)].groupby(i)['target'].mean()
        )
    data[i + '_mean_last_1'] = data[i + '_mean_last_1'].astype(float)

# 对object类型特征进行编码
lbl = LabelEncoder()
object_col = [i for i in data.select_dtypes(object).columns if i not in ['id']]
for i in tqdm(object_col):
    data[i] = lbl.fit_transform(data[i].astype(str))

feature_name = [i for i in data.columns if i not in ['id', 'target', 'ts','ID', 'fold', 'timestamp']]
tr_index = ~data['target'].isnull()
X_train = data[tr_index].reset_index(drop=True)[feature_name].reset_index(drop=True).values
y = data[tr_index]['target'].reset_index(drop=True).values
X_test = data.loc[data['id'].isin(test['id'].unique())][feature_name].reset_index(drop=True).values

lgb_param = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'num_leaves': 1000,
    'verbose': -1,
    'max_depth': -1,
    'seed': 2019,
    'n_jobs': -1,
    # 'device': 'gpu',
    # 'gpu_device_id': 0,
}


def eval_func(y_pred, train_data):
    y_true = train_data.get_label()
    score = f1_score(y_true, np.round(y_pred))
    return 'f1', score, True


# print(X_train.shape, X_test[feature_name].shape)
oof = np.zeros(X_train.shape[0])
prediction = np.zeros(X_test.shape[0])
seeds = [19970412, 1, 4096, 2048, 1024]
num_model_seed = 1
for model_seed in range(num_model_seed):
    oof_lgb = np.zeros(X_train.shape[0])
    prediction_lgb = np.zeros(X_test.shape[0])
    skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=False)
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
        print(index)
        train_x, test_x, train_y, test_y = X_train[train_index], X_train[test_index], y[train_index], \
                                           y[test_index]
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_valid = lgb.Dataset(test_x, test_y, reference=lgb_train)
        lgb_model = lgb.train(lgb_param, lgb_train, num_boost_round=40000, valid_sets=[lgb_valid],
                              valid_names=['valid'], early_stopping_rounds=50, feval=eval_func,
                              verbose_eval=10)

        oof_lgb[test_index] += lgb_model.predict(test_x)
        prediction_lgb += lgb_model.predict(X_test) / 5

        del lgb_model
        del train_x, test_x, train_y, test_y
        del lgb_train,lgb_valid

    print('AUC', roc_auc_score(y, oof_lgb))
    print(prediction_lgb.mean())
    oof += oof_lgb / num_model_seed
    prediction += prediction_lgb / num_model_seed
print('AUC', roc_auc_score(y, oof))
print('f1', f1_score(y, np.round(oof)))

# 生成文件，因为类别不平衡预测出来概率值都很小，根据训练集正负样本比例，来确定target
# 0.8934642948637943为训练集中0样本的比例，阈值可以进一步调整
submit = test[['id']]
submit['target'] = prediction
submit.to_csv('result/lgb_prob')
submit['target'] = submit['target'].rank()
submit['target'] = (submit['target'] >= submit.shape[0] * 0.8934642948637943).astype(int)
submit.to_csv("result/lgb_result.csv", index=False)
