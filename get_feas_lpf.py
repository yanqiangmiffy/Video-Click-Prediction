# coding:utf-8
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import copy
import lightgbm as lgb
from tqdm import tqdm
import os
from datetime import timedelta
from sklearn.feature_selection import chi2, SelectPercentile
from matplotlib import pyplot as plt
import time
import gc
pd.set_option('display.max_columns', None)


scaler = StandardScaler()

def get_fea(train, test, user, app):
    test['target'] = 'test'

    user['mark'] = 0
    user['deviceid_count'] = user.groupby('deviceid').mark.transform('count')
    user['guid_count'] = user.groupby('guid').mark.transform('count')

    def get_same_tag(x, y):
        x = str(x)
        y = str(y)

        if '|' not in x or '|' not in y:
            return 0

        x = x.split('|')
        x = [i.split(':')[0] for i in x]

        y = y.split('|')
        y = [i.split(':')[0] for i in y]

        return len(set(x).intersection(set(y)))

    user['tag_same'] = user.apply(lambda x: get_same_tag(x['outertag'], x['tag']), axis=1)

    df = pd.concat((train, test))
    user_duplicate = user.drop_duplicates(subset=['deviceid', 'guid'])
    app_duplicate = app.drop_duplicates(subset=['deviceid'])
    df = pd.merge(df, user_duplicate, on=['deviceid', 'guid'], how='left')
    df = pd.merge(df, app_duplicate, on=['deviceid'], how='left')
    print(df[df['tag_same'] != 0][['outertag', 'tag', 'tag_same']])

    df['ts'] = df['ts'].apply(lambda x: datetime.fromtimestamp(x / 1000))
    df['ts_day'] = df['ts'].apply(lambda x: x.day)
    df['ts_hour'] = df['ts'].apply(lambda x: x.hour)

    no_features = ['id', 'target', 'timestamp']

    lb_feas = ['app_version', 'device_vendor', 'device_version', 'deviceid', 'guid', 'netmodel', 'newsid', 'osversion',
               'timestamp', 'outertag', 'tag', 'applist', 'ts']

    for fea in lb_feas:
        print(fea)
        df[fea].fillna('w', inplace=True)
        df[fea] = df[fea].astype(str)
        df[fea] = LabelEncoder().fit_transform(df[fea])


    train = df[df['target'] != 'test']
    test = df[df['target'] == 'test']


    train = train[train['target'].notnull()].reset_index(drop=True)
    print(train.shape)
    return train, test, no_features

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
user = pd.read_csv('data/user.csv')
app = pd.read_csv('data/app.csv')
sample = pd.read_csv('data/sample.csv')

train, test, no_features = get_fea(train, test, user, app)
print('get_fea ok')

label = train['target']
sub = test[['id']]

features = [fea for fea in train.columns if fea not in no_features]

# train_df = train[features]
#
# test_df = test[features]


def load_data():
    return train, test, no_features, features

