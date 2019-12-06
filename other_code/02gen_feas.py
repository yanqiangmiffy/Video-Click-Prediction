import pandas_profiling as pdf
import seaborn as sns
import pandas as pd
import numpy as np
import os
import pickle
from hyperopt import hp, fmin, rand, tpe, space_eval
from scipy.stats import norm, skew
import matplotlib.pyplot as plt
import tensorflow as tf
import lightgbm as lgb
import warnings
# import autopep8
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, auc, accuracy_score, log_loss, f1_score, precision_score, recall_score
from sklearn import preprocessing

from keras.layers import TimeDistributed, Bidirectional, BatchNormalization
from keras.layers import Dropout, Dense, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.models import Sequential
from keras.datasets import mnist
from keras import backend as K
from keras_radam import RAdam

import deepctr
from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat

from collections import namedtuple
from bayes_opt import BayesianOptimization
from scipy import stats
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'
import datetime
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['sans-serif']
color = sns.color_palette()
sns.set_style('darkgrid')

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 50)

# path = os.getcwd()
path = '.'
path_sub = path + '/result/'
path_npy = path + '/features/'
path_model = path + '/tmp/'
path_pickle = path + '/tmp/'


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def csv2pkl_all():
    with open(path_pickle + 'train.pickle', 'wb') as f:
        train_df = pd.read_csv('data/train.csv')
        pickle.dump(train_df, f)

    with open(path_pickle + 'test.pickle', 'wb') as f:
        test_df = pd.read_csv('data/test.csv')
        pickle.dump(test_df, f)

    with open(path_pickle + 'app.pickle', 'wb') as f:
        app_df = pd.read_csv('data/app.csv')
        pickle.dump(app_df, f)

    with open(path_pickle + 'user.pickle', 'wb') as f:
        user_df = pd.read_csv('data/user.csv')
        pickle.dump(user_df, f)


train = pd.read_pickle(path_pickle + 'train.pickle')[:10000]
test = pd.read_pickle(path_pickle + 'test.pickle')[:10000]

train.loc[(train[['timestamp', 'deviceid', 'newsid']].duplicated()) & (-train['timestamp'].isna()), 'target'] = 0
test['isTest'] = 1

print("train.shape",train.shape)
print("test.shape",test.shape)

def csv2pkl_data():
    def split_key(key_ans):
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        # print(list(map(lambda x: key2index[x], key_ans)))
        return list(map(lambda x: key2index[x], key_ans))

    print("提取特征....")
    data = pd.concat([train, test], sort=False)
    data = data.sort_values(['deviceid', 'ts']).reset_index().drop('index', axis=1)

    data['day'] = data['ts'].apply(
        lambda x: datetime.datetime.utcfromtimestamp(x // 1000).day)
    data['hour'] = data['ts'].apply(
        lambda x: datetime.datetime.utcfromtimestamp(x // 1000).hour)
    data['minute'] = data['ts'].apply(
        lambda x: datetime.datetime.utcfromtimestamp(x // 1000).minute)
    data['second'] = data['ts'].apply(
        lambda x: datetime.datetime.utcfromtimestamp(x // 1000).second)
    data['time1'] = np.int64(data['hour']) * 60 + np.int64(data['minute'])

    data.loc[~data['newsid'].isna(), 'isLog'] = 1
    data.loc[data['newsid'].isna(), 'isLog'] = 0

    # 把相隔广告曝光相隔时间较短的数据视为同一个事件，这里暂取间隔为3min
    # rank按时间排序同一个事件中每条数据发生的前后关系
    group = data.groupby('deviceid')
    data['gap_before'] = group['ts'].shift(0) - group['ts'].shift(1)
    data['gap_before'] = data['gap_before'].fillna(3 * 60 * 1000)
    INDEX = data[data['gap_before'] > (3 * 60 * 1000 - 1)].index
    data['gap_before'] = np.log(data['gap_before'] // 1000 + 1)
    data['gap_before_int'] = np.rint(data['gap_before'])
    LENGTH = len(INDEX)
    ts_group = []
    ts_len = []
    for i in tqdm(range(1, LENGTH)):
        ts_group += [i - 1] * (INDEX[i] - INDEX[i - 1])
        ts_len += [(INDEX[i] - INDEX[i - 1])] * (INDEX[i] - INDEX[i - 1])
    ts_group += [LENGTH - 1] * (len(data) - INDEX[LENGTH - 1])
    ts_len += [(len(data) - INDEX[LENGTH - 1])] * (len(data) - INDEX[LENGTH - 1])
    data['ts_before_group'] = ts_group
    data['ts_before_len'] = ts_len
    data['ts_before_rank'] = group['ts'].apply(lambda x: (x).rank())
    data['ts_before_rank'] = (data['ts_before_rank'] - 1) / \
                             (data['ts_before_len'] - 1)
    del ts_group
    group = data.groupby('deviceid')
    data['gap_after'] = group['ts'].shift(-1) - group['ts'].shift(0)
    data['gap_after'] = data['gap_after'].fillna(3 * 60 * 1000)
    INDEX = data[data['gap_after'] > (3 * 60 * 1000 - 1)].index
    data['gap_after'] = np.log(data['gap_after'] // 1000 + 1)
    data['gap_after_int'] = np.rint(data['gap_after'])
    LENGTH = len(INDEX)
    ts_group = [0] * (INDEX[0] + 1)
    ts_len = [INDEX[0]] * (INDEX[0] + 1)
    for i in tqdm(range(1, LENGTH)):
        ts_group += [i] * (INDEX[i] - INDEX[i - 1])
        ts_len += [(INDEX[i] - INDEX[i - 1])] * (INDEX[i] - INDEX[i - 1])
    data['ts_after_group'] = ts_group
    data['ts_after_len'] = ts_len
    data['ts_after_rank'] = group['ts'].apply(lambda x: (-x).rank())
    data['ts_after_rank'] = (data['ts_after_rank'] - 1) / (data['ts_after_len'] - 1)
    del group, ts_group

    data.loc[data['ts_before_rank'] == np.inf, 'ts_before_rank'] = 0
    data.loc[data['ts_after_rank'] == np.inf, 'ts_after_rank'] = 0
    data['ts_before_len'] = np.log(data['ts_before_len'] + 1)
    data['ts_after_len'] = np.log(data['ts_after_len'] + 1)

    # 'deviceid'不唯一
    app = pd.read_pickle(path_pickle + 'app.pickle')
    app['applist'] = app['applist'].apply(lambda x: x[1:-2])
    group = app.groupby('deviceid')
    del app

    gps = group['applist'].apply(lambda x: list(set(' '.join(x).split(' '))))
    del group
    gps = pd.DataFrame(gps)
    key2index = {}

    gps['applist_key'] = list(map(split_key, gps['applist']))
    gps['applist_len'] = gps['applist'].apply(lambda x: len(x))
    gps['applist_weight'] = gps['applist_len'].apply(lambda x: x * [1])
    gps.drop('applist', axis=1, inplace=True)
    print(len(key2index))
    data = pd.merge(data, gps, on=['deviceid'], how='left')
    # del key2index, gps

    #  ['deviceid', 'guid']唯一， 'deviceid'不唯一
    user = pd.read_pickle(path_pickle + 'user.pickle')
    for i in ['tag', 'outertag']:
        user.loc[user['%s' % i].isna() == False, '%s_weight' % i] = user.loc[user['%s' % i].isna() == False, '%s' %
                                                                             i].apply(
            lambda x: [np.float16(i.split(':')[1]) if len(i.split(':')) == 2 else 0 for i in x.split('|')])
        user.loc[user['%s' % i].isna() == False, '%s_key' % i] = user.loc[user['%s' % i].isna(
        ) == False, '%s' % i].apply(lambda x: [i.split(':')[0] for i in x.split('|')])
        user.loc[user['%s_weight' % i].isna() == False, '%s_len' % i] = user.loc[user['%s_weight' %
                                                                                      i].isna() == False, '%s_weight' % i].apply(
            lambda x: len(x))
        key2index = {}
        user.loc[user['%s_key' % i].isna() == False, '%s_key' % i] = list(
            map(split_key, user.loc[user['%s_key' % i].isna() == False, '%s_key' % i]))
        user.drop(i, axis=1, inplace=True)
        print(len(key2index))
    user['guid'].fillna('', inplace=True)
    data['guid'].fillna('', inplace=True)
    data = pd.merge(data, user, on=['deviceid', 'guid'], how='left')
    del user

    min_time = data['ts'].min()
    data['timestamp'] -= min_time
    data['ts'] -= min_time
    data['lat_int'] = np.int64(np.rint(data['lat'] * 100))
    data['lng_int'] = np.int64(np.rint(data['lng'] * 100))
    data.loc[data['level'].isna() == False, 'level_int'] = np.int64(
        data.loc[data['level'].isna() == False, 'level'])
    group = data[['deviceid', 'lat', 'lng']].groupby('deviceid')
    gp = group[['lat', 'lng']].agg(lambda x: stats.mode(x)[0][0]).reset_index()
    gp.columns = ['deviceid', 'lat_mode', 'lng_mode']
    data = pd.merge(data, gp, on='deviceid', how='left')
    del group, gp
    data['dist'] = np.log((data['lat'] - data['lat_mode']) **
                          2 + (data['lng'] - data['lng_mode']) ** 2 + 1)
    data['dist_int'] = np.rint(data['dist'])
    data.loc[data['lat'] != data['lat_mode'], 'isLatSame'] = 0
    data.loc[data['lat'] == data['lat_mode'], 'isLatSame'] = 1
    data.loc[data['lng'] != data['lng_mode'], 'isLngSame'] = 0
    data.loc[data['lng'] == data['lng_mode'], 'isLngSame'] = 1

    data.loc[data['personalscore'].isna(), 'personalscore'] = data['personalscore'].mode()

    data = reduce_mem_usage(data)
    data.to_pickle(path_pickle + 'data.pickle')
    data = pd.read_pickle(path_pickle + 'data.pickle')

    cate_cols = ['deviceid', 'guid', 'pos', 'app_version',
                 'device_vendor', 'netmodel', 'osversion',
                 'device_version', 'hour', 'minute', 'second',
                 'personalscore', 'gender', 'level_int', 'dist_int',
                 'lat_int', 'lng_int', 'gap_before_int', 'ts_before_group',
                 'time1', 'gap_after_int', 'ts_after_group',
                 'personidentification']
    drop_cols = ['id', 'target', 'timestamp', 'ts', 'isTest', 'day',
                 'lat_mode', 'lng_mode', 'abtarget', 'applist_key',
                 'applist_weight', 'tag_key', 'tag_weight', 'outertag_key',
                 'outertag_weight', 'newsid']

    fillna_cols = ['outertag_len', 'tag_len', 'lng', 'lat', 'level',
                   'followscore', 'dist', 'applist_len', 'ts_before_rank',
                   'ts_after_rank']
    data = reduce_mem_usage(data)
    data.to_pickle(path_pickle + 'data.pickle')


csv2pkl_data()
data = pd.read_pickle(path_pickle + 'data.pickle')
print("data.shape",data.shape)
no_features = ['id', 'target', 'timestamp', 'ts', 'isTest', 'day',
               'lat_mode', 'lng_mode', 'abtarget', 'applist_key',
               'applist_weight', 'tag_key', 'tag_weight', 'outertag_key',
               'outertag_weight', 'newsid']
features = [fea for fea in data.columns if fea not in no_features]
data = data.fillna(0)

final_train, final_test = data[data['isTest'] != 1], data[data['isTest'] == 1]
data.to_csv('tmp/df.csv', index=None)
print("final_train.shape",final_train.shape)
print("final_test.shape",final_train.shape)
print(features)
print(data[features].shape[0]-data[features].count())
print(final_train[features].shape[0]-final_train[features].count())
print(final_test[features].shape[0]-final_test[features].count())
del data

def load_data():
    return final_train, final_test, no_features, features
