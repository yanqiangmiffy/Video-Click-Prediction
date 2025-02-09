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
    # test['target'] = 'test'
    test['is_test'] = 1

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

    app['applist'] = app['applist'].apply(lambda x: str(x)[1:-2])
    app['applist'] = app['applist'].apply(lambda x: str(x).replace(' ', '|'))
    app = app.groupby('deviceid')['applist'].apply(lambda x: '|'.join(x)).reset_index()
    app['app_len'] = app['applist'].apply(lambda x: len(x.split('|')))

    df = pd.concat((train, test)).reset_index(drop=True)
    user_duplicate = user.drop_duplicates(subset=['deviceid', 'guid'])
    app_duplicate = app.drop_duplicates(subset=['deviceid'])
    df = pd.merge(df, user_duplicate, on=['deviceid', 'guid'], how='left')
    df = pd.merge(df, app_duplicate, on=['deviceid'], how='left')

    cat_list = [i for i in df.columns if i not in ['id', 'lat', 'lng', 'target', 'timestamp', 'ts']] + ['level']


    # 排序 相减
    df = df.sort_values(by=['guid', 'ts'], ascending=False)
    df['ts_1'] = df['ts'].shift(1)
    df['ts_1'].fillna(1573142399626, inplace=True)
    df['ts_diff'] = df['ts'] - df['ts_1']
    df['ts_diff'] = df['ts_diff']/10000
    print(df[['ts', 'ts_1', 'ts_diff']])

    # # 把相隔广告曝光相隔时间较短的数据视为同一个事件，这里暂取间隔为3min
    # # rank按时间排序同一个事件中每条数据发生的前后关系
    # group = df.groupby('deviceid')
    # df['gap_before'] = group['ts'].shift(0) - group['ts'].shift(1)
    # df['gap_before'] = df['gap_before'].fillna(3 * 60 * 1000)
    # INDEX = df[df['gap_before'] > (3 * 60 * 1000 - 1)].index
    # df['gap_before'] = np.log(df['gap_before'] // 1000 + 1)
    # df['gap_before_int'] = np.rint(df['gap_before'])
    # LENGTH = len(INDEX)
    # ts_group = []
    # ts_len = []
    # for i in tqdm(range(1, LENGTH)):
    #     ts_group += [i - 1] * (INDEX[i] - INDEX[i - 1])
    #     ts_len += [(INDEX[i] - INDEX[i - 1])] * (INDEX[i] - INDEX[i - 1])
    # ts_group += [LENGTH - 1] * (len(df) - INDEX[LENGTH - 1])
    # ts_len += [(len(df) - INDEX[LENGTH - 1])] * (len(df) - INDEX[LENGTH - 1])
    # df['ts_before_group'] = ts_group
    # df['ts_before_len'] = ts_len
    # df['ts_before_rank'] = group['ts'].apply(lambda x: (x).rank())
    # df['ts_before_rank'] = (df['ts_before_rank'] - 1) / \
    #                          (df['ts_before_len'] - 1)
    #
    # group = df.groupby('deviceid')
    # df['gap_after'] = group['ts'].shift(-1) - group['ts'].shift(0)
    # df['gap_after'] = df['gap_after'].fillna(3 * 60 * 1000)
    # INDEX = df[df['gap_after'] > (3 * 60 * 1000 - 1)].index
    # df['gap_after'] = np.log(df['gap_after'] // 1000 + 1)
    # df['gap_after_int'] = np.rint(df['gap_after'])
    # LENGTH = len(INDEX)
    # ts_group = [0] * (INDEX[0] + 1)
    # ts_len = [INDEX[0]] * (INDEX[0] + 1)
    # for i in tqdm(range(1, LENGTH)):
    #     ts_group += [i] * (INDEX[i] - INDEX[i - 1])
    #     ts_len += [(INDEX[i] - INDEX[i - 1])] * (INDEX[i] - INDEX[i - 1])
    # df['ts_after_group'] = ts_group
    # df['ts_after_len'] = ts_len
    # df['ts_after_rank'] = group['ts'].apply(lambda x: (-x).rank())
    # df['ts_after_rank'] = (df['ts_after_rank'] - 1) / (df['ts_after_len'] - 1)
    #
    # df.loc[df['ts_before_rank'] == np.inf, 'ts_before_rank'] = 0
    # df.loc[df['ts_after_rank'] == np.inf, 'ts_after_rank'] = 0
    # df['ts_before_len'] = np.log(df['ts_before_len'] + 1)
    # df['ts_after_len'] = np.log(df['ts_after_len'] + 1)



    df['ts'] = df['ts'].apply(lambda x: datetime.fromtimestamp(x / 1000))
    df['ts_day'] = df['ts'].apply(lambda x: x.day)
    df['ts_hour'] = df['ts'].apply(lambda x: x.hour)


    df['minute'] = df['ts'].apply(lambda x: x.minute)
    df['time1'] = np.int64(df['ts_hour']) * 60 + np.int64(df['minute'])
    df.loc[~df['newsid'].isna(), 'isLog'] = 1
    df.loc[df['newsid'].isna(), 'isLog'] = 0

    # 类别特征count特征
    cat_list.append('ts_day')
    cat_list.append('ts_hour')

    no_features = ['id', 'target', 'timestamp', 'ID', 'fold', 'is_test', 'ts_1']

    print(cat_list)
    print(df[cat_list])
    for i in tqdm(cat_list):
        df['{}_count'.format(i)] = df.groupby(['{}'.format(i)])['id'].transform('count')
        df['{}_ts_day_hour_count'.format(i)] = df.groupby(['{}'.format(i), 'ts_day', 'ts_hour'])['id'].transform('count')

    # 类别特征五折转化率特征
    df['ID'] = df.index
    df['fold'] = df['ID'] % 5
    df.loc[df.target.isnull(), 'fold'] = 5
    target_feat = []
    for i in tqdm(cat_list):
        target_feat.extend([i + '_mean_last_1'])
        df[i + '_mean_last_1'] = None
        for fold in range(6):
            df.loc[df['fold'] == fold, i + '_mean_last_1'] = df[df['fold'] == fold][i].map(
                df[(df['fold'] != fold) & (df['fold'] != 5)].groupby(i)['target'].mean()
            )
        df[i + '_mean_last_1'] = df[i + '_mean_last_1'].astype(float)

    for feas in [('guid', 'netmodel'), ('guid', 'osversion'), ('guid', 'device_version')]:
        i, j = feas
        df['%s_%s_unique'%(i, j)] = df.groupby([i])[j].transform('nunique')

    def tag_score_sum(x):
        x = str(x)

        if '|' not in x:
            return 0

        x = x.split('|')
        x = [float(i.split(':')[1]) for i in x if len(i.split(':')) > 1]
        return sum(x)
    df['tag_sum'] = df['tag'].apply(lambda x: tag_score_sum(x))
    df['outertag_sum'] = df['outertag'].apply(lambda x: tag_score_sum(x))

    # print(df[['tag', 'tag_sum', 'outertag', 'outertag_sum']])

    lb_feas = ['app_version', 'device_vendor', 'device_version', 'deviceid', 'guid', 'netmodel', 'newsid', 'osversion',
               'timestamp', 'outertag', 'tag', 'applist', 'ts']

    for fea in lb_feas:
        print(fea)
        df[fea].fillna('w', inplace=True)
        df[fea] = df[fea].astype(str)
        df[fea] = LabelEncoder().fit_transform(df[fea])

    train = df[df['is_test'] != 1]
    test = df[df['is_test'] == 1]

    print(train.shape)
    print(test.shape)

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

# label = train['target']
# sub = test[['id']]

features = [fea for fea in train.columns if fea not in no_features]

# train_df = train[features]
#
# test_df = test[features]

# print(train)
def load_data():
    return train, test, no_features, features