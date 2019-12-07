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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

start_time = time.time()


def get_time_str(x):
    date_utc = datetime.datetime.utcfromtimestamp(x)
    time_str = date_utc.strftime('%Y-%m-%d %H:%M:%S')
    return time_str


def preprocess_ts(data):
    """
    时间特征,提取天，时，分，周几
    :param data:DataFrame
    :return:
    """
    data["day"] = data["datetime"].dt.day
    data["hour"] = data["datetime"].dt.hour
    data["minute"] = data["datetime"].dt.minute
    data["dayofweek"] = data["datetime"].dt.dayofweek
    return data


# 加载数据
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
app_df = pd.read_csv('data/app.csv')
user_df = pd.read_csv('data/user.csv')

# -------------- 数据预处理 -----------------
# 对数据进行排序
train_df = train_df.sort_values(['deviceid', 'guid', 'ts'])
test_df = test_df.sort_values(['deviceid', 'guid', 'ts'])

# 查看数据是否存在交集
# train deviceid 104736
# test deviceid 56681
# train&test deviceid 46833
# train guid 104333
# test guid 56861
# train&test guid 46654

print('train deviceid', len((set(train_df['deviceid']))))
print('test deviceid', len((set(test_df['deviceid']))))
print('train&test deviceid', len((set(train_df['deviceid']) & set(test_df['deviceid']))))
print('train guid', len((set(train_df['guid']))))
print('test guid', len((set(test_df['guid']))))
print('train&test guid', len((set(train_df['guid']) & set(test_df['guid']))))

# 将时间戳转为datetime
train_df['datetime'] = train_df['ts'].apply(lambda x: get_time_str(x / 1000))
test_df['datetime'] = test_df['ts'].apply(lambda x: get_time_str(x / 1000))
train_df['datetime'] = pd.to_datetime(train_df['datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])

# 时间范围
# 2019-11-07 23:59:59 2019-11-10 23:59:59
# 2019-11-10 23:59:59 2019-11-11 23:59:59
print(train_df['datetime'].min(), train_df['datetime'].max())
print(test_df['datetime'].min(), test_df['datetime'].max())

# 提取相关时间
train_df = preprocess_ts(train_df)
test_df = preprocess_ts(test_df)

# 8 9 10 11
# 历史3天预测未来1天
train_df['flag'] = train_df['day']
test_df['flag'] = 11

df = pd.concat([train_df, test_df], sort=False, axis=0)
del train_df, test_df
gc.collect()

# guid  缺失值处理 当做游客 abc填充
df['guid'] = df['guid'].fillna('abc')
# -------------- 数据预处理 end -----------------

# -------------- 特征工程 -----------------

cate_cols = ['device_version', 'device_vendor', 'app_version', 'osversion', 'netmodel'] + \
            ['pos', 'osversion']

# df=pd.get_dummies(df,columns=cate_cols)
for col in cate_cols:
    lb = LabelEncoder()
    df[col] = df[col].fillna('999')
    df[col] = lb.fit_transform(df[col])
    df['{}_count'] = df.groupby(col)['id'].transform('count')  #


def get_app_fea():
    print("生成 app 特征....")
    app_grouped_df = pd.DataFrame({'deviceid': app_df['deviceid'].unique()})

    # 统计一个设备的出现过的app总数
    app_df['app_nums'] = app_df['applist'].apply(lambda x: len(x.replace('[', '').replace(']', '').split(' ')))
    app_df.app_nums.head()

    grouped_df = app_df.groupby(by='deviceid').agg({'app_nums': ['sum']})
    grouped_df.columns = ['app_nums_sum']
    grouped_df = grouped_df.reset_index()
    app_grouped_df = pd.merge(app_grouped_df, grouped_df, on='deviceid', how='left')

    # 统计一个设备上applist对应的不同device个数总数
    app_df['applist_count'] = app_df.groupby('applist')['deviceid'].transform('count')
    grouped_df = app_df.groupby(by='deviceid').agg({'applist_count': ['sum']})
    grouped_df.columns = ['applist_count_sum']
    grouped_df = grouped_df.reset_index()
    app_grouped_df = pd.merge(app_grouped_df, grouped_df, on='deviceid', how='left')

    return app_grouped_df


def get_user_fea():
    print("生成 user 特征....")

    user_grouped_df = pd.DataFrame({'deviceid': user_df['deviceid'].unique()})

    # 统计一个设备的注册不同用户个数
    grouped_df = user_df.groupby(by='deviceid').agg({'guid': ['nunique']})
    grouped_df.columns = ['deviceid_unique_guid']
    grouped_df = grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    user_df['deviceid_nunique_guid'] = user_df.groupby('deviceid').guid.transform('nunique')

    # 一个设备的outertag 的统计
    def get_outertag_nums(x):
        """
        获取一个outertag的tag个数
        """
        if x == 'nan':
            return 0
        return len(x.split('|')) - 1

    def get_outertag_score(x):
        """
        获取一个outertag的tag分数和
        """
        tags = x.split('|')

        score = 0
        if len(tags) == 1 and tags[0] == 'nan':
            return score
        else:
            for tag in tags:
                if len(tag.split(':')) == 2:
                    score += float(tag.split(':')[1])
                else:
                    score += 0
        return score

    user_df['outertag_nums'] = user_df['outertag'].astype('str').apply(lambda x: get_outertag_nums(x))
    user_df['outertag_score'] = user_df['outertag'].astype('str').apply(lambda x: get_outertag_score(x))

    user_df['tag_nums'] = user_df['tag'].astype('str').apply(lambda x: get_outertag_nums(x))
    user_df['tag_score'] = user_df['tag'].astype('str').apply(lambda x: get_outertag_score(x))

    grouped_df = user_df.groupby(by='deviceid').agg({'outertag_nums': ['sum']})
    grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    grouped_df = user_df.groupby(by='deviceid').agg({'outertag_score': ['sum']})
    grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    grouped_df = user_df.groupby(by='deviceid').agg({'tag_nums': ['sum']})
    grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    grouped_df = user_df.groupby(by='deviceid').agg({'tag_score': ['sum']})
    grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')
    #
    # 设备的用户等级统计
    grouped_df = user_df.groupby(by='deviceid').agg({'level': ['sum']})
    grouped_df.columns = ['deviceid_level_sum']
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    # 设备的用户劣质统计
    # 1表示劣质用户 0表示正常用户。
    grouped_df = user_df.groupby(by='deviceid').agg({'personidentification': ['sum']})
    grouped_df.columns = ['deviceid_personidentification_sum']
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    grouped_df = user_df.groupby(by='deviceid').agg({'personalscore': ['sum']})
    grouped_df.columns = ['deviceid_personalscore_sum']
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')
    #
    grouped_df = user_df.groupby(by='deviceid').agg({'followscore': ['sum']})
    grouped_df.columns = ['deviceid_followscore_sum']
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    return user_grouped_df


def get_news_fea(df):
    print("get_news_fea....")
    # 视频出现次数
    df['news_count'] = df.groupby('newsid')['id'].transform('count')  #
    # df['news_target_sum'] = df.groupby('newsid')['target'].transform('sum')  # 点击次数
    # 视频推荐的人数
    df['news_guid_unique'] = df.groupby(by='newsid')['guid'].transform('nunique')  # 人数

    df['news_deviceid_unique'] = df.groupby(by='newsid')['deviceid'].transform('nunique')  # 设备
    df['news_pos_unique'] = df.groupby(by='newsid')['pos'].transform('nunique')
    df['news_app_version_unique'] = df.groupby(by='newsid')['app_version'].transform('nunique')
    df['news_device_vendor_unique'] = df.groupby(by='newsid')['device_vendor'].transform('nunique')
    df['news_netmodel_unique'] = df.groupby(by='newsid')['netmodel'].transform('nunique')
    df['news_osversion_unique'] = df.groupby(by='newsid')['osversion'].transform('nunique')
    df['news_device_version_unique'] = df.groupby(by='newsid')['device_version'].transform('nunique')

    df['news_lng_unique'] = df.groupby(by='newsid')['lng'].transform('nunique')  # 地理
    df['news_lat_unique'] = df.groupby(by='newsid')['lat'].transform('nunique')

    # 时间阶段出现的次数
    df['news_hour_unique'] = df.groupby(by='newsid')['hour'].transform('nunique')  # 地理
    df['news_dayofweek_unique'] = df.groupby(by='newsid')['dayofweek'].transform('nunique')

    # 逆向unique
    df['guid_news_unique'] = df.groupby(by='guid')['newsid'].transform('nunique')  # 人数
    df['deviceid_news_unique'] = df.groupby(by='deviceid')['newsid'].transform('nunique')  # 设备
    df['pos_news_unique'] = df.groupby(by='pos')['newsid'].transform('nunique')
    df['app_version_news_unique'] = df.groupby(by='app_version')['newsid'].transform('nunique')
    df['device_vendor_news_unique'] = df.groupby(by='device_vendor')['newsid'].transform('nunique')
    df['netmodel_news_unique'] = df.groupby(by='netmodel')['newsid'].transform('nunique')
    df['osversion_news_unique'] = df.groupby(by='osversion')['newsid'].transform('nunique')
    df['device_version_news_unique'] = df.groupby(by='device_version')['newsid'].transform('nunique')
    df['lng_news_unique'] = df.groupby(by='lng')['newsid'].transform('nunique')  # 地理
    df['lat_news_unique'] = df.groupby(by='lat')['newsid'].transform('nunique')
    df['hour_news_unique'] = df.groupby(by='hour')['newsid'].transform('nunique')  # 地理
    df['dayofweek_news_unique'] = df.groupby(by='dayofweek')['newsid'].transform('nunique')
    return df


def get_ctr_fea(df):
    print("get_ctr_fea....")
    df['news_ctr_rate'] = df.groupby('newsid')['target'].transform('mean')  #
    # df['lat_ctr_rate'] = df.groupby('lat')['target'].transform('mean')  #
    # df['lng_ctr_rate'] = df.groupby('lng')['target'].transform('mean')  #
    # df['ts_ctr_rate'] = df.groupby('ts')['target'].transform('mean')  #
    # df['deviceid_ctr_rate'] = df.groupby('deviceid')['target'].transform('mean')  #
    # df['guid_ctr_rate'] = df.groupby('guid')['target'].transform('mean')  #
    # df['device_version_ctr_rate'] = df.groupby('device_version')['target'].transform('mean')  #
    # df['device_vendor_ctr_rate'] = df.groupby('device_vendor')['target'].transform('mean')  #
    # df['app_version_ctr_rate'] = df.groupby('app_version')['target'].transform('mean')  #
    # df['osversion_ctr_rate'] = df.groupby('osversion')['target'].transform('mean')  #
    # df['pos_ctr_rate'] = df.groupby('pos')['target'].transform('mean')  #
    df['netmodel_ctr_rate'] = df.groupby('netmodel')['target'].transform('mean')  #
    return df


def get_combination_fea(df):
    """
    添加组合特征
    :return:
    """
    print('添加组合特征...')
    combination_cols = []
    df['deviceid_newsid'] = (df['deviceid'].astype(str) + df['newsid'].astype(str)).astype('category')
    df['guid_newsid'] = (df['guid'].astype(str) + df['newsid'].astype(str)).astype('category')
    df['pos_newsid'] = (df['pos'].astype(str) + df['newsid'].astype(str)).astype('category')
    df['device_vendor_newsid'] = (df['device_vendor'].astype(str) + df['newsid'].astype(str)).astype('category')
    df['lng_newsid'] = (df['lng'].astype(str) + df['newsid'].astype(str)).astype('category')
    df['hour_newsid'] = (df['hour'].astype(str) + df['newsid'].astype(str)).astype('category')
    df['dayofweek_newsid'] = (df['dayofweek'].astype(str) + df['newsid'].astype(str)).astype('category')

    df['dayofweek_hour'] = (df['dayofweek'].astype(str) + df['hour'].astype(str)).astype('category')

    df['netmodel_hour'] = (df['netmodel'].astype(str) + df['hour'].astype(str)).astype('category')
    df['netmodel_dayofweek'] = (df['netmodel'].astype(str) + df['dayofweek'].astype(str)).astype('category')

    combination_cols.extend([
        'deviceid_newsid', 'guid_newsid',
        'pos_newsid', 'device_vendor_newsid',
        'lng_newsid', 'hour_newsid',
        'dayofweek_newsid', 'dayofweek_hour',
        'netmodel_hour', 'netmodel_dayofweek'
    ])

    for col in combination_cols:
        print(col)
        df['{}_count'.format(col)] = df.groupby(col)['id'].transform('count')
        del df[col]
        gc.collect()
    return df


def get_outertag_fea():
    print("get_outertag_fea....")
    user_df['outertag'] = user_df['outertag'].astype(str)
    grouped_df = user_df.groupby('deviceid').agg({'outertag': '|'.join})
    grouped_df.columns = ['deviceid_' + 'outertag']
    # 最受欢迎的50个outertag
    all_outertag = {}
    for x in user_df.outertag:
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
    for tag, score in sorted(all_outertag.items(), key=lambda item: item[1], reverse=True)[:5]:
        top_outertag[tag] = score
    for tag in top_outertag:
        grouped_df[tag] = grouped_df['deviceid_outertag'].apply(lambda x: top_outertag[tag] if tag in x else 0)
    del top_outertag, all_outertag
    del grouped_df['deviceid_outertag']
    gc.collect()

    return grouped_df


def get_tag_fea():
    # 最受欢迎的100个tag
    print("get_tag_fea....")
    user_df['tag'] = user_df['tag'].astype(str)
    grouped_df = user_df.groupby('deviceid').agg({'tag': '|'.join})
    grouped_df.columns = ['deviceid_' + 'tag']
    all_tag = {}
    for x in user_df.tag:
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
    for tag, score in sorted(all_tag.items(), key=lambda item: item[1], reverse=True)[:10]:
        top_tag[tag] = score

    for tag in top_tag:
        grouped_df[tag] = grouped_df['deviceid_tag'].apply(lambda x: top_tag[tag] if tag in x else 0)
    del top_tag, all_tag
    del grouped_df['deviceid_tag']
    gc.collect()
    return grouped_df


def get_cvr_fea(data, cat_list=None):
    print("cat_list", cat_list)
    # 类别特征五折转化率特征
    print("转化率特征....")
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

    return data


df = get_news_fea(df)
df = get_combination_fea(df)
#
app_fea = get_app_fea()
df = pd.merge(df, app_fea, on='deviceid', how='left')
del app_fea
gc.collect()

user_fea = get_user_fea()
df = pd.merge(df, user_fea, on='deviceid', how='left')
del user_fea
gc.collect()

outertag_fea = get_outertag_fea()
df = pd.merge(df, outertag_fea, on='deviceid', how='left')
del outertag_fea
gc.collect()

tag_fea = get_tag_fea()
df = pd.merge(df, tag_fea, on='deviceid', how='left')
del tag_fea
gc.collect()

# cluster_fea = pd.read_csv('features/01_user_cluster.csv')
# df = pd.merge(df, cluster_fea, on='deviceid', how='left')
# del cluster_fea
# gc.collect()

user = user_df.drop_duplicates('deviceid')
df = df.merge(user[['deviceid', 'level', 'personidentification', 'followscore', 'personalscore', 'gender']],
              how='left', on='deviceid')
del user

df = get_cvr_fea(df,
                 cate_cols + ['deviceid', 'level', 'personidentification', 'followscore', 'personalscore', 'gender'])


def get_deepfm(data):
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
    # del ts_group
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
    # del group, ts_group

    data.loc[data['ts_before_rank'] == np.inf, 'ts_before_rank'] = 0
    data.loc[data['ts_after_rank'] == np.inf, 'ts_after_rank'] = 0
    data['ts_before_len'] = np.log(data['ts_before_len'] + 1)
    data['ts_after_len'] = np.log(data['ts_after_len'] + 1)

    def split(key_ans):
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    # 'deviceid'不唯一
    app = app_df
    app['applist'] = app['applist'].apply(lambda x: x[1:-2])
    group = app.groupby('deviceid')
    # del app

    gps = group['applist'].apply(lambda x: list(set(' '.join(x).split(' '))))
    # del group
    gps = pd.DataFrame(gps)
    key2index = {}
    gps['applist_key'] = list(map(split, gps['applist']))
    gps['applist_len'] = gps['applist'].apply(lambda x: len(x))
    gps['applist_weight'] = gps['applist_len'].apply(lambda x: x * [1])
    gps.drop('applist', axis=1, inplace=True)
    print(len(key2index))
    data = pd.merge(data, gps, on=['deviceid'], how='left')
    # del key2index, gps

    #  ['deviceid', 'guid']唯一， 'deviceid'不唯一
    user = user_df
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
            map(split, user.loc[user['%s_key' % i].isna() == False, '%s_key' % i]))
        user.drop(i, axis=1, inplace=True)
        print(len(key2index))
    user['guid'].fillna('', inplace=True)
    data['guid'].fillna('', inplace=True)
    print(user.columns)
    print(data.columns)
    data = pd.merge(data, user, on=['deviceid', 'guid'], how='left')
    # del user
    from scipy import stats
    min_time = data['ts'].min()
    data['timestamp'] -= min_time
    data['ts'] -= min_time
    data['lat_int'] = np.int64(np.rint(data['lat'] * 100))
    data['lng_int'] = np.int64(np.rint(data['lng'] * 100))
    # data.loc[data['level'].isna() == False, 'level_int'] = np.int64(
    #     data.loc[data['level'].isna() == False, 'level'])
    group = data[['deviceid', 'lat', 'lng']].groupby('deviceid')
    gp = group[['lat', 'lng']].agg(lambda x: stats.mode(x)[0][0]).reset_index()
    gp.columns = ['deviceid', 'lat_mode', 'lng_mode']
    data = pd.merge(data, gp, on='deviceid', how='left')
    # del group, gp
    data['dist'] = np.log((data['lat'] - data['lat_mode']) **
                          2 + (data['lng'] - data['lng_mode']) ** 2 + 1)
    data['dist_int'] = np.rint(data['dist'])
    data.loc[data['lat'] != data['lat_mode'], 'isLatSame'] = 0
    data.loc[data['lat'] == data['lat_mode'], 'isLatSame'] = 1
    data.loc[data['lng'] != data['lng_mode'], 'isLngSame'] = 0
    data.loc[data['lng'] == data['lng_mode'], 'isLngSame'] = 1

    # data.loc[data['personalscore'].isna(), 'personalscore'] = data['personalscore'].mode()
    return data


df = get_deepfm(df)

df['day_diff'] = np.sign(df[['day']].diff().fillna(0))
df['hour_diff'] = np.sign(df[['hour']].diff().fillna(0))
df['minute_diff'] = np.sign(df[['minute']].diff().fillna(0))
# 构造历史特征 分别统计前一天 guid deviceid 的相关信息
# 8 9 10 11
history_9 = df[df['day'] == 8]
history_10 = df[df['day'] == 9]
history_11 = df[df['day'] == 10]
history_12 = df[df['day'] == 11]
del df
gc.collect()
# 61326
# 64766
# 66547
# 41933
# 42546
print(len(set(history_9['deviceid'])))
print(len(set(history_10['deviceid'])))
print(len(set(history_11['deviceid'])))
print(len(set(history_12['deviceid'])))
print(len(set(history_9['deviceid']) & set(history_10['deviceid'])))
print(len(set(history_10['deviceid']) & set(history_11['deviceid'])))
print(len(set(history_11['deviceid']) & set(history_12['deviceid'])))

# 61277
# 64284
# 66286
# 41796
# 42347

print(len(set(history_9['guid'])))
print(len(set(history_10['guid'])))
print(len(set(history_11['guid'])))
print(len(set(history_12['guid'])))
print(len(set(history_9['guid']) & set(history_10['guid'])))
print(len(set(history_10['guid']) & set(history_11['guid'])))
print(len(set(history_11['guid']) & set(history_12['guid'])))

# 640066
# 631547
# 658787
# 345742
# 350542

print(len(set(history_9['newsid'])))
print(len(set(history_10['newsid'])))
print(len(set(history_11['newsid'])))
print(len(set(history_12['newsid'])))
print(len(set(history_9['newsid']) & set(history_10['newsid'])))
print(len(set(history_10['newsid']) & set(history_11['newsid'])))
print(len(set(history_11['newsid']) & set(history_12['newsid'])))


# deviceid guid timestamp ts 时间特征
def get_history_visit_time(data1, date2):
    data1 = data1.sort_values(['ts', 'timestamp'])
    data1['timestamp_ts'] = data1['timestamp'] - data1['ts']
    data1_tmp = data1[data1['target'] == 1].copy()
    del data1
    for col in ['deviceid', 'guid']:
        for ts in ['timestamp_ts']:
            f_tmp = data1_tmp.groupby([col], as_index=False)[ts].agg({
                '{}_{}_max'.format(col, ts): 'max',
                '{}_{}_mean'.format(col, ts): 'mean',
                '{}_{}_min'.format(col, ts): 'min',
                '{}_{}_median'.format(col, ts): 'median'
            })
        date2 = pd.merge(date2, f_tmp, on=[col], how='left', copy=False)

    return date2


history_10 = get_history_visit_time(history_9, history_10)
history_11 = get_history_visit_time(history_10, history_11)
history_12 = get_history_visit_time(history_11, history_12)

data = pd.concat([history_10, history_11], axis=0, sort=False, ignore_index=True)
data = pd.concat([data, history_12], axis=0, sort=False, ignore_index=True)
del history_9, history_10, history_11, history_12

data = data.sort_values('ts')
data['ts_next'] = data.groupby(['deviceid'])['ts'].shift(-1)
data['ts_next_ts'] = data['ts_next'] - data['ts']

# 当前一天内的特征 leak
for col in [['deviceid'], ['guid'], ['newsid']]:
    print(col)
    data['{}_days_count'.format('_'.join(col))] = data.groupby(['day'] + col)['id'].transform('count')



print('train and predict')
X_train = data[data['flag'].isin([9])]
X_valid = data[data['flag'].isin([10])]
X_test = data[data['flag'].isin([11])]

X_train_2 = data[data['flag'].isin([9, 10])]

X_train = reduce_mem_usage(X_train)
X_valid = reduce_mem_usage(X_valid)
X_test = reduce_mem_usage(X_test)

no_features = ['id', 'target', 'ts', 'guid', 'deviceid', 'newsid', 'timestamp', 'ID', 'fold'] + \
              ['id', 'target', 'timestamp', 'ts', 'isTest', 'day',
               'lat_mode', 'lng_mode', 'abtarget', 'applist_key',
               'applist_weight', 'tag_key', 'tag_weight', 'outertag_key',
               'outertag_weight', 'newsid','datetime']
features = [fea for fea in X_train.columns if fea not in no_features]

end_time = time.time()
print("生成特征耗时：", end_time - start_time)


def load_data():
    return X_train,X_train_2, X_valid, X_test, no_features, features
