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

start_time = time.time()


def get_time_str(x):
    dateArray = datetime.datetime.utcfromtimestamp(x)
    otherStyleTime = dateArray.strftime('%Y-%m-%d %H:%M:%S')
    return otherStyleTime


# 辅助函数
def statics():
    stats = []
    for col in train_df.columns:
        stats.append((col, train_df[col].nunique(), train_df[col].isnull().sum() * 100 / train_df.shape[0],
                      train_df[col].value_counts(normalize=True, dropna=False).values[0] * 100, train_df[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                            'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('Unique_values', ascending=False, inplace=True)
    stats_df.to_excel('tmp/stats_train.xlsx', index=None)


# 加载数据
root = Path('./data/')
train_df = pd.read_csv(root / 'train.csv')
train_df['target'] = train_df['target'].astype(int)
test_df = pd.read_csv(root / 'test.csv')
test_df['target'] = 0

# 将时间戳转为datetime
train_df['ts'] = train_df['ts'].apply(lambda x: get_time_str(x / 1000))
test_df['ts'] = test_df['ts'].apply(lambda x: get_time_str(x / 1000))
train_df['ts'] = pd.to_datetime(train_df['ts'])
test_df['ts'] = pd.to_datetime(test_df['ts'])


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


df = pd.concat([train_df, test_df], sort=False, axis=0)
preprocess_ts(df)

# statics()
app_df = pd.read_csv(root / 'app.csv')
user_df = pd.read_csv(root / 'user.csv')

# print(df)
# print(df.info())

cate_cols = ['device_version', 'device_vendor', 'app_version', 'osversion', 'netmodel'] + \
            ['pos', 'osversion'] + \
            ['guid', 'newsid']

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

    grouped_df = user_df.groupby(by='deviceid').agg({'outertag_nums': ['sum', 'median', 'mean']})
    grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    grouped_df = user_df.groupby(by='deviceid').agg({'outertag_score': ['sum', 'median', 'mean']})
    grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    grouped_df = user_df.groupby(by='deviceid').agg({'tag_nums': ['sum', 'median', 'mean']})
    grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    grouped_df = user_df.groupby(by='deviceid').agg({'tag_score': ['sum', 'median', 'mean']})
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
    # print('添加组合特征...')
    # pairs = [
    #     'deviceid,newsid', 'guid,newsid',
    #     'pos,newsid', 'device_vendor,newsid',
    #     'lng,newsid', 'hour,newsid',
    #     'dayofweek,newsid', 'dayofweek,hour',
    #     'netmodel,hour', 'netmodel,dayofweek'
    # ]
    # for pair in pairs:
    #     col1, col2 = pair.split(',')[0], pair.split(',')[1]
    #     df[pair] = (df[col1].astype(str) + df[col2].astype(str)).astype('category')
    #     tmp_count = dict(df[pair].value_counts())
    #     df[pair + '_count'] = df[pair].apply(lambda x: tmp_count[x])
    #     del tmp_count, df[pair]
    #     gc.collect()
    #
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
    #
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
#

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
    for tag, score in sorted(all_outertag.items(), key=lambda item: item[1], reverse=True)[:10]:
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
    for tag, score in sorted(all_tag.items(), key=lambda item: item[1], reverse=True)[:20]:
        top_tag[tag] = score

    for tag in top_tag:
        grouped_df[tag] = grouped_df['deviceid_tag'].apply(lambda x: top_tag[tag] if tag in x else 0)
    del top_tag, all_tag
    del grouped_df['deviceid_tag']
    gc.collect()
    return grouped_df


def get_cvr_fea(data,cat_list =None):

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



df=get_news_fea(df)
# df = get_ctr_fea(df)
df=get_combination_fea(df)
#
app_fea = get_app_fea()
df = pd.merge(df, app_fea, on='deviceid', how='left')
del app_fea
gc.collect()
#
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

no_features = ['id', 'target', 'ts', 'guid', 'deviceid', 'newsid', 'timestamp', 'ID', 'fold', '{}_count']
features = [fea for fea in df.columns if fea not in no_features]
for col in features:
    print(col)
    df[col] = df[col].fillna(-1)




user = user_df.drop_duplicates('deviceid')
df = df.merge(user[['deviceid', 'level', 'personidentification', 'followscore', 'personalscore', 'gender']],
                  how='left', on='deviceid')
del user

df = get_cvr_fea(df, cat_list=cate_cols+['deviceid','level', 'personidentification', 'followscore', 'personalscore', 'gender'])
df.fillna(0,inplace=True)

df = reduce_mem_usage(df)

train, test = df[:len(train_df)], df[len(train_df):]
df.head(200).to_csv('tmp/df.csv', index=None)

print("df shape", df.shape)
print("len(features),features", len(features), features)
print(train['target'].value_counts())
print("train shape", train.shape)
print("test shape", test.shape)

end_time = time.time()
print("生成特征耗时：", end_time - start_time)

X_train, y_train = train, train.target
X_test = test
del train, test

#
cols_use = [c for c in df.columns if (c not in no_features)]
X_train = X_train[cols_use]
X_test = X_test[cols_use]

categorical_features = features

col_vals_dict = {c: list(df[c].unique()) for c in categorical_features}
# print(col_vals_dict)
embed_cols = []
for c in col_vals_dict:
    if 2 <= len(col_vals_dict[c]) <= 1000:
        if 'last_1' not in c:
            embed_cols.append(c)
            print(c + ': %d values' % len(col_vals_dict[c]))  # look at value counts to know the embedding dimensions
print('\n')
print(len(embed_cols))

numerical_features = [c for c in df.columns if (not c in (embed_cols + no_features))]
print(len(numerical_features))

# scaler = StandardScaler()
# X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
# X_test[numerical_features] = scaler.transform(X_test[numerical_features])  # 标准化

for feat in numerical_features:
    X_train[feat] = X_train[feat].rank() / float(X_train.shape[0])  # 排序，并且进行归一化
    X_test[feat] = X_test[feat].rank() / float(X_test.shape[0])  # 排序，并且进行归一化

del df
del train_df
del test_df
gc.collect()


def load_data():
    return X_train, y_train, X_test, embed_cols, col_vals_dict, numerical_features
