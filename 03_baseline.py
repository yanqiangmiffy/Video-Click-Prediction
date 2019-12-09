import pandas as pd
import numpy as np
import time, datetime
import lightgbm as lgb
from sklearn.metrics import f1_score
import gc
from tqdm import tqdm

train = pd.read_csv('data/train.csv')[:100000]
test = pd.read_csv('data/test.csv')[:100000]

df_app = pd.read_csv('data/app.csv')
df_user = pd.read_csv('data/user.csv')

# 对数据进行排序
train = train.sort_values(['deviceid', 'guid', 'ts'])
test = test.sort_values(['deviceid', 'guid', 'ts'])


# 时间格式转化 ts
def time_data2(time_sj):
    data_sj = time.localtime(time_sj / 1000)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", data_sj)
    return time_str


train['datetime'] = train['ts'].apply(time_data2)
test['datetime'] = test['ts'].apply(time_data2)

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

train['days'] = train['datetime'].dt.day
test['days'] = test['datetime'].dt.day

train['flag'] = train['days']
test['flag'] = 11

# 8 9 10 11
data = pd.concat([train, test], axis=0, sort=False)


# 全局特征


def get_app_fea():
    print("生成 app 特征....")
    app_grouped_df = pd.DataFrame({'deviceid': df_app['deviceid'].unique()})

    # 统计一个设备的出现过的app总数
    df_app['app_nums'] = df_app['applist'].apply(lambda x: len(x.replace('[', '').replace(']', '').split(' ')))
    df_app.app_nums.head()

    grouped_df = df_app.groupby(by='deviceid').agg({'app_nums': ['sum']})
    grouped_df.columns = ['app_nums_sum']
    grouped_df = grouped_df.reset_index()
    app_grouped_df = pd.merge(app_grouped_df, grouped_df, on='deviceid', how='left')

    # 统计一个设备上applist对应的不同device个数总数
    df_app['applist_count'] = df_app.groupby('applist')['deviceid'].transform('count')
    grouped_df = df_app.groupby(by='deviceid').agg({'applist_count': ['sum']})
    grouped_df.columns = ['applist_count_sum']
    grouped_df = grouped_df.reset_index()
    app_grouped_df = pd.merge(app_grouped_df, grouped_df, on='deviceid', how='left')

    return app_grouped_df


def get_user_fea():
    print("生成 user 特征....")

    user_grouped_df = pd.DataFrame({'deviceid': df_user['deviceid'].unique()})

    # 统计一个设备的注册不同用户个数
    grouped_df = df_user.groupby(by='deviceid').agg({'guid': ['nunique']})
    grouped_df.columns = ['deviceid_unique_guid']
    grouped_df = grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    df_user['deviceid_nunique_guid'] = df_user.groupby('deviceid').guid.transform('nunique')

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

    df_user['outertag_nums'] = df_user['outertag'].astype('str').apply(lambda x: get_outertag_nums(x))
    df_user['outertag_score'] = df_user['outertag'].astype('str').apply(lambda x: get_outertag_score(x))

    df_user['tag_nums'] = df_user['tag'].astype('str').apply(lambda x: get_outertag_nums(x))
    df_user['tag_score'] = df_user['tag'].astype('str').apply(lambda x: get_outertag_score(x))

    grouped_df = df_user.groupby(by='deviceid').agg({'outertag_nums': ['sum']})
    grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    grouped_df = df_user.groupby(by='deviceid').agg({'outertag_score': ['sum']})
    grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    grouped_df = df_user.groupby(by='deviceid').agg({'tag_nums': ['sum']})
    grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    grouped_df = df_user.groupby(by='deviceid').agg({'tag_score': ['sum']})
    grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')
    #
    # 设备的用户等级统计
    grouped_df = df_user.groupby(by='deviceid').agg({'level': ['sum']})
    grouped_df.columns = ['deviceid_level_sum']
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    # 设备的用户劣质统计
    # 1表示劣质用户 0表示正常用户。
    grouped_df = df_user.groupby(by='deviceid').agg({'personidentification': ['sum']})
    grouped_df.columns = ['deviceid_personidentification_sum']
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    grouped_df = df_user.groupby(by='deviceid').agg({'personalscore': ['sum']})
    grouped_df.columns = ['deviceid_personalscore_sum']
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')
    #
    grouped_df = df_user.groupby(by='deviceid').agg({'followscore': ['sum']})
    grouped_df.columns = ['deviceid_followscore_sum']
    grouped_df.reset_index()
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    return user_grouped_df


def get_outertag_fea():
    print("get_outertag_fea....")
    df_user['outertag'] = df_user['outertag'].astype(str)
    grouped_df = df_user.groupby('deviceid').agg({'outertag': '|'.join})
    grouped_df.columns = ['deviceid_' + 'outertag']
    # 最受欢迎的50个outertag
    all_outertag = {}
    for x in df_user.outertag:
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
    df_user['tag'] = df_user['tag'].astype(str)
    grouped_df = df_user.groupby('deviceid').agg({'tag': '|'.join})
    grouped_df.columns = ['deviceid_' + 'tag']
    all_tag = {}
    for x in df_user.tag:
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


app_fea = get_app_fea()
data = pd.merge(data, app_fea, on='deviceid', how='left')
del app_fea
gc.collect()

user_fea = get_user_fea()
data = pd.merge(data, user_fea, on='deviceid', how='left')
del user_fea
gc.collect()

outertag_fea = get_outertag_fea()
data = pd.merge(data, outertag_fea, on='deviceid', how='left')
del outertag_fea
gc.collect()

tag_fea = get_tag_fea()
data = pd.merge(data, tag_fea, on='deviceid', how='left')
del tag_fea
gc.collect()

cluster_fea = pd.read_csv('features/01_user_cluster.csv')
data = pd.merge(data, cluster_fea, on='deviceid', how='left')
del cluster_fea
gc.collect()

del train, test

# 小时信息
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute
data['dayofweek'] = data['datetime'].dt.dayofweek

# 缺失值填充
data['guid'] = data['guid'].fillna('abc')
from sklearn.preprocessing import LabelEncoder

cate_cols = ['device_version', 'device_vendor', 'app_version', 'osversion', 'netmodel'] + \
            ['pos', 'osversion']

# df=pd.get_dummies(df,columns=cate_cols)
for col in cate_cols:
    lb = LabelEncoder()
    data[col] = data[col].fillna('999')
    data[col] = lb.fit_transform(data[col])
    data['{}_count'] = data.groupby(col)['id'].transform('count')  #
# 构造历史特征 分别统计前一天 guid deviceid 的相关信息
# 8 9 10 11
history_9 = data[data['days'] == 8]
history_10 = data[data['days'] == 9]
history_11 = data[data['days'] == 10]
history_12 = data[data['days'] == 11]

del data


# deviceid guid timestamp ts 时间特征
def get_history_visit_time(data1, date2):
    print("get_history_visit_time")
    data1 = data1.sort_values(['ts', 'timestamp'])
    data1['timestamp_ts'] = data1['timestamp'] - data1['ts']
    data1_tmp = data1[data1['target'] == 1].copy()
    del data1
    for col in ['deviceid', 'guid', 'newsid']:
        for ts in ['timestamp_ts']:
            f_tmp = data1_tmp.groupby([col], as_index=False)[ts].agg({
                '{}_{}_max'.format(col, ts): 'max',
                '{}_{}_mean'.format(col, ts): 'mean',
                '{}_{}_min'.format(col, ts): 'min',
                '{}_{}_median'.format(col, ts): 'median'
            })
        date2 = pd.merge(date2, f_tmp, on=[col], how='left', copy=False)

    return date2


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


# def get_ctr_fea(df):
#     print("get_ctr_fea....")
#     df['news_ctr_rate'] = df.groupby('newsid')['target'].transform('mean')  #
#     # df['lat_ctr_rate'] = df.groupby('lat')['target'].transform('mean')  #
#     # df['lng_ctr_rate'] = df.groupby('lng')['target'].transform('mean')  #
#     # df['ts_ctr_rate'] = df.groupby('ts')['target'].transform('mean')  #
#     # df['deviceid_ctr_rate'] = df.groupby('deviceid')['target'].transform('mean')  #
#     # df['guid_ctr_rate'] = df.groupby('guid')['target'].transform('mean')  #
#     # df['device_version_ctr_rate'] = df.groupby('device_version')['target'].transform('mean')  #
#     # df['device_vendor_ctr_rate'] = df.groupby('device_vendor')['target'].transform('mean')  #
#     # df['app_version_ctr_rate'] = df.groupby('app_version')['target'].transform('mean')  #
#     # df['osversion_ctr_rate'] = df.groupby('osversion')['target'].transform('mean')  #
#     # df['pos_ctr_rate'] = df.groupby('pos')['target'].transform('mean')  #
#     df['netmodel_ctr_rate'] = df.groupby('netmodel')['target'].transform('mean')  #
#     return df


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


def get_cvr_fea(df):
    cat_list = set(['device_version', 'device_vendor', 'app_version', 'osversion', 'netmodel'] + \
                   ['pos', 'osversion'] + \
                   ['deviceid', 'level', 'personidentification', 'followscore', 'personalscore',
                    'gender'])
    print("cat_list", cat_list)

    # 类别特征五折转化率特征
    print("转化率特征....")
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

    return df


history_9 = get_news_fea(history_9)
history_10 = get_news_fea(history_10)
history_11 = get_news_fea(history_11)
history_12 = get_news_fea(history_12)

# history_9 = get_ctr_fea(history_9)
# history_10 = get_ctr_fea(history_10)
# history_11 = get_ctr_fea(history_11)
# history_12 = get_ctr_fea(history_12)

history_9 = get_combination_fea(history_9)
history_10 = get_combination_fea(history_10)
history_11 = get_combination_fea(history_11)
history_12 = get_combination_fea(history_12)

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
    data['{}_days_count'.format('_'.join(col))] = data.groupby(['days'] + col)['id'].transform('count')

# netmodel
data['netmodel'] = data['netmodel'].map({'o': 1, 'w': 2, 'g4': 4, 'g3': 3, 'g2': 2})

# pos
data['pos'] = data['pos']

print('train and predict')
X_train = data[data['flag'].isin([9])]
X_valid = data[data['flag'].isin([10])]
X_test = data[data['flag'].isin([11])]
X_train_2 = data[data['flag'].isin([9, 10])]
del data
lgb_param = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': -1,
    'seed': 42,
    'boost_from_average': 'false',
}

# feature = [
#     'pos', 'netmodel', 'hour', 'minute',
#     'deviceid_timestamp_ts_max', 'deviceid_timestamp_ts_mean',
#     'deviceid_timestamp_ts_min', 'deviceid_timestamp_ts_median',
#     'guid_timestamp_ts_max', 'guid_timestamp_ts_mean',
#     'guid_timestamp_ts_min', 'guid_timestamp_ts_median',
#     'deviceid_days_count', 'guid_days_count', 'newsid_days_count',
#     'ts_next_ts'
# ]
target = 'target'

no_features = ['id', 'target', 'ts', 'guid', 'deviceid', 'newsid', 'timestamp', 'ID', 'fold'] + \
              ['id', 'target', 'timestamp', 'ts', 'isTest', 'day',
               'lat_mode', 'lng_mode', 'abtarget', 'applist_key',
               'applist_weight', 'tag_key', 'tag_weight', 'outertag_key', 'tag', 'outertag',
               'outertag_weight', 'newsid', 'datetime'] + ['days']

feature = [fea for fea in X_train.columns if fea not in no_features]
print(len(feature), feature, )
lgb_train = lgb.Dataset(X_train[feature].values, X_train[target].values)
lgb_valid = lgb.Dataset(X_valid[feature].values, X_valid[target].values, reference=lgb_train)
lgb_model = lgb.train(lgb_param, lgb_train, num_boost_round=10000, valid_sets=[lgb_train, lgb_valid],
                      early_stopping_rounds=50, verbose_eval=10)

p_test = lgb_model.predict(X_valid[feature].values, num_iteration=lgb_model.best_iteration)
xx_score = X_valid[[target]].copy()
xx_score['predict'] = p_test
xx_score = xx_score.sort_values('predict', ascending=False)
xx_score = xx_score.reset_index()
xx_score.loc[xx_score.index <= int(xx_score.shape[0] * 0.103), 'score'] = 1
xx_score['score'] = xx_score['score'].fillna(0)
print(f1_score(xx_score['target'], xx_score['score']))
F1_score = f1_score(xx_score['target'], xx_score['score'])
del lgb_train, lgb_valid
del X_train, X_valid
# 没加 newsid 之前的 f1 score
# 0.5129179717875857
# 0.5197833317587095
# 0.6063125458760602

lgb_train_2 = lgb.Dataset(X_train_2[feature].values, X_train_2[target].values)
lgb_model_2 = lgb.train(lgb_param, lgb_train_2, num_boost_round=lgb_model.best_iteration, valid_sets=[lgb_train_2],
                        verbose_eval=10)

p_predict = lgb_model_2.predict(X_test[feature].values)

submit_score = X_test[['id']].copy()
submit_score['predict'] = p_predict
submit_score = submit_score.sort_values('predict', ascending=False)
submit_score = submit_score.reset_index()
submit_score.loc[submit_score.index <= int(submit_score.shape[0] * 0.103), 'target'] = 1
submit_score['target'] = submit_score['target'].fillna(0)

submit_score = submit_score.sort_values('id')
submit_score['target'] = submit_score['target'].astype(int)
sample = pd.read_csv('data/sample.csv')
sample.columns = ['id', 'non_target']
submit_score = pd.merge(sample, submit_score, on=['id'], how='left')
submit_score[['id', 'target']].to_csv('result/baseline{}.csv'.format(F1_score), index=False)
