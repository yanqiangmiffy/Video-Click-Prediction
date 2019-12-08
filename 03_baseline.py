import pandas as pd
import numpy as np
import time, datetime
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

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
del train, test
cate_cols = ['device_version', 'device_vendor', 'app_version', 'osversion', 'netmodel'] + \
            ['pos', 'osversion']

for col in cate_cols:
    lb = LabelEncoder()
    data[col] = data[col].fillna('999')
    data[col] = lb.fit_transform(data[col])
    data['{}_count'.format(col)] = data.groupby(col)['id'].transform('count')  #

# 小时信息
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute

# 缺失值填充
data['guid'] = data['guid'].fillna('abc')

# 构造历史特征 分别统计前一天 guid deviceid 的相关信息
# 8 9 10 11
history_9 = data[data['days'] == 8]
history_10 = data[data['days'] == 9]
history_11 = data[data['days'] == 10]
history_12 = data[data['days'] == 11]
del data


def get_history_visit_time(data1, date2):
    # 类别特征五折转化率特征
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
leak_cols = cate_cols = ['days', 'hour']
for col in [['deviceid'], ['guid'], ['newsid']]:
    for lc in leak_cols:
        data['{}_{}_count'.format('_'.join(col), lc)] = data.groupby([lc] + col)['id'].transform('count')

# netmodel
# data['netmodel'] = data['netmodel'].map({'o': 1, 'w': 2, 'g4': 4, 'g3': 3, 'g2': 2})
# pos
data['pos'] = data['pos']
print('train and predict')
X_train = data[data['flag'].isin([9])]
X_valid = data[data['flag'].isin([10])]
X_test = data[data['flag'].isin([11])]

lgb_param = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': -1,
    'seed': 42,
    'boost_from_average': 'false',
}

# features = [
#     'pos', 'netmodel', 'hour', 'minute',
#     'deviceid_timestamp_ts_max', 'deviceid_timestamp_ts_mean',
#     'deviceid_timestamp_ts_min', 'deviceid_timestamp_ts_median',
#     'guid_timestamp_ts_max', 'guid_timestamp_ts_mean',
#     'guid_timestamp_ts_min', 'guid_timestamp_ts_median',
#     'deviceid_days_count', 'guid_days_count', 'newsid_days_count',
#     'ts_next_ts'
# ]
no_features = ['id', 'target', 'ts', 'guid', 'deviceid', 'newsid', 'timestamp', 'ID', 'fold'] + \
              ['id', 'target', 'timestamp', 'ts', 'isTest', 'day',
               'lat_mode', 'lng_mode', 'abtarget', 'applist_key',
               'applist_weight', 'tag_key', 'tag_weight', 'outertag_key', 'tag', 'outertag',
               'outertag_weight', 'newsid', 'datetime'] + \
              ['days']
features = [fea for fea in data.columns if fea not in no_features]
print(features)
target = 'target'

lgb_train = lgb.Dataset(X_train[features].values, X_train[target].values)
lgb_valid = lgb.Dataset(X_valid[features].values, X_valid[target].values, reference=lgb_train)
lgb_model = lgb.train(lgb_param, lgb_train, num_boost_round=10000, valid_sets=[lgb_train, lgb_valid],
                      early_stopping_rounds=50, verbose_eval=10)

p_test = lgb_model.predict(X_valid[features].values, num_iteration=lgb_model.best_iteration)
xx_score = X_valid[[target]].copy()
xx_score['predict'] = p_test
xx_score = xx_score.sort_values('predict', ascending=False)
xx_score = xx_score.reset_index()
xx_score.loc[xx_score.index <= int(xx_score.shape[0] * 0.103), 'score'] = 1
xx_score['score'] = xx_score['score'].fillna(0)
print(f1_score(xx_score['target'], xx_score['score']))

del lgb_train, lgb_valid
del X_train, X_valid
# 没加 newsid 之前的 f1 score
# 0.5129179717875857
# 0.5197833317587095
# 0.6063125458760602
X_train_2 = data[data['flag'].isin([9, 10])]

lgb_train_2 = lgb.Dataset(X_train_2[features].values, X_train_2[target].values)
lgb_model_2 = lgb.train(lgb_param, lgb_train_2, num_boost_round=lgb_model.best_iteration, valid_sets=[lgb_train_2],
                        verbose_eval=10)

p_predict = lgb_model_2.predict(X_test[features].values)

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

submit_score[['id', 'target']].to_csv('result/baseline.csv', index=False)
