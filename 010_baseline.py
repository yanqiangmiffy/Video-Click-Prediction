import pandas as pd
import numpy as np
import time,datetime
import lightgbm as lgb
from sklearn.metrics import f1_score


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 对数据进行排序
train = train.sort_values(['deviceid','guid','ts'])
test = test.sort_values(['deviceid','guid','ts'])

# 查看数据是否存在交集
# train deviceid 104736
# test deviceid 56681
# train&test deviceid 46833
# train guid 104333
# test guid 56861
# train&test guid 46654

print('train deviceid',len((set(train['deviceid']))))
print('test deviceid',len((set(test['deviceid']))))
print('train&test deviceid',len((set(train['deviceid'])&set(test['deviceid']))))
print('train guid',len((set(train['guid']))))
print('test guid',len((set(test['guid']))))
print('train&test guid',len((set(train['guid'])&set(test['guid']))))

# 时间格式转化 ts
def time_data2(time_sj):
    data_sj = time.localtime(time_sj/1000)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S",data_sj)
    return time_str

train['datetime'] = train['ts'].apply(time_data2)
test['datetime'] = test['ts'].apply(time_data2)

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

# 时间范围
# 2019-11-07 23:59:59 2019-11-10 23:59:59
# 2019-11-10 23:59:59 2019-11-11 23:59:59
print(train['datetime'].min(),train['datetime'].max())
print(test['datetime'].min(),test['datetime'].max())
# 7     0.000000
# 8     0.107774
# 9     0.106327
# 10    0.105583

# 7          11
# 8     3674871
# 9     3743690
# 10    3958109
# 11    3653592

train['days'] = train['datetime'].dt.day
test['days'] = test['datetime'].dt.day

train['flag'] = train['days']
test['flag'] = 11

# 8 9 10 11
data = pd.concat([train,test],axis=0,sort=False)
del train,test


# 小时信息
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute

# 缺失值填充
data['guid'] = data['guid'].fillna('abc')

# 构造历史特征 分别统计前一天 guid deviceid 的相关信息
# 8 9 10 11
history_9 = data[data['days']==8]
history_10 = data[data['days']==9]
history_11 = data[data['days']==10]
history_12 = data[data['days']==11]
del data
# 61326
# 64766
# 66547
# 41933
# 42546
print(len(set(history_9['deviceid'])))
print(len(set(history_10['deviceid'])))
print(len(set(history_11['deviceid'])))
print(len(set(history_12['deviceid'])))
print(len(set(history_9['deviceid'])&set(history_10['deviceid'])))
print(len(set(history_10['deviceid'])&set(history_11['deviceid'])))
print(len(set(history_11['deviceid'])&set(history_12['deviceid'])))

# 61277
# 64284
# 66286
# 41796
# 42347

print(len(set(history_9['guid'])))
print(len(set(history_10['guid'])))
print(len(set(history_11['guid'])))
print(len(set(history_12['guid'])))
print(len(set(history_9['guid'])&set(history_10['guid'])))
print(len(set(history_10['guid'])&set(history_11['guid'])))
print(len(set(history_11['guid'])&set(history_12['guid'])))

# 640066
# 631547
# 658787
# 345742
# 350542

print(len(set(history_9['newsid'])))
print(len(set(history_10['newsid'])))
print(len(set(history_11['newsid'])))
print(len(set(history_12['newsid'])))
print(len(set(history_9['newsid'])&set(history_10['newsid'])))
print(len(set(history_10['newsid'])&set(history_11['newsid'])))
print(len(set(history_11['newsid'])&set(history_12['newsid'])))

# deviceid guid timestamp ts 时间特征
def get_history_visit_time(data1,date2):
    data1 = data1.sort_values(['ts','timestamp'])
    data1['timestamp_ts'] = data1['timestamp'] - data1['ts']
    data1_tmp = data1[data1['target']==1].copy()
    del data1
    for col in ['deviceid','guid']:
        for ts in ['timestamp_ts']:
            f_tmp = data1_tmp.groupby([col],as_index=False)[ts].agg({
                '{}_{}_max'.format(col,ts):'max',
                '{}_{}_mean'.format(col,ts):'mean',
                '{}_{}_min'.format(col,ts):'min',
                '{}_{}_median'.format(col,ts):'median'
            })
        date2 = pd.merge(date2,f_tmp,on=[col],how='left',copy=False)

    return date2

history_10 = get_history_visit_time(history_9,history_10)
history_11 = get_history_visit_time(history_10,history_11)
history_12 = get_history_visit_time(history_11,history_12)

data = pd.concat([history_10,history_11],axis=0,sort=False,ignore_index=True)
data = pd.concat([data,history_12],axis=0,sort=False,ignore_index=True)
del history_9,history_10,history_11,history_12

data = data.sort_values('ts')
data['ts_next'] = data.groupby(['deviceid'])['ts'].shift(-1)
data['ts_next_ts'] = data['ts_next'] - data['ts']

# 当前一天内的特征 leak
for col in [['deviceid'],['guid'],['newsid']]:
    print(col)
    data['{}_days_count'.format('_'.join(col))] = data.groupby(['days'] + col)['id'].transform('count')


# netmodel
data['netmodel'] = data['netmodel'].map({'o':1, 'w':2, 'g4':4, 'g3':3, 'g2':2})

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
    'seed':42,
    'boost_from_average':'false',
    }


feature = [
       'pos','netmodel',  'hour', 'minute',
       'deviceid_timestamp_ts_max', 'deviceid_timestamp_ts_mean',
       'deviceid_timestamp_ts_min', 'deviceid_timestamp_ts_median',
       'guid_timestamp_ts_max', 'guid_timestamp_ts_mean',
       'guid_timestamp_ts_min', 'guid_timestamp_ts_median',
       'deviceid_days_count', 'guid_days_count','newsid_days_count',
        'ts_next_ts'
           ]
target = 'target'


lgb_train = lgb.Dataset(X_train[feature].values, X_train[target].values)
lgb_valid = lgb.Dataset(X_valid[feature].values, X_valid[target].values, reference=lgb_train)
lgb_model = lgb.train(lgb_param, lgb_train, num_boost_round=10000, valid_sets=[lgb_train,lgb_valid],
                      early_stopping_rounds=50,verbose_eval=10)

p_test = lgb_model.predict(X_valid[feature].values,num_iteration=lgb_model.best_iteration)
xx_score = X_valid[[target]].copy()
xx_score['predict'] = p_test
xx_score = xx_score.sort_values('predict',ascending=False)
xx_score = xx_score.reset_index()
xx_score.loc[xx_score.index<=int(xx_score.shape[0]*0.103),'score'] = 1
xx_score['score'] = xx_score['score'].fillna(0)
print(f1_score(xx_score['target'],xx_score['score']))

del lgb_train,lgb_valid
del X_train,X_valid
# 没加 newsid 之前的 f1 score
# 0.5129179717875857
# 0.5197833317587095
# 0.6063125458760602
X_train_2 = data[data['flag'].isin([9,10])]


lgb_train_2 = lgb.Dataset(X_train_2[feature].values, X_train_2[target].values)
lgb_model_2 = lgb.train(lgb_param, lgb_train_2, num_boost_round=lgb_model.best_iteration, valid_sets=[lgb_train_2],verbose_eval=10)

p_predict = lgb_model_2.predict(X_test[feature].values)

submit_score = X_test[['id']].copy()
submit_score['predict'] = p_predict
submit_score = submit_score.sort_values('predict',ascending=False)
submit_score = submit_score.reset_index()
submit_score.loc[submit_score.index<=int(submit_score.shape[0]*0.103),'target'] = 1
submit_score['target'] = submit_score['target'].fillna(0)

submit_score = submit_score.sort_values('id')
submit_score[['id','predict']].to_csv('result/baseline_prob.csv',index=False)
submit_score['target'] = submit_score['target'].astype(int)

sample = pd.read_csv('data/sample.csv')
sample.columns = ['id','non_target']
submit_score = pd.merge(sample,submit_score,on=['id'],how='left')

submit_score[['id','target']].to_csv('result/baseline.csv',index=False)