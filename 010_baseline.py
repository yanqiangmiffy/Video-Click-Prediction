import pandas as pd
import numpy as np
import time, datetime
import lightgbm as lgb
from sklearn.metrics import f1_score
from gen_feas import load_data

X_train,X_train_2, X_valid, X_test, no_features, features= load_data()
print(X_train)
print(X_valid)
print(features)

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
submit_score[['id', 'predict']].to_csv('result/baseline_prob.csv', index=False)
submit_score['target'] = submit_score['target'].astype(int)

sample = pd.read_csv('data/sample.csv')
sample.columns = ['id', 'non_target']
submit_score = pd.merge(sample, submit_score, on=['id'], how='left')

submit_score[['id', 'target']].to_csv('result/baseline.csv', index=False)
