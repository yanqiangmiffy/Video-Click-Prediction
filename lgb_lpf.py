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
import gc

# from get_feas_lpf import load_data
# from get_feas_lpf_2 import load_data
# from get_feas_lpf_3 import load_data
from get_feas_lpf_4 import load_data
from utils import *


scaler = StandardScaler()

def get_result(train, test, label, my_model, splits_nums=10):
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    label = label.values.astype(int)

    score_list = []
    pred_cv = []
    important = []

    k_fold = StratifiedKFold(n_splits=splits_nums, shuffle=True, random_state=1314)
    for index, (train_index, test_index) in enumerate(k_fold.split(train, label)):

        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = label[train_index], label[test_index]
        model = my_model(X_train, y_train, X_test, y_test)

        vali_pre = model.predict(X_test, num_iteration=model.best_iteration)
        print(np.array(y_test))
        print(np.array(vali_pre))

        score = roc_auc_score(list(y_test), list(vali_pre))
        score_list.append(score)
        print('AUC:', score)

        # 训练完成 发送邮件
        try:
            mail_all(str(index) + "lgb cpu 训练完成，cv AUC:{}".format(score), '17853530715@163.com')
        except:
            print('邮件发送失败')

        pred_result = model.predict(test, num_iteration=model.best_iteration)
        pred_cv.append(pred_result)

    res = np.array(pred_cv)
    r = res.mean(axis=0)

    print("总的结果：", res.shape)

    print(score_list)
    print(np.mean(score_list))

    return r

def lgb_para_binary_model(X_train, y_train, X_test, y_test):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 1000,
        'learning_rate': 0.1,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'random_state': 1024,
        'n_jobs': -1,
    }

    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_test, y_test)
    num_round = 20000
    model = lgb.train(params,
                      trn_data,
                      num_round,
                      valid_sets=[trn_data, val_data],
                      verbose_eval=10,
                      early_stopping_rounds=100)
    return model

train, test, no_features, features = load_data()
print('get_fea ok')

label = train['target']
sub = test[['id']]


train_df = train[features]
del train
gc.collect()

test_df = test[features]
del test
gc.collect()


print(train_df.head())

r = get_result(train_df, test_df, label, lgb_para_binary_model, splits_nums=5)
sub['target'] = r

r_c = sorted(r, reverse=True)
cut = r_c[400000]

sub['target'] = sub['target'].apply(lambda x: 0 if x <= cut else 1)
sub[['id', 'target']].to_csv('submission_lgb.csv', index=None)
print(sub['target'].value_counts())



