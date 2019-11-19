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

    # def get_same_tag_score(x, y):
    #     x = str(x)
    #     y = str(y)
    #
    #     if '|' not in x or '|' not in y:
    #         return 0
    #
    #     x = x.split('|')
    #     x = [i.split(':')[0] for i in x]
    #     x_d = [{i.split(':')[0]:i.split(':')[1]} for i in x]
    #
    #     y = y.split('|')
    #     y = [i.split(':')[0] for i in y]
    #     y_d = [{i.split(':')[0]:i.split(':')[1]} for i in y]
    #
    #     same = list(set(x).intersection(set(y)))
    #     if len(same) == 0:
    #         return 0
    #     score = 0
    #     for i in same:
    #         score += float(x_d[i])
    #         score += float(y_d[i])
    #
    #     return len(set(x).intersection(set(y)))

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
    #
    #
    train = df[df['target'] != 'test']
    test = df[df['target'] == 'test']


    train = train[train['target'].notnull()].reset_index(drop=True)
    print(train.shape)
    return train, test, no_features


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

        # importance = pd.DataFrame({
        #     'column': features,
        #     'importance': model.feature_importance(),
        # }).sort_values(by='importance')
        # important.append(model.feature_importance())

        # plt.figure(figsize=(12,6))
        # lgb.plot_importance(model, max_num_features=300)
        # plt.title("Featurertances_%s" % index)
        # plt.show()

        vali_pre = model.predict(X_test, num_iteration=model.best_iteration)
        print(np.array(y_test))
        print(np.array(vali_pre))

        score = roc_auc_score(list(y_test), list(vali_pre))
        score_list.append(score)
        print('AUC:', score)

        # if index == 1:
        #     continue

        pred_result = model.predict(test, num_iteration=model.best_iteration)
        pred_cv.append(pred_result)

    res = np.array(pred_cv)
    r = res.mean(axis=0)

    # importance_all = pd.DataFrame({
    #     'column': features,
    #     'importance': np.array(important).mean(axis=0),
    # }).sort_values(by='importance')
    #
    # print(importance_all)

    print("总的结果：", res.shape)

    print(score_list)
    print(np.mean(score_list))

    return r

def lgb_para_binary_model(X_train, y_train, X_test, y_test):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'max_depth': 4,
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 16,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
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

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
user = pd.read_csv('../data/user.csv')
app = pd.read_csv('../data/app.csv')
sample = pd.read_csv('../data/sample.csv')

# train = pd.merge(train, app, on='deviceid', how='left')
# get_fea(train, test, user, app)
train, test, no_features = get_fea(train, test, user, app)
print('get_fea ok')

label = train['target']
sub = test[['id']]

features = [fea for fea in train.columns if fea not in no_features]

train_df = train[features]

test_df = test[features]

del train
del test
gc.collect()


print(train_df.head())

r = get_result(train_df, test_df, label, lgb_para_binary_model, splits_nums=5)
sub['target'] = r

r_c = sorted(r, reverse=True)
cut = r_c[400000]

sub['target'] = sub['target'].apply(lambda x: 0 if x <= cut else 1)
sub[['id', 'target']].to_csv('../result/submission_lgb.csv', index=None)
print(sub['target'].value_counts())



