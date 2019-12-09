# !pip install xgboost --user
# !pip install tqdm --user
# !pip install seaborn --user
import pandas as pd
from sklearn.metrics import *
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from gen_feas import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from utils import *
from tqdm import tqdm
import types

X_train, X_train_2, X_valid_hist, X_test, no_features, features = load_data()
train = X_train_2
test = X_test
sample_submission = pd.read_csv('data/sample.csv')
print(features)

label = ['target']
n_fold = 5
y_scores = 0
test_size = test.shape[0]

y_pred_all_l1 = np.zeros(test_size)
fea_importances = np.zeros(len(features))
train[label] = train[label].astype(int)
cv_score = []
hist_cv_score = []


def pred(X_test, model, batch_size=10000):
    iterations = (X_test.shape[0] + batch_size - 1) // batch_size
    print('iterations', iterations)

    y_test_pred_total = np.zeros(test_size)
    print(f'predicting {i}-th model')
    for k in tqdm(range(iterations)):
        y_pred_test = model.predict_proba(X_test[k * batch_size:(k + 1) * batch_size])[:, 1]
        y_test_pred_total[k * batch_size:(k + 1) * batch_size] += y_pred_test
    return y_test_pred_total


kfold = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=1314)
for i, (train_index, valid_index) in enumerate(kfold.split(train[features], train[label])):
    print("n。{}_th fold".format(i))
    X_train, y_train, X_valid, y_valid = train.loc[train_index][features].values, train[label].loc[train_index].values, \
                                         train.loc[valid_index][features].values, train[label].loc[valid_index].values
    bst = lgb.LGBMClassifier(boosting_type='gbdt',
                             num_leaves=1000,
                             max_depth=-1,
                             learning_rate=0.1,
                             n_estimators=40000,
                             n_jobs=-1,
                             feature_fraction=0.6,
                             bagging_fraction=0.8,
                             bagging_freq=5,
                             seed=2019,
                             )

    bst.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=['logloss', 'auc'],
            verbose=True,
            early_stopping_rounds=50)
    valid_pred = bst.predict(X_valid)
    # print("accuracy:",accuracy_score(y_valid, valid_pred))
    print("训练验证集f1-score:", f1_score(y_valid, valid_pred))
    y_pred_all_l1 += pred(test[features].values, bst)
    cv_score.append(f1_score(y_valid, valid_pred))
    # 历史验证集
    p_test = bst.predict_proba(X_valid_hist[features].values)[:, 1]
    xx_score = X_valid_hist[['target']].copy()
    xx_score['predict'] = p_test
    xx_score = xx_score.sort_values('predict', ascending=False)
    xx_score = xx_score.reset_index()
    xx_score.loc[xx_score.index <= int(xx_score.shape[0] * 0.103), 'score'] = 1
    xx_score['score'] = xx_score['score'].fillna(0)
    print("历史验证集 f1_score", f1_score(xx_score['target'], xx_score['score']))
    hist_cv_score.append(f1_score(xx_score['target'], xx_score['score']))

    fea_importances += bst.feature_importances_
    del bst
    del valid_pred
    gc.collect()

print("cv_score", cv_score)
print("hist_cv_score", hist_cv_score)
# 训练完成 发送邮件
mail("lgb cpu 训练完成，cv f1-score:{}".format(np.mean(cv_score)))

fea_importance_df = pd.DataFrame({
    'features': features,
    'importance': fea_importances / kfold.n_splits
})
fea_importance_df.sort_values(by="importance", ascending=False).to_csv('tmp/lgb_fea_importance.csv', index=None)

r = y_pred_all_l1 / n_fold

submit_score = X_test[['id']].copy()
submit_score['predict'] = r
submit_score = submit_score.sort_values('predict', ascending=False)
submit_score = submit_score.reset_index()
submit_score.loc[submit_score.index <= int(submit_score.shape[0] * 0.103), 'target'] = 1
submit_score['target'] = submit_score['target'].fillna(0)

submit_score = submit_score.sort_values('id')
submit_score['target'] = submit_score['target'].astype(int)
sample = pd.read_csv('data/sample.csv')
sample.columns = ['id', 'non_target']
submit_score = pd.merge(sample, submit_score, on=['id'], how='left')
submit_score[['id', 'target']].to_csv('result/baseline{}.csv'.format(1), index=False)
