# !pip install xgboost --user
# !pip install tqdm --user
# !pip install seaborn --user
import pandas as pd
from sklearn.metrics import *
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
from gen_feas import load_data




train, test, no_featuress, features = load_data()
sample_submission = pd.read_feather('data/sample_submission.feather')

n_fold = 10
y_scores = 0
y_pred_l1 = np.zeros([n_fold, test.shape[0]])
y_pred_all_l1 = np.zeros(test.shape[0])

fea_importances = np.zeros(len(features))

label = ['target']
train[label] = train[label].astype(int)
print(train[label])

# [1314, 4590]
kfold = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=1314)
for i, (train_index, valid_index) in enumerate(kfold.split(train[features], train[label])):
    print("n。{}_th fold".format(i))
    X_train, y_train, X_valid, y_valid = train.loc[train_index][features], train[label].loc[train_index], \
                                         train.loc[valid_index][features], train[label].loc[valid_index]
    bst = xgb.XGBClassifier(max_depth=3,
                            n_estimators=100,
                            verbosity=1,
                            learning_rate=0.01,
                            tree_method='gpu_hist'
                            )
    bst.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=['logloss', 'auc'],
            verbose=True,
            early_stopping_rounds=500)
    valid_pred = bst.predict(X_valid)
    print("accuracy:",accuracy_score(y_valid, valid_pred))
    print("f1-score:",f1_score(y_valid, valid_pred))
    y_pred_l1[i] = bst.predict_proba(test[features])[:, 1]
    y_pred_all_l1 += y_pred_l1[i]
    y_scores += bst.best_score

    fea_importances += bst.feature_importances_

r = y_pred_all_l1 / n_fold
sample_submission['target'] = r
sample_submission.to_csv('result/xgb_prob.csv', index=False, sep=",")

sample_submission['target'] = [1 if x > 0.50 else 0 for x in r]
print(sample_submission['target'].value_counts())
sample_submission.to_csv('result/xgb_result.csv', index=False)
