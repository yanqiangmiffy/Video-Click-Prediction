# !pip install xgboost --user
# !pip install tqdm --user
# !pip install seaborn --user
import pandas as pd
from sklearn.metrics import *
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
from gen_feas import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import gc


train, test, no_featuress, features = load_data()
sample_submission = pd.read_feather('data/sample_submission.feather')

n_fold = 5
y_scores = 0
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
                            n_estimators=10000,
                            verbosity=1,
                            learning_rate=0.2,
                            tree_method='gpu_hist'
                            )
    bst.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=['logloss', 'auc'],
            verbose=True,
            early_stopping_rounds=500)
    valid_pred = bst.predict(X_valid)
    # print("accuracy:",accuracy_score(y_valid, valid_pred))
    print("f1-score:",f1_score(y_valid, valid_pred))
    y_pred_all_l1 += bst.predict_proba(test[features])[:, 1]
    y_scores += bst.best_score

    fea_importances += bst.feature_importances_
    del bst
    del valid_pred
    gc.collect()

r = y_pred_all_l1 / n_fold
sample_submission['target'] = r
sample_submission.to_csv('result/xgb_prob.csv', index=False, sep=",")

sample_submission['target'] = [1 if x > 0.50 else 0 for x in r]
print(sample_submission['target'].value_counts())
sample_submission.to_csv('result/xgb_result.csv', index=False)


fea_importance_df=pd.DataFrame({
    'features':features,
    'importance':fea_importances
})

plt.figure(figsize=(14, 30))
sns.barplot(x="importance", y="features", data=fea_importance_df.sort_values(by="importance", ascending=False))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
plt.show()

