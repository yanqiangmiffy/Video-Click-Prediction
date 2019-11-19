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
from utils import *
from tqdm import tqdm

train, test, no_featuress, features = load_data()
sample_submission = pd.read_feather('data/sample_submission.feather')

n_fold = 5
y_scores = 0
test_size=test.shape[0]
y_pred_all_l1 = np.zeros(test_size)

fea_importances = np.zeros(len(features))

label = ['target']
train[label] = train[label].astype(int)
print(train[label])


def pred(X_test, model, batch_size=100000):
    iterations = (X_test.shape[0] + batch_size - 1) // batch_size
    print('iterations', iterations)

    y_test_pred_total = np.zeros(test_size)
    print(f'predicting {i}-th model')
    for k in tqdm(range(iterations)):
        y_pred_test = model.predict_proba(X_test[k*batch_size:(k+1)*batch_size])[:, 1]
        y_test_pred_total[k * batch_size:(k + 1) * batch_size] += y_pred_test
    return y_test_pred_total

# [1314, 4590]
kfold = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=1314)
for i, (train_index, valid_index) in enumerate(kfold.split(train[features], train[label])):
    print("n。{}_th fold".format(i))
    X_train, y_train, X_valid, y_valid = train.loc[train_index][features], train[label].loc[train_index], \
                                         train.loc[valid_index][features], train[label].loc[valid_index]
    bst = xgb.XGBClassifier(max_depth=3,
                            n_estimators=6000,
                            verbosity=1,
                            learning_rate=0.2,
                            tree_method='gpu_hist',
                            n_jobs=-1,
                            )
    bst.fit(X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=['logloss', 'auc'],
            verbose=True,
            early_stopping_rounds=100)
    valid_pred = bst.predict(X_valid)
    # print("accuracy:",accuracy_score(y_valid, valid_pred))
    print("f1-score:", f1_score(y_valid, valid_pred))
    y_pred_all_l1 += pred(test[features],bst)
    y_scores += bst.best_score

    # 训练完成 发送邮件
    # mail(str(i) + "xgb cpu 训练完成，cv f1-score:{}".format(f1_score(y_valid, valid_pred)))

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

fea_importance_df = pd.DataFrame({
    'features': features,
    'importance': fea_importances
})
fea_importance_df.sort_values(by="importance", ascending=False).to_csv('tmp/fea_importance.csv', index=None)

# plt.figure(figsize=(14, 30))
# sns.barplot(x="importance", y="features", data=fea_importance_df.sort_values(by="importance", ascending=False))
# plt.title('Features importance (averaged/folds)')
# plt.tight_layout()
# plt.show()
