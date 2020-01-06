import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import entropy, kurtosis
import time
import gc

pd.set_option('display.max_columns', None)


def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


print('=============================================== read train ===============================================')
t = time.time()
train_df = pd.read_csv('data/train.csv')
train_df['date'] = pd.to_datetime(
    train_df['ts'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
train_df['day'] = train_df['date'].dt.day
train_df.loc[train_df['day'] == 7, 'day'] = 8
train_df['hour'] = train_df['date'].dt.hour
train_df['minute'] = train_df['date'].dt.minute
train_num = train_df.shape[0]
labels = train_df['target'].values
print('runtime:', time.time() - t)

print('=============================================== click data ===============================================')
click_df = train_df[train_df['target'] == 1].sort_values('timestamp').reset_index(drop=True)
click_df['exposure_click_gap'] = click_df['timestamp'] - click_df['ts']
click_df = click_df[click_df['exposure_click_gap'] >= 0].reset_index(drop=True)
click_df['date'] = pd.to_datetime(
    click_df['timestamp'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
click_df['day'] = click_df['date'].dt.day
click_df.loc[click_df['day'] == 7, 'day'] = 8
del train_df['target'], train_df['timestamp']
for f in ['date', 'exposure_click_gap', 'timestamp', 'ts', 'target', 'hour', 'minute']:
    del click_df[f]
print('runtime:', time.time() - t)

print('=============================================== read test ===============================================')
test_df = pd.read_csv('data/test.csv')
test_df['date'] = pd.to_datetime(
    test_df['ts'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
test_df['day'] = test_df['date'].dt.day
test_df.loc[test_df['day'] == 10, 'day'] = 11
test_df['hour'] = test_df['date'].dt.hour
test_df['minute'] = test_df['date'].dt.minute
df = pd.concat([train_df, test_df], axis=0, ignore_index=False)
del train_df, test_df, df['date']
gc.collect()
print('runtime:', time.time() - t)

print('=============================================== cate enc ===============================================')
df['lng_lat'] = df['lng'].astype('str') + '_' + df['lat'].astype('str')
del df['guid']
click_df['lng_lat'] = click_df['lng'].astype('str') + '_' + click_df['lat'].astype('str')
sort_df = df.sort_values('ts').reset_index(drop=True)
cate_cols = [
    'deviceid', 'newsid', 'pos', 'app_version', 'device_vendor',
    'netmodel', 'osversion', 'device_version', 'lng', 'lat', 'lng_lat'
]
for f in cate_cols:
    print(f)
    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
    df[f] = df[f].map(map_dict).fillna(-1).astype('int32')
    click_df[f] = click_df[f].map(map_dict).fillna(-1).astype('int32')
    sort_df[f] = sort_df[f].map(map_dict).fillna(-1).astype('int32')
    df[f + '_count'] = df[f].map(df[f].value_counts())
df = reduce_mem(df)
click_df = reduce_mem(click_df)
sort_df = reduce_mem(sort_df)
print('runtime:', time.time() - t)

print('=============================================== feat eng ===============================================')

print('*************************** history stats ***************************')
for f in [
    ['deviceid'],
    ['pos', 'deviceid'],
    # ...
]:
    print('------------------ {} ------------------'.format('_'.join(f)))

    # 对前一天的点击次数进行统计
    tmp = click_df[f + ['day', 'id']].groupby(f + ['day'], as_index=False)['id'].agg(
        {'_'.join(f) + '_prev_day_click_count': 'count'})
    tmp['day'] += 1
    df = df.merge(tmp, on=f + ['day'], how='left')
    df['_'.join(f) + '_prev_day_click_count'] = df['_'.join(f) + '_prev_day_click_count'].fillna(0)
    df.loc[df['day'] == 8, '_'.join(f) + '_prev_day_click_count'] = None

    # 对前一天的曝光量进行统计
    tmp = df[f + ['day', 'id']].groupby(f + ['day'], as_index=False)['id'].agg(
        {'_'.join(f) + '_prev_day_count': 'count'})
    tmp['day'] += 1
    df = df.merge(tmp, on=f + ['day'], how='left')
    df['_'.join(f) + '_prev_day_count'] = df['_'.join(f) + '_prev_day_count'].fillna(0)
    df.loc[df['day'] == 8, '_'.join(f) + '_prev_day_count'] = None

    # 计算前一天的点击率
    df['_'.join(f) + '_prev_day_ctr'] = df['_'.join(f) + '_prev_day_click_count'] / (
            df['_'.join(f) + '_prev_day_count'] + df['_'.join(f) + '_prev_day_count'].mean())

    del tmp
    print('runtime:', time.time() - t)
del click_df
df = reduce_mem(df)

print('*************************** exposure_ts_gap ***************************')
for f in [
    ['deviceid'], ['newsid'], ['lng_lat'],
    ['pos', 'deviceid'], ['pos', 'newsid'], ['pos', 'lng_lat'],
    ['pos', 'deviceid', 'lng_lat'],
    ['netmodel', 'deviceid'],
    ['pos', 'netmodel', 'deviceid'],
    # 这些特征或者特征组合基本上按照特征重要性随便选择的，也比较佛系，在这个地方继续改造继续拓展，提升空间很大。
]:
    print('------------------ {} ------------------'.format('_'.join(f)))

    tmp = sort_df[f + ['ts']].groupby(f)
    # 前x次、后x次曝光到当前的时间差
    for gap in [1, 2, 3, 5, 10]:
        sort_df['{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap)] = tmp['ts'].shift(0) - tmp['ts'].shift(gap)
        sort_df['{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)] = tmp['ts'].shift(-gap) - tmp['ts'].shift(0)
        tmp2 = sort_df[
            f + ['ts', '{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap),
                 '{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)]
            ].drop_duplicates(f + ['ts']).reset_index(drop=True)
        df = df.merge(tmp2, on=f + ['ts'], how='left')
        del sort_df['{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap)]
        del sort_df['{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)]
        del tmp2

    del tmp
    df = reduce_mem(df)
    print('runtime:', time.time() - t)
del sort_df, df['ts']
gc.collect()

print('*************************** cross feat (second order) ***************************')
# 二阶交叉特征，可以继续做更高阶的交叉。
cross_cols = ['deviceid', 'newsid', 'pos', 'netmodel', 'lng_lat']  # 这些也是按特征重要性随缘选择，继续改造继续拓展，还能有所提升。
for f in cross_cols:
    for col in cross_cols:
        if col == f:
            continue
        print('------------------ {} {} ------------------'.format(f, col))
        df = df.merge(df[[f, col]].groupby(f, as_index=False)[col].agg({
            'cross_{}_{}_nunique'.format(f, col): 'nunique',
            'cross_{}_{}_ent'.format(f, col): lambda x: entropy(x.value_counts() / x.shape[0])  # 熵
        }), on=f, how='left')
        if 'cross_{}_{}_count'.format(f, col) not in df.columns.values and 'cross_{}_{}_count'.format(col,
                                                                                                      f) not in df.columns.values:
            df = df.merge(df[[f, col, 'id']].groupby([f, col], as_index=False)['id'].agg({
                'cross_{}_{}_count'.format(f, col): 'count'  # 共现次数
            }), on=[f, col], how='left')
        if 'cross_{}_{}_count_ratio'.format(col, f) not in df.columns.values:
            df['cross_{}_{}_count_ratio'.format(col, f)] = df['cross_{}_{}_count'.format(f, col)] / df[
                f + '_count']  # 比例偏好
        if 'cross_{}_{}_count_ratio'.format(f, col) not in df.columns.values:
            df['cross_{}_{}_count_ratio'.format(f, col)] = df['cross_{}_{}_count'.format(f, col)] / df[
                col + '_count']  # 比例偏好
        df['cross_{}_{}_nunique_ratio_{}_count'.format(f, col, f)] = df['cross_{}_{}_nunique'.format(f, col)] / df[
            f + '_count']
        print('runtime:', time.time() - t)
    df = reduce_mem(df)
del df['id']
gc.collect()

print('***************************添加app特征*******************************')
app_df = pd.read_csv('data/app.csv')
app_grouped_df = pd.DataFrame({'deviceid': app_df['deviceid'].unique()})

# 统计一个设备的出现过的app总数
app_df['app_nums'] = app_df['applist'].apply(lambda x: len(x.replace('[', '').replace(']', '').split(' ')))
app_df.app_nums.head()

grouped_df = app_df.groupby(by='deviceid').agg({'app_nums': ['sum']})
grouped_df.columns = ['app_nums_sum']
grouped_df = grouped_df.reset_index()
app_grouped_df = pd.merge(app_grouped_df, grouped_df, on='deviceid', how='left')

# 统计一个设备上applist对应的不同device个数总数
app_df['applist_count'] = app_df.groupby('applist')['deviceid'].transform('count')
grouped_df = app_df.groupby(by='deviceid').agg({'applist_count': ['sum']})
grouped_df.columns = ['applist_count_sum']
grouped_df = grouped_df.reset_index()
app_grouped_df = pd.merge(app_grouped_df, grouped_df, on='deviceid', how='left')

df = pd.merge(df, app_grouped_df, on='deviceid', how='left')
del app_grouped_df,grouped_df
del app_df
gc.collect()

print('***************************添加user*******************************')
user_df = pd.read_csv('data/user.csv')
user_grouped_df = pd.DataFrame({'deviceid': user_df['deviceid'].unique()})
# 统计一个设备的注册不同用户个数
grouped_df = user_df.groupby(by='deviceid').agg({'guid': ['nunique', 'count']})
grouped_df.columns = ['deviceid_guid_unique', 'deviceid_guid_count']
grouped_df = grouped_df.reset_index()
user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')


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


user_df['outertag_nums'] = user_df['outertag'].astype('str').apply(lambda x: get_outertag_nums(x))
user_df['outertag_score'] = user_df['outertag'].astype('str').apply(lambda x: get_outertag_score(x))

user_df['tag_nums'] = user_df['tag'].astype('str').apply(lambda x: get_outertag_nums(x))
user_df['tag_score'] = user_df['tag'].astype('str').apply(lambda x: get_outertag_score(x))

grouped_df = user_df.groupby(by='deviceid').agg({'outertag_nums': ['sum']})
grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
grouped_df.reset_index()
user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

grouped_df = user_df.groupby(by='deviceid').agg({'outertag_score': ['sum']})
grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
grouped_df.reset_index()
user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

grouped_df = user_df.groupby(by='deviceid').agg({'tag_nums': ['sum']})
grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
grouped_df.reset_index()
user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

grouped_df = user_df.groupby(by='deviceid').agg({'tag_score': ['sum']})
grouped_df.columns = ['deviceid_' + '_'.join(col).strip() for col in grouped_df.columns.values]
grouped_df.reset_index()
user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')
#
# 设备的用户等级统计
grouped_df = user_df.groupby(by='deviceid').agg({'level': ['sum']})
grouped_df.columns = ['deviceid_level_sum']
grouped_df.reset_index()
user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

# 设备的用户劣质统计
# 1表示劣质用户 0表示正常用户。
grouped_df = user_df.groupby(by='deviceid').agg({'personidentification': ['sum']})
grouped_df.columns = ['deviceid_personidentification_sum']
grouped_df.reset_index()
user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

grouped_df = user_df.groupby(by='deviceid').agg({'personalscore': ['sum']})
grouped_df.columns = ['deviceid_personalscore_sum']
grouped_df.reset_index()
user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')
#
grouped_df = user_df.groupby(by='deviceid').agg({'followscore': ['sum']})
grouped_df.columns = ['deviceid_followscore_sum']
grouped_df.reset_index()
user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

df = pd.merge(df, user_grouped_df, on='deviceid', how='left')
del user_grouped_df,grouped_df
del user_df
gc.collect()

print('========================================================================================================')
train_df = df[:train_num].reset_index(drop=True)
train_df['label'] = labels
train_df = train_df[train_df['day'] > 8].reset_index(drop=True)
labels = train_df['label'].values
test_df = df[train_num:].reset_index(drop=True)
del df, train_df['label']
gc.collect()

train_idx = train_df[train_df['day'] < 10].index.tolist()
val_idx = train_df[train_df['day'] == 10].index.tolist()

train_x = train_df.iloc[train_idx].reset_index(drop=True)
train_y = labels[train_idx]
val_x = train_df.iloc[val_idx].reset_index(drop=True)
val_y = labels[val_idx]

del train_x['day'], val_x['day'], train_df['day'], test_df['day']
gc.collect()
print('runtime:', time.time() - t)
print('========================================================================================================')

print(
    '=============================================== training validate ===============================================')
fea_imp_list = []
clf = LGBMClassifier(
    learning_rate=0.01,
    n_estimators=20000,
    num_leaves=255,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=2019,
    metric=None
)

print('************** training **************')
clf.fit(
    train_x, train_y,
    eval_set=[(val_x, val_y)],
    eval_metric='auc',
    categorical_feature=cate_cols,
    early_stopping_rounds=200,
    verbose=50
)
print('runtime:', time.time() - t)

print('************** validate predict **************')
best_rounds = clf.best_iteration_
best_auc = clf.best_score_['valid_0']['auc']
val_pred = clf.predict_proba(val_x)[:, 1]
fea_imp_list.append(clf.feature_importances_)
print('runtime:', time.time() - t)

print(
    '=============================================== training predict ===============================================')
clf = LGBMClassifier(
    learning_rate=0.01,
    n_estimators=best_rounds,
    num_leaves=255,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=2019
)

print('************** training **************')
clf.fit(
    train_df, labels,
    eval_set=[(train_df, labels)],
    eval_metric='auc',
    categorical_feature=cate_cols,
    early_stopping_rounds=200,
    verbose=50
)
print('runtime:', time.time() - t)

print('************** test predict **************')
sub = pd.read_csv('data/sample.csv')
sub['target'] = clf.predict_proba(test_df)[:, 1]
fea_imp_list.append(clf.feature_importances_)
print('runtime:', time.time() - t)

print(
    '=============================================== feat importances ===============================================')
# 特征重要性可以好好看看
fea_imp_dict = dict(zip(train_df.columns.values, np.mean(fea_imp_list, axis=0)))
fea_imp_item = sorted(fea_imp_dict.items(), key=lambda x: x[1], reverse=True)
for f, imp in fea_imp_item:
    print('{} = {}'.format(f, imp))

print(
    '=============================================== threshold search ===============================================')
# f1阈值敏感，所以对阈值做一个简单的迭代搜索。
t0 = 0.05
v = 0.002
best_t = t0
best_f1 = 0
for step in range(201):
    curr_t = t0 + step * v
    y = [1 if x >= curr_t else 0 for x in val_pred]
    curr_f1 = f1_score(val_y, y)
    if curr_f1 > best_f1:
        best_t = curr_t
        best_f1 = curr_f1
        print('step: {}   best threshold: {}   best f1: {}'.format(step, best_t, best_f1))
print('search finish.')

val_pred = [1 if x >= best_t else 0 for x in val_pred]
print('\nbest auc:', best_auc)
print('best f1:', f1_score(val_y, val_pred))
print('validate mean:', np.mean(val_pred))
print('runtime:', time.time() - t)

print('=============================================== sub save ===============================================')
sub.to_csv('sub_prob_{}_{}_{}.csv'.format(best_auc, best_f1, sub['target'].mean()), index=False)
sub['target'] = sub['target'].apply(lambda x: 1 if x >= best_t else 0)
sub.to_csv('sub_{}_{}_{}.csv'.format(best_auc, best_f1, sub['target'].mean()), index=False)
print('runtime:', time.time() - t)
print('finish.')
print('========================================================================================================')
