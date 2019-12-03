import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import *
from numpy import random
from pathlib import Path
import gc
import datetime
from sklearn.utils import shuffle
from utils import *
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

start_time = time.time()


def get_time_str(x):
    dateArray = datetime.datetime.utcfromtimestamp(x)
    otherStyleTime = dateArray.strftime('%Y-%m-%d %H:%M:%S')
    return otherStyleTime


# 辅助函数
def statics():
    stats = []
    for col in train_df.columns:
        stats.append((col, train_df[col].nunique(), train_df[col].isnull().sum() * 100 / train_df.shape[0],
                      train_df[col].value_counts(normalize=True, dropna=False).values[0] * 100, train_df[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                            'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('Unique_values', ascending=False, inplace=True)
    stats_df.to_excel('tmp/stats_train.xlsx', index=None)


# 加载数据
root = Path('../data/')
train_df = pd.read_csv(root / 'train.csv')
print("train_df.shape", train_df.shape)

train_df['target'] = train_df['target'].astype(int)
test_df = pd.read_csv(root / 'test.csv')
test_df['target'] = 0
train_df['ts'] = train_df['ts'].apply(lambda x: get_time_str(x / 1000))
test_df['ts'] = test_df['ts'].apply(lambda x: get_time_str(x / 1000))
train_df['ts'] = pd.to_datetime(train_df['ts'])
test_df['ts'] = pd.to_datetime(test_df['ts'])
# train_df[['deviceid', 'target', 'newsid', 'ts']].to_csv('train_ts.csv', index=None)
# train_df[['deviceid', 'target', 'newsid', 'ts']].to_csv('test_ts.csv', index=None)


def preprocess_ts(df):
    """
    时间特征
    :param df:
    :return:
    """
    df["day"] = df["ts"].dt.day
    df["hour"] = df["ts"].dt.hour
    df["minute"] = df["ts"].dt.minute

    # df["weekend"] = df["ts"].dt.weekday
    # df["month"] = df["ts"].dt.month
    df["dayofweek"] = df["ts"].dt.dayofweek


preprocess_ts(train_df)
train_df['day_diff'] = np.sign(train_df[['day']].diff().fillna(0))
train_df['hour_diff'] = np.sign(train_df[['hour']].diff().fillna(0))
train_df['minute_diff'] = np.sign(train_df[['minute']].diff().fillna(0))
train_df[['deviceid', 'target', 'newsid', 'ts','day_diff','hour_diff','minute_diff']].head(100).to_csv('train_head.csv',index=None)
