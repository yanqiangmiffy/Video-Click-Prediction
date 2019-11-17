#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: fastload.py 
@time: 2019-11-16 10:45
@description:
"""
import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from utils import *


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


# Read data...
root = './data'

train_df = pd.read_csv(os.path.join(root, 'train.csv'))
app_df = pd.read_csv(os.path.join(root, 'app.csv'))
user_df = pd.read_csv(os.path.join(root, 'user.csv'))
test_df = pd.read_csv(os.path.join(root, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(root, 'sample.csv'))

train_df['ts'] = pd.to_datetime(train_df['ts'])
test_df['ts'] = pd.to_datetime(test_df['ts'])

# train_df['ts'] = pd.to_datetime(train_df['ts']).dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
# test_df['ts'] = pd.to_datetime(test_df['ts']).dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')

# train_df['ts'] = pd.to_datetime(train_df['ts'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
# test_df['ts'] = pd.to_datetime(test_df['ts'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')

# train_df['ts'] = pd.to_datetime(train_df['ts'],utc=True).tz_convert("Asia/Shanghai")
# test_df['ts'] = pd.to_datetime(test_df['ts'],utc=True).tz_convert("Asia/Shanghai")

# dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
reduce_mem_usage(train_df)
reduce_mem_usage(test_df)
reduce_mem_usage(app_df)
reduce_mem_usage(user_df)

train_df.to_feather(os.path.join(root, 'train.feather'))
test_df.to_feather(os.path.join(root, 'test.feather'))
app_df.to_feather(os.path.join(root, 'app.feather'))
user_df.to_feather(os.path.join(root, 'user.feather'))
sample_submission.to_feather(os.path.join(root, 'sample_submission.feather'))
