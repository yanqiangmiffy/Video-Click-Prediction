#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 03_get_add_feas.py 
@time: 2019-11-23 00:42
@description:
"""
import pandas as pd
from pathlib import Path
root = Path('../data/')

app_df = pd.read_csv(root / 'app.csv')

print("生成 app 特征....")
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

app_grouped_df.to_csv('03_app_feas.csv', index=None)

