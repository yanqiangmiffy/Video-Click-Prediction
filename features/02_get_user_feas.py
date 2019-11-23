#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 02_get_user_feas.py 
@time: 2019-11-23 00:31
@description:
"""
import pandas as pd
from pathlib import Path

root = Path('../data/')
user_df = pd.read_csv(root / 'user.csv')


print("生成 user 特征....")

user_grouped_df = pd.DataFrame({'deviceid': user_df['deviceid'].unique()})

# 统计一个设备的注册不同用户个数
grouped_df = user_df.groupby(by='deviceid').agg({'guid': ['nunique','count']})
grouped_df.columns = ['deviceid_guid_unique','deviceid_guid_count']
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

user_grouped_df.to_csv('02_user_feas.csv', index=None)
