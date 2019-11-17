#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 04_tune_threshold.py 
@time: 2019-11-17 11:49
@description:
"""
import pandas as pd

xgb_prob=pd.read_csv('result/xgb_prob.csv')

threshold=0.45
xgb_prob['target'] = xgb_prob['target'].apply(lambda x:1 if x>threshold else 0)
print(xgb_prob['target'].value_counts())
xgb_prob.to_csv('result/xgb_result_{}.csv'.format(threshold), index=False)