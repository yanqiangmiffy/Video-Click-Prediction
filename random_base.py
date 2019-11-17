#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: random_base.py 
@time: 2019-11-16 23:52
@description:
"""
import pandas as pd
import random

sample_submission = pd.read_feather('data/sample_submission.feather')

sample_submission['target'] = [random.choice([1, 0]) for i in range(len(sample_submission))]
print(sample_submission['target'].value_counts())
sample_submission.to_csv('result/xgb_result.csv', index=False)
