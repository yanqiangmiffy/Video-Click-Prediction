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
from argparse import ArgumentParser
if __name__ == '__main__':
    parser=ArgumentParser()
    parser.add_argument('threshold',type=float)
    args = parser.parse_args()

    print(args.threshold)
    threshold=float(args.threshold)
    # xgb_prob=pd.read_csv('result/xgb_prob.csv')
    # xgb_prob=pd.read_csv('result/NN_EntityEmbed_10fold-sub.csv')
    xgb_prob = pd.read_csv('result/lgb_prob.csv')[['id', 'target']]

    lgb_prob = pd.read_csv('result/lgb_prob.csv')
    # xgb_prob = pd.read_csv('result/xgb_prob.csv')
    # xgb_prob = lgb_prob
    entity = pd.read_csv('result/NN_EntityEmbed_10fold-sub.csv')

    # xgb_prob['target'] = lgb_prob['target'] * 0.7 + xgb_prob['target'] * 0.3
    # xgb_prob['target'] = lgb_prob['target'] * 0.4 + entity['target'] * 0.3 + xgb_prob['target'] * 0.3
    xgb_prob['target'] = lgb_prob['target'] *0.7 + entity['target'] * 0.3
    # xgb_prob['target'] = lgb_prob['target']

    # threshold=0.45
    xgb_prob['target'] = xgb_prob['target'].apply(lambda x:1 if x>threshold else 0)
    print(xgb_prob['target'].value_counts())
    xgb_prob.to_csv('result/xgb_result_{}.csv'.format(threshold), index=False)