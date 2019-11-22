#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 06_en.py 
@time: 2019-11-21 23:19
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
    lgb_prob = pd.read_csv('result/lgb_prob.csv')
    xgb_prob = pd.read_csv('result/xgb_prob.csv')
    entity=pd.read_csv('result/NN_EntityEmbed_10fold-sub.csv')


    xgb_prob['target'] = lgb_prob['target'] * 0.7 + entity['target'] * 0.1+xgb_prob['target']*0.2

    # xgb_prob = pd.read_csv('result/lgb_prob.csv')[['id', 'target']]

    # threshold=0.45
    # xgb_prob['target'] = xgb_prob['target'].apply(lambda x:1 if x>threshold else 0)
    # print(xgb_prob['target'].value_counts())
    # xgb_prob.to_csv('result/xgb_result_{}.csv'.format(threshold), index=False)

    submit = xgb_prob[['id']]
    submit['target'] = xgb_prob['target']

    submit['target'] = submit['target'].rank()
    # submit['target'] = (submit['target'] >= submit.shape[0] * 0.8934642948637943).astype(int)
    submit['target'] = (submit['target'] >= submit.shape[0] * 0.90).astype(int)
    print(submit['target'].value_counts())
    submit.to_csv("result/en.csv", index=False)

