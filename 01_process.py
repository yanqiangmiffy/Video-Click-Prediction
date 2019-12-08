#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author: quincy qiang 
@license: Apache Licence 
@file: 01_process.py 
@time: 2019/12/08
@software: PyCharm 
"""
import gc
import pickle
import pandas as pd
import numpy as np


def csv2pkl():

    df_train = pd.read_csv('data/train.csv')
    df_train.to_pickle('data/train.pickle')

    df_test = pd.read_csv('data/test.csv')
    df_test.to_pickle('data/test.pickle')

    df_app = pd.read_csv('data/app.csv')
    df_app.to_pickle('data/app.pickle')

    df_user = pd.read_csv('data/user.csv')
    df_user.to_pickle('data/user.pickle')

    df_sample = pd.read_csv('data/sample.csv')
    df_sample.to_csv('data/sample.pickle')
