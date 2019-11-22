#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 08_deepctr.py 
@time: 2019-11-22 07:55
@description:
"""

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names
from gen_feas_nn import load_data

if __name__ == "__main__":
    X_train,y_train,X_test,embed_cols,col_vals_dict,numerical_features = load_data()

    # data = pd.read_csv('./criteo_sample.txt')
    #
    # sparse_features = ['C' + str(i) for i in range(1, 27)]
    # dense_features = ['I' + str(i) for i in range(1, 14)]
    #
    # data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0, )
    # target = ['label']
    #
    # # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, col_vals_dict[feat])
                           for feat in embed_cols] + [DenseFeat(feat, 1,)
                          for feat in numerical_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train_model_input = {name:X_train[name] for name in feature_names}
    test_model_input = {name:X_test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, y_train.values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )

    pred_ans = model.predict(test_model_input, batch_size=256)
    # print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    # print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))