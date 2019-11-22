#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 02_entity_embedding.py 
@time: 2019-11-21 19:09
@description:
"""

# https://www.kaggle.com/aquatic/entity-embedding-neural-net

import numpy as np
import pandas as pd
from keras.callbacks import *
from keras.layers import *
from tqdm import tqdm
import gc
# random seeds for stochastic parts of neural network
np.random.seed(10)
from tensorflow import set_random_seed
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

set_random_seed(15)
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
from sklearn.model_selection import StratifiedKFold
import ipykernel
from gen_feas_nn import load_data


class roc_auc_callback(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs=None):
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x, verbose=0)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc'] = roc_auc_score(self.y, y_pred)
        logs['norm_gini'] = (roc_auc_score(self.y, y_pred) * 2) - 1

        y_pred_val = self.model.predict(self.x_val, verbose=0)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)
        logs['norm_gini_val'] = (roc_auc_score(self.y_val, y_pred_val) * 2) - 1

        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (
            str(round(roc, 5)), str(round(roc_val, 5)), str(round((roc * 2 - 1), 5)), str(round((roc_val * 2 - 1), 5))),
              end=10 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return


X_train,y_train,X_test,embed_cols,col_vals_dict,numerical_features = load_data()
submission=pd.read_feather('data/sample_submission.feather')


def build_embedding_network():
    inputs = []
    embeddings = []
    for i in range(len(embed_cols)):
        cate_input = Input(shape=(1,))
        input_dim = len(col_vals_dict[embed_cols[i]])
        if input_dim > 1000:
            output_dim = 50
        else:
            output_dim = (len(col_vals_dict[embed_cols[i]]) // 2) + 1

        embedding = Embedding(input_dim, output_dim, input_length=1)(cate_input)
        embedding = Reshape(target_shape=(output_dim,))(embedding)
        inputs.append(cate_input)
        embeddings.append(embedding)

    input_numeric = Input(shape=(len(numerical_features),))  # 20
    embedding_numeric = Dense(64)(input_numeric)
    inputs.append(input_numeric)
    embeddings.append(embedding_numeric)

    x = Concatenate()(embeddings)
    x = Dense(128, activation='relu')(x)
    x = Dropout(.35)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(.15)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, output)

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


# converting data to list format to match the network structure
def preproc(X_train, X_val, X_test):
    input_list_train = []
    input_list_val = []
    input_list_test = []

    # the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)

    # the rest of the columns
    # other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    other_cols = numerical_features
    print(other_cols, len(other_cols))
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)

    return input_list_train, input_list_val, input_list_test


# gini scoring function from kernel at:
# https://www.kaggle.com/tezdhar/faster-gini-calculation
def ginic(actual, pred):
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_c[-1] - (n + 1) / 2.0
    return giniSum / n


def gini_normalizedc(a, p):
    return ginic(a, p) / ginic(a, a)


def train():
    # network training
    K = 5
    runs_per_fold = 1
    n_epochs = 5

    cv_ginis = []
    full_val_preds = np.zeros(np.shape(X_train)[0])
    y_preds = np.zeros((np.shape(X_test)[0], K))

    kfold = StratifiedKFold(n_splits=K,
                            random_state=231,
                            shuffle=True)

    for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):
        X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
        y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]

        X_test_f = X_test.copy()

        # upsampling adapted from kernel:
        # https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
        # pos = (pd.Series(y_train_f == 1))
        # # Add positive examples
        # X_train_f = pd.concat([X_train_f, X_train_f.loc[pos]], axis=0)
        # y_train_f = pd.concat([y_train_f, y_train_f.loc[pos]], axis=0)

        # Shuffle data
        # idx = np.arange(len(X_train_f))
        # np.random.shuffle(idx)
        # X_train_f = X_train_f.iloc[idx]
        # y_train_f = y_train_f.iloc[idx]

        # preprocessing
        proc_X_train_f, proc_X_val_f, proc_X_test_f = preproc(X_train_f, X_val_f, X_test_f)

        # track oof prediction for cv scores
        val_preds = 0
        # auc_callback = roc_auc_callback(training_data=(proc_X_train_f, y_train_f),
        #                                 validation_data=(proc_X_val_f, y_val_f))
        NN = build_embedding_network()

        NN.summary()
        NN.fit(proc_X_train_f,
               y_train_f.values,
               epochs=n_epochs,
               batch_size=4096,
               verbose=1,
               # callbacks=[auc_callback]
               )

        val_preds += NN.predict(proc_X_val_f)[:, 0] / runs_per_fold
        y_preds[:, i] += NN.predict(proc_X_test_f)[:, 0] / runs_per_fold
        NN.save('tmp/entity{}.hd5'.format(i + 1))

        full_val_preds[outf_ind] += val_preds
        cv_gini = gini_normalizedc(y_val_f.values, val_preds)
        cv_auc = roc_auc_score(y_val_f.values, val_preds)
        cv_ginis.append(cv_gini)
        print('\nFold %i prediction cv gini: %.5f\n' % (i, cv_gini))
        print('\nFold %i prediction cv auc: %.5f\n' % (i, cv_auc))
        del NN
        gc.collect()
    print('Mean out of fold gini: %.5f' % np.mean(cv_ginis))
    print('Full validation gini: %.5f' % gini_normalizedc(y_train.values, full_val_preds))

    y_pred_final = np.mean(y_preds, axis=1)
    print(len(submission.id))
    print(len(y_pred_final))
    df_sub = pd.DataFrame({'id': submission.id,
                           'target': y_pred_final},
                          columns=['id', 'target'])
    df_sub.to_csv('result/NN_EntityEmbed_10fold-sub.csv', index=False)

    pd.DataFrame(full_val_preds).to_csv('result/NN_EntityEmbed_10fold-val_preds.csv', index=False)


train()
