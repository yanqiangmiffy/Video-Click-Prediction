#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 01_get_user_cluster.py 
@time: 2019-11-22 23:49
@description:
根绝tag标签获取设备聚类
"""
import pandas as pd
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

root = Path('../data/')
user_df = pd.read_csv(root / 'user.csv')


def get_user_articles(user_grouped_df):
    user_df['outertag'] = user_df['outertag'].astype(str)
    grouped_df = user_df.groupby('deviceid').agg({'outertag': '|'.join})
    grouped_df.columns = ['deviceid_' + 'outertag']
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    user_df['tag'] = user_df['tag'].astype(str)
    grouped_df = user_df.groupby('deviceid').agg({'tag': '|'.join})
    grouped_df.columns = ['deviceid_' + 'tag']
    user_grouped_df = pd.merge(user_grouped_df, grouped_df, on='deviceid', how='left')

    # %%
    user_grouped_df['tags'] = user_df['outertag'] + '|' + user_df['tag']
    # %%
    user_grouped_df['text'] = user_grouped_df['tags'].apply(
        lambda tag: " ".join([w.split('_')[0] for w in tag.split('|')]))
    # %%

    # %%
    articles = []
    for text in user_grouped_df['text']:
        articles.append(text)

    return articles


# %%
def transform(articles, n_features=10000000):
    """
    提取tf-idf特征
    :param articles:
    :param n_features:
    :return:
    """
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=2, use_idf=True)
    X = vectorizer.fit_transform(articles)
    return X, vectorizer


def train(X, vectorizer, true_k=10, mini_batch=False, show_label=False):
    """
    训练 k-means
    :param X:
    :param vectorizer:
    :param true_k:
    :param mini_batch:
    :param show_label:
    :return:
    """
    print("training k-means")
    if mini_batch:
        k_means = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                  init_size=1000, batch_size=1000, verbose=False)
    else:
        k_means = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1,
                         verbose=False)
    k_means.fit(X)
    if show_label:  # 显示标签
        print("Top terms per cluster:")
        order_centroids = k_means.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        # print(vectorizer.get_stop_words())
        for i in range(true_k):
            print("Cluster %d" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()
    result = list(k_means.predict(X))
    return result


user_grouped_df = pd.DataFrame({'deviceid': user_df['deviceid'].astype(str).unique()})
articles = get_user_articles(user_grouped_df)
X, vectorizer = transform(articles)

k_cls=[5,9,15,20]
for k in k_cls:
    cluster_pred = train(X, vectorizer, true_k=k, show_label=True)
    user_grouped_df['cluster_{}'.format(k)] = cluster_pred

for k in k_cls:
    cluster_pred = train(X, vectorizer, true_k=k,mini_batch=True,show_label=True)
    user_grouped_df['mini_cluster_{}'.format(k)] = cluster_pred
user_grouped_df.to_csv('01_user_cluster.csv', index=None)
