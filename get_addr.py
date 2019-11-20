#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincyqiang
@license: Apache Licence
@file: demo.py
@time: 2019-11-20 22:52
@description:
"""
import math
import pandas as pd
import time
import json
import urllib.request
from tqdm import tqdm


# https://blog.csdn.net/tyt_XiaoTao/article/details/80410279
# 基于百度地图API下的经纬度信息来解析地理位置信息
def getlocation(lat, lng):
    # 31.809928, 102.537467, 3019.300
    # lat = '31.809928'
    # lng = '102.537467'
    url = 'http://api.map.baidu.com/geocoder/v2/?location=' + lat + ',' + lng + '&output=json&pois=1&ak=W5aw4wGTVrrWEbgOLyU9TSZ39KhkfyaV'
    req = urllib.request.urlopen(url)  # json格式的返回数据
    res = req.read().decode("utf-8")  # 将其他编码的字符串解码成unicode
    return json.loads(res)


if __name__ == '__main__':
    train = pd.read_feather('data/train.feather', )
    test = pd.read_feather('data/test.feather')
    df = pd.concat([train, test], sort=False, axis=0)
    df['lat'] = df['lat'].astype(str)
    df['lng'] = df['lng'].astype(str)
    print(df.shape)
    df.dropna(subset=['lat', 'lng'], inplace=True)
    print(df.shape)

    address = []
    print(111)
    for x, y in zip(df['lat'], df['lng']):
        address.append((x, y))
    print(len(address))
    print(len(set(address)))
    address = list(set(address))
    print(address[:10])
    with open('tmp/address.csv', 'a', encoding='utf-8') as f:
        for i in range(len(address)):
            print(i)
            index = address[i]
            lat = str(index[0])
            lng = str(index[1])
            f.write(lng + ',' + lat + ',' + json.dumps(getlocation(lat, lng), ensure_ascii=False))
            f.write('\n')
            time.sleep(0.1)
    # jsonFormat()
