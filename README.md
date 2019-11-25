# Video-Click-Prediction
视频点击预测大赛

## 内存优化

将csv转为feather格式，加速IO速率

```text
Memory usage of dataframe is 1301.96 MB
Memory usage after optimization is: 543.34 MB
Decreased by 58.3%
Memory usage of dataframe is 362.37 MB
Memory usage after optimization is: 347.11 MB
Decreased by 4.2%
Memory usage of dataframe is 2.46 MB
Memory usage after optimization is: 8.61 MB
Decreased by -249.5%
Memory usage of dataframe is 8.83 MB
Memory usage after optimization is: 16.85 MB
Decreased by -90.9%
```
## gpu
lgb gpu 弃用
xgb gpu 速度很快


## 特征工程

- 转化率特征：过拟合

## 规则发现
- 某些设备出现次数过多

```text
5b02f07eafae65fdbf9760867bcd8856    102979
29078bf9ecff29c67c8f52c997445ee4     41052
3af79e5941776d10da5427bfaa733b15     40925
f4abf0d603045a3403133d25ab0fc60d     32907
457d68dc078349635f3360fdc56d5a31     27315
b89b4b8d9209c77531e7978cad4e088b     26737
32d5f316d9357a3bfed17c3547e5aceb     26624
cbc518e46c68e7cda3aaf6c2898d3b24     23420
fe2745f02d1f287eacb965d218a3e653     21927
5ea2d95b5a2d46a23cb5dacd0271dff7     20360

```

## 参考资料
- [基于特征工程的视频点击率预测算法](http://xblk.ecnu.edu.cn/CN/html/20180309.htm)
- [美图个性化推荐的实践与探索 | 机器之心 ](https://www.jiqizhixin.com/articles/2018-06-27-10)
- [美团DSP广告策略实践 - 美团技术团队](https://tech.meituan.com/2017/05/05/mt-dsp.html)
- [基于机器学习的Web日志异常检测实践 | Wz's Blog ](https://www.wzsite.cn/2018/10/22/%E5%9F%BA%E4%BA%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9A%84Web%E6%97%A5%E5%BF%97%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E5%AE%9E%E8%B7%B5/)
- [推荐系统与特征工程 - 固本培元的专栏 - CSDN博客 ](https://blog.csdn.net/gubenpeiyuan/article/details/80834099)
- [十二种特征工程相关技术简介_ITPUB博客 ](http://blog.itpub.net/29829936/viewspace-2648602/)
- [数据挖掘实践与我的想法之特征工程 - TcD的博客 - CSDN博客  ](https://blog.csdn.net/u011094454/article/details/78572417)
- [主流CTR预估模型的演化及对比 ](https://zhuanlan.zhihu.com/p/35465875)
- [如何解决神经网络训练时loss不下降的问题](https://blog.ailemon.me/2019/02/26/solution-to-loss-doesnt-drop-in-nn-train/)
- [神经网络不收敛的11个常见问题](https://zhuanlan.zhihu.com/p/36369878)