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

- 

## 参考资料
- [基于特征工程的视频点击率预测算法](http://xblk.ecnu.edu.cn/CN/html/20180309.htm)
- [美图个性化推荐的实践与探索 | 机器之心 ](https://www.jiqizhixin.com/articles/2018-06-27-10)
- [美团DSP广告策略实践 - 美团技术团队](https://tech.meituan.com/2017/05/05/mt-dsp.html)
- [基于机器学习的Web日志异常检测实践 | Wz's Blog ](https://www.wzsite.cn/2018/10/22/%E5%9F%BA%E4%BA%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9A%84Web%E6%97%A5%E5%BF%97%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E5%AE%9E%E8%B7%B5/)
- [推荐系统与特征工程 - 固本培元的专栏 - CSDN博客 ](https://blog.csdn.net/gubenpeiyuan/article/details/80834099)
- [十二种特征工程相关技术简介_ITPUB博客 ](http://blog.itpub.net/29829936/viewspace-2648602/)
- [数据挖掘实践与我的想法之特征工程 - TcD的博客 - CSDN博客  ](https://blog.csdn.net/u011094454/article/details/78572417)
