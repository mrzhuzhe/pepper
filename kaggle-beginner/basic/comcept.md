#   数据分析，销售额预测

> 最开始还是要定位问题的类型

## Eda and Preproccess

1. 剔除重复
2. 有些非常相似的增加字段
3. 似乎需要把shop的信息加到item上
4. 提取数据时要用延时提取和滚动框提取


## Basic Concept

|  名称   | 说明  | 链接 |
|  ----  | ----  | ----  |
| 俄罗斯销售预测  | 一个非常详细的eda，特征提取等，还根据一些人工提取方式做了优化 | https://github.com/KubaMichalczyk/kaggle-predict-future-sales/blob/master/notebooks/eda.ipynb |
| 俄罗斯销售预测 | 这里面对延时特征做了详细的说明 | https://www.kaggle.com/szhou42/predict-future-sales-top-11-solution |
| 房价预测 | 一般回归问题的approach | https://www.kaggle.com/c/house-prices-advanced-regression-techniques/discussion/23409 |
| nlp twitter | nlp twitter 的 eda | https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert |
| nlp twitter | 仔细看了一下，还是用了多种模型做了模型融合 | https://www.kaggle.com/nxhong93/tweet-predict1 |
|  鸭子蛇  | 强化学习 | https://www.kaggle.com/c/hungry-geese/discussion/218190  |
|  时序特征  | ----  | 
https://www.kaggle.com/jagangupta/time-series-basics-exploring-traditional-ts |
|  dice系数  | ----  | https://www.aiuai.cn/aifarm1159.html |
|  如何使用tpu  | 目前tpu的配置有点问题  | https://www.kaggle.com/joshi98kishan/foldtraining-pytorch-tpu-8-cores?scriptVersionId=49786435  |
|  其实如果是“连续特征”  | ？使用条件存疑，可以整体用降噪自编码器dae来做特征工程，无缺失值，无分类，数量够  | 详情可见kaggle每月tablur模拟赛  |




## Roadmap

|  待查   | 说明  |
|  ----  | ----  |
| 岭回归  | nlp和eda中都有 |
| eda  | 调查eda的基本方法 |
| openview.org  | 查顶会论文 |
| slidelive.com  | 顶会演讲录像 |
| 查一下py lambda函数 |  |
| 时序问题 | https://otexts.com/fpp2/ |
| 图像分割 | https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/200955 |
| efficientnet的不同 | https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html/ |
| persudo label | https://www.kaggle.com/cdeotte/pseudo-labeling-qda-0-969 / qda: https://www.kaggle.com/c/instant-gratification/discussion/93843|


## Preproccess Checklist

|  项目   | 说明  | 对结果的影响 |
|  ----  | ---- | ---- |
| 缺失值处理  | 检查缺失值是否存在和样本特征的相关性 | ？ |
| 用户手动输入值统一化  | 如果有些值不是选择是用户手动输入，就会存在很多不同输入对应相同值的情况 | ？ |
| 原子特征 | 断句方式 hashtag url 数量等特征 @数 一元词 二元词 | 其实这次不把fake报道算入，这些特征有必要和bert本身放一起么？【为什么】 |
|  数据清理  | 缩写展开 特殊符号删掉 错别字更正 | 首先这个对结果有影响吗？第二这个似乎不能覆盖到百分之百把这样的话，少数没转换的岂不是成了离群点？ |
|  删除内存中的embedding  | 例如词嵌入 | py垃圾回收机制导致 |
|  重复值  | 重复值竟然有误标注的 | 这个似乎可以理解 |
|  无效特征和离群点  | ---- | ---- |
|  groupby  | 获取群的 max min agg | ---- |


