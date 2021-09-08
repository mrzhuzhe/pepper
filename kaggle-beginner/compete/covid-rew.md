# 方案rew

## 第二十七名

1. 预训练 chexpert
2. 自蒸馏和标签平滑
3. adamW 模拟退火



## 第三名

https://github.com/yujiariyasu/siim_covid19_detection
参考历史方案
https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver


## 第六名

Unet++？？？
https://github.com/b02202050/2021-SIIM-COVID19-Detection


## 第五名

writeup
https://www.kaggle.com/c/siim-covid19-detection/discussion/266571

noise studnet
https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/226616

pcam 池化
https://github.com/jfhealthcare/Chexpert

## 第四名

加aux loss https://www.kaggle.com/c/siim-covid19-detection/discussion/263676

## 第十名

https://github.com/GuanshuoXu/SIIM-FISABIO-RSNA-COVID-19-Detection-10th-place-solution

## 第一名

https://www.kaggle.com/c/siim-covid19-detection/discussion/263658

chexpert https://stanfordmlgroup.github.io/competitions/chexpert/

1. 偏斜loss
2. 滑动平均
3. central crop augment

经过仔细翻代码

1. 下游任务通过decoder接到encoder上，并没有用加到某一层
2. 用了 clahe embross 和 sharpen 的数据自增
3. 训练过程中更新权重用了 指数加权滑动平均 见 timm moving averge
4. cosineAnnealingLR 只有一个周期
5. tta 中单独取出中间百分之八十（单独祛除肺部部分）
6. wbf iou 0.6 
7. focal loss 给不平衡的更多权重，但是这个没看到参数
8. 根据分数定融合权重
9. 标签平滑
10. pcam池化
11. noise student 和 自蒸馏

# 杂谈

1. 这个比赛中很多人都提到之前的比赛 ranzer  https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/discussion/207577 chexper vin之类的

2. ranzer 中用额外数据用来训练分割，再把分割当作额外channel加入分类中 unet cnn架构

3. 再谈一下 aux ， aux 和 multi stage

4. 标签平滑 psedulabel 和 自蒸馏

5. 模型融合 和 vit

6. 查一下adam和adamw的区别