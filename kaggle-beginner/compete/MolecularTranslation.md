# 图像转文字

## 概述

1. 可以研究dacon的方案, 似乎都是 cnn + attension + lstm 的方案 跟这个实现差不多 https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
2. 数据生成 https://www.kaggle.com/stainsby/improved-synthetic-data-for-bms-competition-v3?scriptVersionId=58294825
3. 这个问题能不能不当作图像转文字，当作图网络解析问题是否可以？ 我觉得估计头几名估计都是做图解析而非 image-caption
4. 这个 attention + lstm 能不能换成 transformer 的多头注意力  很可惜 没成功

## 需要调研的内容

1. keras concept 
2. 常见模型
3. 常见调试技巧
4. 常见sota模型和代码 https://github.com/google-research/vision_transformer 把 vit 嫁接到现在的模型上来

## 实验记录

- 试了三条路径吧

1.  蛤蟆哥的 tnt + tansformer 这个太耗时了 估计是因为数据集没有正确load到gpu导致的 https://www.kaggle.com/drzhuzhe/training-on-gpu-bms#Net-Modeules
2.  cnn + lstm 这个速度快是快 
3.  cnn + transformer 很遗憾 由于对 transformer 知之甚少 没法子用起来


## keras concept

1. conv2d
2. lstm


## 调试transformer 中遇到的问题

1. 最开始的文件上传io问题
2. cnn 到底应该如何 embedding 
3. postion embedding 是啥