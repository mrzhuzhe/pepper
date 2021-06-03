# 图像转文字

## 概述 2021/5/25

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


# 2021/06/03 更新

## 本次notebook

- 蛤蟆哥的notebook 
    
    (1) 文件预处理的 notebook     
    > https://www.kaggle.com/drzhuzhe/bms-preprocess-data-parallel
    1. 首先处理好的文件非常多，要用multiproccess来处理会快很多，第二kaggle会自动解压后缀名为tar和zip的文件
    2. 青蛙哥做的好的地方：文件预处理后生成了csv将不同长度文件分别处理
    3. 青蛙哥做的不好的地方：文件同时占用cpu和gpu似乎cuda操作不对
    
    (2) 训练的notebook 
    > https://www.kaggle.com/drzhuzhe/training-on-gpu-bms 
    1. 尝试 del 和 gc.collect() 来减少内存占用，并没有成功
    2. valid 很占用内存， 内存随着读取数据而线性增加 batch32 时只能维持 20k个iterate

- cnn + trasnformer
    > https://www.kaggle.com/drzhuzhe/transformer
    (1) 当时原本考虑的方向：
      找到青蛙哥和其他人的差别：TNT嘛
      分模块自测，自己搭建baseline
      迁移青蛙的版本上tpu
    (2) 当时想到的问题
    
    【tnt】理解tnt的流程，画一下流程图  cnn -> token embeddeing -> text embedding -> text_pos -> text_embed -> text_decoder
    
    【tnt】 block 的详细结构

    【vit原理】patch 的作用

    【工程】backbone是什么，keras入门，创建 decoder mask 梯度积累

    （3）valid 速度特别慢， 似乎没有拉起tpu

    （4）*当时有个致命的问题* spike learning rate 
    当时我尝试了好几种方案
    由于是第一次搭建模型，一直以为是模型结构问题
    例如：类型不一致造成的浮点精度转换问题，模型容量不足，维度没有对齐，需要更多epoch
    https://www.kaggle.com/c/bms-molecular-translation/discussion/241716#1326759
    这个问题的解决方案出处还是在 attetion is all you need 论文中

- vision trasnformer
    > https://www.kaggle.com/drzhuzhe/bms-vision-transformer
    1. valid 特别慢，跟模型深度成正比
    2. 训练中很不稳定
    最可能的解决办法是用预训练权重来精调
    目前采用的方式是用梯度裁剪的方式解决的
    3. 目前只用了半长度，可以用全长度来精调 https://www.kaggle.com/c/bms-molecular-translation/discussion/239595#1310687