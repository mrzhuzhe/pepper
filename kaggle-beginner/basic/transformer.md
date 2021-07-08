# Transformer 

> hugface 文档的记录

## 整体大流程

1. tokenizer: byte-pair wordpieces sentence pieces  自己训练tokenizer？ 
2. 部署问题 tensorRT ONMX
3. transformer 的变种 （1）自监督任务不同 （2）模型参数数量不同
4. attention 机制的表示学习解释
5. 蒸馏模型：https://medium.com/huggingface/distilbert-8cf3380435b5
  通过概率接近最高的结果，找到特征之间的相似性
  损失函数是模型a 的结果 和模型b结果的温度交叉熵（这能train的动？这岂不是训练的超级慢？）
  难道任何模型都可以做模型蒸馏？
  蒸馏模型是无监督学习？
  你咋知道模型b保留哪几层呢？
6. 移除layer 还是减少 size
