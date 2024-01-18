## GNN from Scratch

### 1.Intro

传统机器学习的数据是独立同分布的，图机器学习处理的是关联数据。图数据具有任意尺寸输入、无固定结点顺序和参考锚点、动态变化、多模态特征等特点。最初图机器学习要进行特征工程，分为节点特征、边特征和全图特征。后来人们利用神经网络端到端地学习这些特征。因此，图机器学习第一步是图嵌入：端到端地将图数据映射为向量表示。之后利用得到的向量表示（结点、边），应用到各种下游任务上。具体而言就是输入数据是图，经过三个MLP分别学习节点、边和全图特征，之后将三者拼接(Pooling层)起来，再经过一个MLP进行分类或预测等任务。但是这样无法将图的结构信息考虑进来。因此出现了GCN，它考虑了图的结构信息，利用邻居节点更新中心节点的表示。

参考[CS224W](https://github.com/TommyZihao/zihao_course/tree/main/CS224W).

### 2.入门资料

+ b站up主：[背包2004的GNN视频及代码资料](https://www.bilibili.com/video/BV1v64y1j7nJ/?spm_id_from=333.788&vd_source=d8d2c837a6fad4a93cc95f349e30e2fe)
  + 覆盖内容为GCN、Cluster-GCN、GraphSAGE、GAT及对应的代码讲解。讲解适合二倍速观看。
  + 看完GCN基本原理，看它的应用时思考，节点分类任务和图分类任务的区别，以及Link prediction任务。
  + 对应的[github库](https://github.com/wisherg/python_data_course)，包括作者搜集的很多博客，质量很好，选看

+ [图表示学习极简教程 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/112295277)
  + 看简单视频入门后，看这篇综述，补充基础知识
  + 覆盖早期的图节点嵌入工作，另一个角度解释GCN、GraphSAGE、GAT
+ CS224W
  + [中文讲解](https://github.com/TommyZihao/zihao_course/tree/main/CS224W)
    + 代码收费，可二倍速听听讲解，有几篇论文精讲
    + 前部分内容主要讲传统图机器学习，特征工程，可不看
  + [cs224w（图机器学习）2021冬季课程学习笔记集合](https://blog.csdn.net/PolarisRisingWar/article/details/117287320)
    + 全开源，可学
+ [零基础多图详解图神经网络（GNN/GCN）](https://www.bilibili.com/video/BV1iT4y1d7zP/?spm_id_from=333.788&vd_source=d8d2c837a6fad4a93cc95f349e30e2fe)
  + 二倍速，半个小时可学完。比较醍醐灌顶，比别的视频废话少
+ GNN最佳应用
  + 2023Nature的Scaling deep learning for materials discovery

### 3.其它资料

+ [3. 2.1_DeepWalk](https://www.bilibili.com/video/BV1Jz4y1z7Ws/?p=3&spm_id_from=pageDriver&vd_source=d8d2c837a6fad4a93cc95f349e30e2fe)
  + 中文讲解较多模型，可当成进阶资料，学技术
  + [【图神经网络】GNN从入门到精通](https://www.bilibili.com/video/BV1K5411H7EQ?p=12&vd_source=d8d2c837a6fad4a93cc95f349e30e2fe) 4K资源
+ [GraphSAGE & PinSAGE详解 - mathor (wmathor.com)](https://wmathor.com/index.php/archives/1533/)
+ [PGL全球冠军团队带你攻破图神经网络_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1rf4y1v7cU/?spm_id_from=333.337.search-card.all.click&vd_source=d8d2c837a6fad4a93cc95f349e30e2fe)
  + 飞桨nb
+ [LOGS第2023/12/02期||密歇根州立大学 毛海涛：图神经网络什么时候失效？_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1jj411s7h5/?spm_id_from=333.1007.tianma.1-2-2.click&vd_source=d8d2c837a6fad4a93cc95f349e30e2fe)
+ [图神经网络(4)_图游走算法deepwalk与node2vec_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV15o4y1R7nC/?spm_id_from=333.880.my_history.page.click&vd_source=d8d2c837a6fad4a93cc95f349e30e2fe)
  + 代码简单易懂，可学
+ [图神经网络系列讲解及代码实现- Node2Vec 2_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1834y1f7P8/?p=8&spm_id_from=pageDriver)
  + 代码实现不错，值得学
+ [图机器学习](https://www.bilibili.com/video/BV1TP411Y7c3/?spm_id_from=333.999.0.0&vd_source=d8d2c837a6fad4a93cc95f349e30e2fe)
  + 比较系统



### 4.GNN-for-PDE

+ ICLR 2020 PHYSICS-AWARE DIFFERENCE GRAPH NETWORKS FOR SPARSELY-OBSERVED DYNAMICS

+ ICLR 2021 LEARNING CONTINUOUS-TIME PDES FROM SPARSE DATA WITH GRAPH NEURAL NETWORKS
+ ICLR 2022 MESSAGE PASSING NEURAL PDE SOLVERS
+ Computer Methods in Applied Mechanics and Engineering 2022 Physics-informed graph neural Galerkin networks: A unified framework for solving PDE-governed forward and inverse problems