# README

本仓库魔改自  https://github.com/verazuo/badnets-pytorch 和 https://github.com/tonggege001/MyNeuralCleanse

网络空间安全导论作业：第十四章 神经网络暗门攻击与防御

包含偏移标签的 BadNets 和 Neural Cleanse 防御方法。

作者：Cold_Chair

## 资料参考：

课程参考书：《网络空间安全原理与实践》

### BadNets:
https://zhuanlan.zhihu.com/p/626020461

### Neural Cleanse:

https://blog.csdn.net/qq_41581588/article/details/126299340

https://zhuanlan.zhihu.com/p/414418322

## 环境配置

创建虚拟环境，按照显卡配置安装好 pytorch, torchvision, scikit-learn

然后 `pip install -r requirements.txt`

## 数据集下载：

```
python data_downloader.py
```

运行该条指令自动下载 MNIST 数据集并解压到 ./dataset 目录下

## 训练暗门模型：
```
python main.py
```

默认参数：

训练 100 周期

SGD 优化器，学习率 = 0.01

投毒：从总数据中选 0.2 比例的数据，把 label 4 改成 label 7

## Neural Cleanse

```
python detect.py
```

会自动调用前面训练的暗门模型。

对每个类别，训练周期 $= 5$。

会输出各个类别的 $a-index$ 值，$>2$ 的即为很可能被注入攻击的类别。

同时输出每个类别的反向工程的 mask 和 trigger，在 mask 文件夹下。

