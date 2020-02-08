#Part Of Speech Tagging Task，词性标注实验

##1.任务简介
词性标注问题（Part Of Speech Tagging，POS Tagging）是自然语言处理的基本任务之一，也即确定每
个词是名词、动词、形容词或其他词性的过程。本次实验利用隐藏马尔可夫模型，来进行词性标注问题。

##2.HMM模型介绍
隐马尔可夫模型是可用于标注问题的统计学习模型，描述由隐藏的马尔可夫随机生成观测序列的过程，
属于生成模型。（参考李航《统计学习方法》）。在实际中，HMM一般面临三个基本问题：

a.概率计算问题，给定模型参数和观测序列O，计算在该模型下，观测序列O出现的概率，一般使用前向-后向算法

b.学习问题，给定观测序列O，用极大似然估计方法学习参数，通过EM算法可解，即使用Baum-Welch算法。

c.预测问题，也称为解码问题，给定模型参数和观测序列O，求解该观测序列O最有可能的隐藏状态。通过动态规划
算法维特比算法可解。

词性标注问题一般为第三个问题，即预测中间隐藏状态（即词性）的问题。用隐马尔可夫模型进行词性标注实验，
需要将标注出的词性作为隐藏状态，将单词作为观测结果。在实际学习中，HMM模型参数可以采用监督学习方法，
词性的预测也转变为给定模型参数，进行隐藏状态评估的问题，采用维特比算法可解。

##3.数据集处理
数据集的预处理，将文件的单词和词性转为数字index，并且扫描记录任意两个状态之间的转移次数和任意状态
与单词之间的转移次数，便于后续参数监督学习和平滑处理。参数平滑处理采用了拉普拉斯平滑，考虑到实际数
据均较小，拉普拉斯平滑操作加上0.01。

各个中间变量含义如下：
self.wordIndex = {}：具体单词和映射后的数字对应关系

self.tagIndex = {}：具体词性和映射后的数字对应关系
        
self.indexTag = {}：映射后的数字与具体词性对应关系，便于后续解码具体序列使用。

self.states = []：数字化的词性序列，一行为一个列表
        
self.observations = []：数字化后的单词序列，一行为一个列表


##4.模型代码简析

HMM.py主要为隐马科夫模型的文件，包括模型参数监督学习和预测隐藏状态的维特比算法。

变量：

self.A: 任意两个状态转移概率矩阵，N x N维，状态Tj到Ti的转移概率P（Ti|Tj) = C(Tj,Ti)/C(Tj)，
即统计从Tj转移到Ti的概率次数，分母为状态Tj出现的总次数。

self.B: 任意状态和单词之间的转移概率矩阵，N X M维，类似，状态Tj输出单词Wm的概率P（Wm|Tj) = C(Wm,Tj)/C(Tj)

self.Pi：初始状态矩阵，N X 1维，可统计每句话初始状态的出现次数，计算得到概率。

函数：

__supervisedCalcParams(self)：根据统计共同出现的状态

testAccuracy(self):计算测试集的准确率

__viterbi(self, observation)：给定输入，利用维特比算法输出预测结果

tagSequence(self,observations):给定输入，计算预测结果并输出

本次实验共有61种词性，即N=61，单词个数看采样输入文件不同而细微变化，大体为四万八千多。

##5.实验结果
本实验利用train_utf16.tag数据集文件中百分之七十为训练集，百分之三十为测试集（训练测试随机划分），利用
准确率（Accuracy）作为评估标准，可以达到百分之91.18%的准确率。

##6.Thinking
1.在实际使用马尔可夫模型处理过程中，因为将已经标注出的词性作为隐藏状态，参数学习较为容易，
但是状态到单词的转移矩阵十分巨大，存在数据高纬稀疏的际难题，数据的平滑处理比较敏感。

2.HMM在这个模型中参数学习利用的信息较为少，仅为基本统计，是否可以进一步利用潜在信息学习？

3.HMM模型计算需要维持两个高纬度的矩阵表，计算空间复杂度较高，是否可以用别的方法来模拟NXM的状态
到单词转移矩阵，利用时间换空间，并且进一步提高拟合的效果？
##7.参考资料

1.https://github.com/IdearHui/posTag

2.李航《统计学习方法》第10章，隐马尔可夫模型