# tf_classification 基于字符级中文文本常见BaseLine汇集

针对相同[数据集](https://pan.baidu.com/share/init?surl=hugrfRu)分别采用**CNN**, **RNN**, **RNN Attention机制**进行分类

### 环境　推荐[Anaconda mini管理](https://www.jianshu.com/p/169403f7e40c)

python3.6.5 

tensorflow==1.8.0


### 项目文件.
1. data_utils.py 为数据预处理文件. 可以有`Document*Sentence*Word` 模式 和`Document*Word`模式. 预处理时采用收尾截取一定数量文本作为训练样本(总分总).
2. cnn_model.py 为cnn模型文件, 取固定句子长度. 采用`Document*Word`模式
3. rnn_model.py 为rnn`bilstm`模型文件, 每批次句子长度不固定. 采用`Document*Word`模式
4. rnn_attention.py 为rnn`gru` attention模型文件, 句子长度不固定. 采用`Document*Sentence*Word`模式
5. train_model.py 为模型加载文件, 分为公共超参数和自定义参数.
6. predict.py 为预测文件. 

### 效果比对(分为训练集和测试集)
#### cnn 

```angular2html
2338 <Train>  Epoch: [3] Iter: [2334] Step: [149279] Loss: [0.044]    Acc: [1.000]
2339 <Train>  Epoch: [3] Iter: [2335] Step: [149343] Loss: [0.163]    Acc: [0.953]
2340 <Train>  Epoch: [3] Iter: [2336] Step: [149407] Loss: [0.029]    Acc: [1.000]
2341 <Train>  Epoch: [3] Iter: [2337] Step: [149471] Loss: [0.032]    Acc: [1.000]
2342 <Train>  Epoch: [3] Iter: [2338] Step: [149535] Loss: [0.140]    Acc: [0.938]
2343 <Train>  Epoch: [3] Iter: [2339] Step: [149599] Loss: [0.134]    Acc: [0.953]
2344 <Train>  Epoch: [3] Iter: [2340] Step: [149663] Loss: [0.120]    Acc: [0.953]
2345 <Train>  Epoch: [3] Iter: [2341] Step: [149727] Loss: [0.112]    Acc: [0.953]
2346 <Train>  Epoch: [3] Iter: [2342] Step: [149791] Loss: [0.048]    Acc: [0.969]
2347 <Train>  Epoch: [3] Iter: [2343] Step: [149855] Loss: [0.045]    Acc: [0.984]
2348 <Train>  Epoch: [3] Iter: [2344] Step: [149919] Loss: [0.350]    Acc: [0.969]
2349 <Train>  Epoch: [3] Iter: [2345] Step: [149983] Loss: [0.103]    Acc: [0.984]
2350 <Train>  Epoch: [3] Iter: [2346] Step: [150000] Loss: [0.004]    Acc: [1.000]
```

```angular2html
2354 <Test>   Iter: [1] Loss: [0.030]     Acc: [1.000]
2355 <Test>   Iter: [2] Loss: [0.152]     Acc: [0.938]
2356 <Test>   Iter: [3] Loss: [0.060]     Acc: [0.984]
2357 <Test>   Iter: [4] Loss: [0.101]     Acc: [0.953]
2358 <Test>   Iter: [5] Loss: [0.017]     Acc: [1.000]
2359 <Test>   Iter: [6] Loss: [0.088]     Acc: [0.969]
2360 <Test>   Iter: [7] Loss: [0.018]     Acc: [1.000]
2361 <Test>   Iter: [8] Loss: [0.071]     Acc: [0.969]
2362 <Test>   Iter: [9] Loss: [0.190]     Acc: [0.953]
2363 <Test>   Iter: [10] Loss: [0.062]    Acc: [0.969]
```

### 参考文献及Demo

主要采用了数据集和CNN相关文档: [CNN中文文本分类Demo](https://github.com/gaussic/text-classification-cnn-rnn)

分别对字和句子进行Attention处理: [Attention机制论文](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

Seq2Seq图文: [Attention机制博客](https://theneuralperspective.com/2016/11/20/recurrent-neural-network-rnn-part-4-attentional-interfaces/)

通俗易懂Attention: [Attention机制CSDN](https://blog.csdn.net/BVL10101111/article/details/78470716)

大神博客(神经网络相关): [Colah博客](http://colah.github.io)
