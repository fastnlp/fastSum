# Baseline

使用FastNLP实现的以Transformer、LSTM（二者可选其一）作为编码器，序列标注作为解码器进行抽取式摘要的代码，可作为抽取式摘要研究的基线。



## 数据集和词典

1. 可以参考testdata文件夹中的数据样例，数据集均以jsonl格式存储；
2. 词典推荐使用[glove.42B.300d](https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.42B.300d.zip)



## 修改ROUGE路径

**在tools中的untils.py文件中写入ROUGE安装的绝对路径**

```python
############## 修改成自己的ROUGE路径 ##############
ROUGE_PATH = "~/ROUGE/"
```



## 运行和使用

LSTM + Sequence Labeling:

```shell
python train.py --cuda --gpu <gpuid> --sentence_encoder deeplstm --sentence_decoder SeqLab --save_root <savedir> --log_root <logdir> --lr_descent --grad_clip --max_grad_norm 10 --use_pyrouge
```

Transformer + Sequence Labeling:

```shell
python train.py --cuda --gpu <gpuid> --sentence_encoder transformer --sentence_decoder SeqLab --save_root <savedir> --log_root <logdir> --lr_descent --grad_clip --max_grad_norm 10 --use_pyrouge
```

