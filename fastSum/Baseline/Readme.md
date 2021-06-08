# Baseline

In this folder, we provide the code (implemented by FastNLP) that uses transformer or LSTM as encoder and sequence labeling as decoder for extractive summarization, which can be used as the baseline of extractive summarization research.



## Dataset and Vob

1. You can refer to the data samples in testdata folder, which are stored in jsonl format.
2. [Glove.42b.300d](https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.42B.300d.zip) dictionary is recommended.



## Modify the ROUGE path

**Write the absolute path of the ROUGE installation in the tools\untils.py ! **

```python
############## Modify to your own Rouge path ##############
ROUGE_PATH = "~/ROUGE/"
```



## Run Cmdline

LSTM + Sequence Labeling:

```shell
python train.py --cuda --gpu <gpuid> --sentence_encoder deeplstm --sentence_decoder SeqLab --save_root <savedir> --log_root <logdir> --lr_descent --grad_clip --max_grad_norm 10 --use_pyrouge
```

Transformer + Sequence Labeling:

```shell
python train.py --cuda --gpu <gpuid> --sentence_encoder transformer --sentence_decoder SeqLab --save_root <savedir> --log_root <logdir> --lr_descent --grad_clip --max_grad_norm 10 --use_pyrouge
```

