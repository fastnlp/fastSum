# MatchSum

FastNLP实现的ACL2020论文: *[Extractive Summarization as Text Matching](https://arxiv.org/abs/2004.08795)*

代码和README文件来自*[maszhongming/MatchSum](https://github.com/maszhongming/MatchSum)*，感谢代码提供者[Ming Zhong](https://github.com/maszhongming)



## 数据集

可以通过[此链接](https://drive.google.com/open?id=1FG4oiQ6rknIeL2WLtXD0GWyh6pBH9-hX)下载处理好的CNN/DailyMail数据集，将其解压并移动到“./data”文件夹下。  其中包含BERT、RoBERTa两个版本的数据集，共六个文件。

此外，你可以通过这个[此链接](https://drive.google.com/file/d/1PnFCwqSzAUr78uEcA_Q15yupZ5bTAQIb/view?usp=sharing)下载其他五个处理过的数据集（WikiHow、PubMed、XSum、MultiNews、Reddit）。



## 训练

模型训练过程中使用的是8个Tesla-V100-16G GPU，训练时间约为30小时。如果没有足够的显存，可以在`train_matching.py`文件中调整 *batch_size*和*candidate_num* ，也可以在`dataloader.py`文件中调整*max_len*。

可以选择BERT或RoBERTa作为**MatchSum**的编码器，例如，要训练RoBERTa模型，可以运行以下命令：

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_matching.py --mode=train --encoder=roberta --save_path=./roberta --gpus=0,1,2,3,4,5,6,7
```



## 测试

训练完成后，你可以在训练时指定的文件夹中寻找最优检查点，例如“./roberta/2020-04-12-09-24-51”。可以运行以下命令来获取测试集的结果（测试时只需要一个GPU）：

```shell
CUDA_VISIBLE_DEVICES=0 python train_matching.py --mode=test --encoder=roberta --save_path=./roberta/2020-04-12-09-24-51/ --gpus=0
```
ROUGE分数将显示在屏幕上，模型的输出存储在`./roberta/result`文件夹中。



## CNN/DailyMail上的表现
在测试集上的表现（取三次运行的平均值）

| Model | R-1 | R-2 | R-L |
| :------ | :------: | :------: | :------: |
| MatchSum (BERT-base) | 44.22 | 20.62 | 40.38 |
| MatchSum (RoBERTa-base) | 44.41 | 20.86 | 40.55 |



## 生成的摘要

模型在CNN/DM数据集上生成的摘要可以从[此链接](https://drive.google.com/open?id=11_eSZkuwtK4bJa_L3z2eblz4iwRXOLzU)中获取。其中，**MatchSum（BERT）**的结果是44.26/20.58/40.40（R-1/R-2/R-L），**MatchSum（RoBERTa）**的结果是44.45/20.88/40.60。

在其他数据集上生成的摘要可以从[此链接](https://drive.google.com/open?id=1iNY1hT_4ZFJZVeyyP1eeoVY14Ej7l9im)中获取。



## 预训练模型

可以从[此链接](https://drive.google.com/file/d/1PxMHpDSvP1OJfj1et4ToklevQzcPr-HQ/view?usp=sharing)中获取在CNN/DM数据集上训练的两个预训练的模型，加载方式如下：

```python
model = torch.load('MatchSum_cnndm_bert.ckpt')
```

其他数据集上的预训练模型可以从[此链接](https://drive.google.com/open?id=1EzRE7aEsyBKCeXJHKSunaR89QoPhdij5)中获取。



## 处理自己的数据

如果要处理自己的数据并获取每个文档的候选摘要，首先需要将数据集转换为与我们相同的*jsonl*格式，并确保其中包含*text*和*summary*字段。其次，应该使用BertExt或其他方法从每个文档中选择一些重要的句子，并获得一个*index.jsonl*文件（我们在`./preprocess/test_cnndm.jsonl`中提供了一个示例）。

之后可以运行以下命令：

```shell
python get_candidate.py --tokenizer=bert --data_path=/path/to/your_original_data.jsonl --index_path=/path/to/your_index.jsonl --write_path=/path/to/store/your_processed_data.jsonl
```

**在运行此命令之前，请在`preprocess/get_candidate.py`中的第22行填写你的ROUGE路径。**值得注意的是，需要根据你的数据集调整候选摘要的数量和候选摘要中的句子数量。如果你喜欢直接阅览代码，请查阅`preprocess/get_candidate.py`中的第89-97行。

完成数据预处理后，在使用我们的代码训练您自己的模型之前，请根据数据集中候选摘要的数量和长度，调整`train_matching.py`中的*candidate_num*和`dataloader.py`中的*max_len*。



## 额外说明

这里发布的代码和数据用于匹配模型。在匹配阶段之前，我们使用BertExt来修剪无意义的候选摘要，BertExt的实现可以参考[PreSumm](https://github.com/nlpyang/PreSumm)。
