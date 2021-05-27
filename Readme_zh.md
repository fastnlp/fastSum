# Summarization 



## Extractive Summarization


### Models

FastNLP中实现的模型包括：

1. Get To The Point: Summarization with Pointer-Generator Networks (See et al. 2017)
2. Searching for Effective Neural Extractive Summarization  What Works and What's Next (Zhong et al. 2019)
3. Fine-tune BERT for Extractive Summarization (Liu et al. 2019)


### Dataset

这里提供的摘要任务数据集包括：
|Name|Paper|Description|
|:---:|:---:|:---:|
|CNN/DailyMail|[Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](https://www.aclweb.org/anthology/K16-1028/)|**新闻**。修改了[原本用于 passage-based question answering 任务](https://arxiv.org/abs/1506.03340)的数据库 。 [CNN 和 DailyMail 的网站为每篇文章都**人工**提供了一些要点信息总结文章。而且这些要点是抽象的而非抽取式摘要形式。](https://arxiv.org/abs/1506.03340 "Both news providers supplement their articles with a number of bullet points, summarising aspects of the information contained in the article. Of key importance is that these summary points are abstractive and do not simply copy sentences from the documents.") [微调 Teaching Machines to Read and Comprehend 的脚本之后，作者生成了一个 **multi-sentence summary** 的数据集合。](https://www.aclweb.org/anthology/K16-1028/ "With a simple modification of the script, we restored all the summary bullets of each story in the original order to obtain a multi-sentence summary, where each bullet is treated as a sentence.")|
|Xsum|[Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://arxiv.org/abs/1808.08745)|**新闻**。[article-**single sentence summary** 的数据集。在 BBC 上，每篇文章开头都附有人工撰写的摘要，提取即可。](https://arxiv.org/abs/1808.08745 "Our extreme summarization dataset (which we call XSum) consists of BBC articles and accompanying single sentence summaries. Specifically, each article is prefaced with an introductory sentence (aka summary) which is professionally written, typically by the author of the article.")|
|The New York Times Annotated Corpus|[The New York Times Annotated Corpus](https://catalog.ldc.upenn.edu/LDC2008T19)|**新闻**。[人工撰写的摘要](https://catalog.ldc.upenn.edu/LDC2008T19 "As part of the New York Times' indexing procedures, most articles are manually summarized and tagged by a staff of library scientists. This collection contains over 650,000 article-summary pairs which may prove to be useful in the development and evaluation of algorithms for automated document summarization. ")|
|DUC|[The Effects of Human Variation in DUC Summarization Evaluation](https://www.aclweb.org/anthology/W04-1003/)| 2002 Task4 -; 2003/2004 Task1:新闻。[2003 和 2004 Task1 基本一致，都是对**每个 doc** 生成一段摘要](https://duc.nist.gov/duc2004/ "Tasks 1 and 2 were essentially the same as in DUC 2003; Use the 50 TDT English clusters. Given each document, create a very short summary (<= 75 bytes) of the document.")|
|arXiv PubMed|[A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents](https://arxiv.org/abs/1804.05685)| [scientific paper。从 arXiv 和 PubMed 获取的**长篇文档**的摘要。](https://arxiv.org/abs/1804.05685 "We also introduce two large-scale datasets of long and structured scientific papers obtained from arXiv and PubMed to support both training and evaluating models on the task of long document summarization.")[论文的 abstract 部分作为摘要的 ground-truth。](https://arxiv.org/abs/1804.05685 "Following these works, we take scientific papers as an example of long documents with discourse information, where their abstracts can be used as ground-truth summaries.")|
|WikiHow|[WikiHow: A Large Scale Text Summarization Dataset](https://arxiv.org/abs/1810.09305)|[WikiHow 有一个关于“怎么做”的数据库。](https://arxiv.org/pdf/1810.09305.pdf "The WikiHow knowledge base contains online articles describing a procedural task about various topics (from arts and entertainment to computers and electronics) with multiple methods or steps and new articles are added to it regularly.")[每个步骤描述是由一段加粗摘要以及详细步骤叙述组成。](https://arxiv.org/pdf/1810.09305.pdf "Each step description starts with a bold line summarizing that step and is followed by a more detailed explanation.")[作者把每个步骤的加粗摘要合并作为最终摘要，每步的剩余部分进行合并组成 article。](https://arxiv.org/pdf/1810.09305.pdf "To generate the reference summaries, bold lines representing the summary of the steps are extracted and concatenated. The remaining parts of the steps (the detailed descriptions) are also concatenated to form the source article.")|
|Multi News|[Multi-News: a Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model](https://arxiv.org/abs/1906.01749)|新闻、Multi-Document Summarization。[数据集由新闻文章和这些文章的**人工摘要**组成，这些文章来自 newser.com 网站。每一篇摘要都是由专业编辑撰写的，并包含到引用的原始文章的链接。](https://arxiv.org/abs/1906.01749 "Our dataset, which we call Multi-News, consists of news articles and human-written summaries of these articles from the site newser.com. Each summary is professionally written by editors and includes links to the original articles cited.")|
|BillSum|[BillSum: A Corpus for Automatic Summarization of US Legislation](https://arxiv.org/abs/1910.00523)|Legislation。[数据是选自美国国会和加利福尼亚州立法机构的**法案文本**，**人为编写**的摘要。](https://www.kaggle.com/akornilo/billsum "The BillSum dataset is the first corpus for automatic summarization of US Legislation. The corpus contains the text of bills and human-written summaries from the US Congress and California Legislature. It was published as a paper at the EMNLP 2019 New Frontiers in Summarization workshop.")|
|AMI|[The AMI meeting corpus: a pre-announcement](http://groups.inf.ed.ac.uk/ami/download/)|Meeting。[AMI会议语料库是一种**多模式**数据集，包含100小时的会议多模式记录。](http://homepages.inf.ed.ac.uk/jeanc/amidata-MLMI-bookversion.final.pdf "As part of the development process, the project is collecting a corpus of 100 hours of meetings using instrumentation that yields high quality, synchronized multimodal recording, with, for technical reasons, a focus on groups of four people.")[本语料库为每个单独的讲话者提供了高质量的人工记录，还包含了**抽取式摘要**、**生成式摘要**、头部动作、手势、情绪状态等。](http://groups.inf.ed.ac.uk/ami/corpus/overview.shtmlhttp://groups.inf.ed.ac.uk/ami/corpus/overview.shtml "The AMI Meeting Corpus includes high quality, manually produced orthographic transcription for each individual speaker, including word-level timings that have derived by using a speech recognizer in forced alignment mode. It also contains a wide range of other annotations, not just for linguistic phenomena but also detailing behaviours in other modalities. These include dialogue acts; topic segmentation; extractive and abstractive summaries; named entities; the types of head gesture, hand gesture, and gaze direction that are most related to communicative intention; movement around the room; emotional state; and where heads are located on the video frames.")|
|ICSI|[ICSI Corpus](http://groups.inf.ed.ac.uk/ami/icsi/)|Meeting。[ICSI会议语料库是一个音频数据集，包含大约70个小时的会议记录。](http://groups.inf.ed.ac.uk/ami/icsi/ "The ICSI Meeting Corpus is an audio data set consisting of about 70 hours of meeting recordings.")[包含了**抽取式摘要**和**生成式摘要**。](http://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_plus_NXT.zip "We will be annotating two different kinds of summaries for this data, both aimed at the external researcher. One is abstractive. The other is extractive.")|
|Reddit TIFU|[Abstractive Summarization of Reddit Posts with Multi-level Memory Networks](https://arxiv.org/abs/1811.00783)|Online discussion。[从 Reddit 爬取数据。](https://arxiv.org/abs/1811.00783 "We collect data from Reddit, which is a discussion forum platform with a large number of subreddits on diverse topics and interests.")[作者生成了两套摘要：用原帖的 title 作为 **short summary**，TL;DR summary 作为 long summary。](https://arxiv.org/abs/1811.00783 "Thus, we regard the body text as source, the title as short summary, and the TL;DR summary as long summary.")|
|SAMSum|[SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization](https://arxiv.org/abs/1911.12237)|Conversation。 [对话由语言学家根据日常对话写成。](https://arxiv.org/abs/1911.12237 "We asked linguists to create conversations similar to those they write on a daily basis, reflecting the proportion of topics of their real-life messenger conversations.")[之后由语言学家标注摘要。](https://arxiv.org/abs/1911.12237 "After collecting all of the conversations, we asked language experts to annotate them with summaries")|



<!-- 其中公开数据集(CNN/DailyMail, Newsroom, arXiv, PubMed)预处理之后的下载地址：

- [百度云盘](https://pan.baidu.com/s/11qWnDjK9lb33mFZ9vuYlzA) (提取码：h1px)
- [Google Drive](https://drive.google.com/file/d/1uzeSdcLk5ilHaUTeJRNrf-_j59CQGe6r/view?usp=drivesdk)

未公开数据集(NYT, NYT50, DUC)数据处理部分脚本放置于data文件夹 -->

你可以运行summarizationLoader.py下载并使用它们。



### Evaluation

#### FastRougeMetric

FastRougeMetric使用python实现的ROUGE非官方库来实现在训练过程中快速计算rouge近似值。
 源代码可见 [https://github.com/pltrdy/rouge](https://github.com/pltrdy/rouge)

在fastNLP中，该方法已经被包装成Metric.py中的FastRougeMetric类以供trainer直接使用。
需要事先使用pip安装该rouge库。

    pip install rouge


**注意：由于实现细节的差异，该结果和官方ROUGE结果存在1-2个点的差异，仅可作为训练过程优化趋势的粗略估计。**

​    

#### PyRougeMetric

PyRougeMetric 使用论文 [*ROUGE: A Package for Automatic Evaluation of Summaries*](https://www.aclweb.org/anthology/W04-1013) 提供的官方ROUGE 1.5.5评测库。

由于原本的ROUGE使用perl解释器，[pyrouge](https://github.com/bheinzerling/pyrouge)对其进行了python包装，而PyRougeMetric将其进一步包装为trainer可以直接使用的Metric类。

为了使用ROUGE 1.5.5，需要使用sudo权限安装一系列依赖库。

1. ROUGE 本身在Ubuntu下的安装可以参考[博客](https://blog.csdn.net/Hay54/article/details/78744912)
2. 配置wordnet可参考：
```shell
$ cd ~/rouge/RELEASE-1.5.5/data/WordNet-2.0-Exceptions/
$ ./buildExeptionDB.pl . exc WordNet-2.0.exc.db
$ cd ../
$ ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```
3. 安装pyrouge
```shell
$ git clone https://github.com/bheinzerling/pyrouge
$ cd pyrouge
$ python setup.py install
```
4. 测试ROUGE安装是否正确
```shell
$ pyrouge_set_rouge_path /absolute/path/to/ROUGE-1.5.5/directory
$ python -m pyrouge.test
```




### Dataset_loader

- SummarizationLoader: 用于读取处理好的jsonl格式数据集，返回以下field
    - text: 文章正文
    - summary: 摘要
    - domain: 可选，文章发布网站
    - tag: 可选，文章内容标签
    - labels: 抽取式句子标签




### Performance and Hyperparameters

Dataset: CNN/DailyMail

|              Model              | ROUGE-1 | ROUGE-2 | ROUGE-L |                    Paper                    |
| :-----------------------------: | :-----: | :-----: | :-----: | :-----------------------------------------: |
|             LEAD 3              |  40.11  |  17.64  |  36.32  |            our data pre-process             |
|             ORACLE              |  55.24  |  31.14  |  50.96  |            our data pre-process             |
|    LSTM + Sequence Labeling     |  40.72  |  18.27  |  36.98  |                                             |
| Transformer + Sequence Labeling |  40.86  |  18.38  |  37.18  |                                             |
|     LSTM + Pointer Network      |    -    |    -    |    -    |                                             |
|           BERTSUMEXT            |  42.71  |  19.76  |  39.03  | [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) |
|          TransSUMEXT            |  42.71  |  19.76  |  39.03  | [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) |
|           BERTSUMABS            |  42.71  |  19.76  |  39.03  | [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) |
|          TransSUMABS            |  42.71  |  19.76  |  39.03  | [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) |



