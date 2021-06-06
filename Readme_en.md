# FastSum

FastSum is a complete solution of text summarization based on fastNLP, including dataset, model and evaluation.



## Models

The models implemented in fastSum:

1. [Baseline (LSTM/Transformer + SeqLab)](./fastSum/Baseline)
2. [Get To The Point: Summarization with Pointer-Generator Networks](./fastSum/PointerGen)
3. [Extractive Summarization as Text Matching](./fastSum/MatchSum)
4. [Text Summarization with Pretrained Encoders](./fastSum/PreSum)



## Dataset

We provide 12 datasets of text summarization tasks:

|                Name                 |                            Paper                             |                 Type                  |                         Description                          |
| :---------------------------------: | :----------------------------------------------------------: | :-----------------------------------: | :----------------------------------------------------------: |
|            CNN/DailyMail            | [Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](https://www.aclweb.org/anthology/K16-1028/) |                 News                  | Modified the database originally used for [passage-based question answering](https://arxiv.org/abs/1506.03340) tasks. "Both news providers supplement their articles with a number of bullet points, summarizing aspects of the information contained in the article. Of key importance is that these summary points are **abstractive** and do not simply copy sentences from the documents.""With a simple modification of the script, we restored all the summary bullets of each story in the original order to obtain **a multi-sentence summary**, where each bullet is treated as a sentence." |
|                Xsum                 | [Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://arxiv.org/abs/1808.08745) |                 News                  | "Our extreme summarization dataset (which we call XSum) consists of BBC articles and accompanying **single sentence summaries**. Specifically, each article is prefaced with an introductory sentence (aka summary) which is professionally written, typically by the author of the article." |
| The New York Times Annotated Corpus | [The New York Times Annotated Corpus](https://catalog.ldc.upenn.edu/LDC2008T19) |                 News                  | "As part of the New York Times' indexing procedures, most articles are manually summarized and tagged by **a staff of library scientists**. This collection contains over **650,000 article-summary pairs** which may prove to be useful in the development and evaluation of algorithms for automated document summarization. " |
|                 DUC                 | [The Effects of Human Variation in DUC Summarization Evaluation](https://www.aclweb.org/anthology/W04-1003/) |                 News                  | 2002 Task4 -; 2003/2004 Task1&2: "Tasks 1 and 2 were essentially the same as in DUC 2003; Use the 50 TDT English clusters. **Given each document**, create a very short summary (<= 75 bytes) of the document." |
|            arXiv PubMed             | [A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents](https://arxiv.org/abs/1804.05685) |           Scientific Paper            | "We also introduce two large-scale datasets of long and structured scientific papers obtained from arXiv and PubMed to support both training and evaluating models on the task of **long document summarization**." "Following these works, we take scientific papers as an example of long documents with discourse information, where their abstracts can be used as **ground-truth summaries**." |
|               WikiHow               | [WikiHow: A Large Scale Text Summarization Dataset](https://arxiv.org/abs/1810.09305) |            Knowledge Base             | "The WikiHow knowledge base contains online articles describing a procedural task about various topics (from arts and entertainment to computers and electronics) with multiple methods or steps and new articles are added to it regularly." "Each step description starts with a bold line summarizing that step and is followed by a more detailed explanation." "The concatenation of all the bold lines (the summary sentences) of all the paragraphs to serve as the reference summary. The concatenation of all paragraphs (except the bold lines) to generate the article to be summarized." |
|             Multi News              | [Multi-News: a Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model](https://arxiv.org/abs/1906.01749) | News&**Multi-Document Summarization** | "Our dataset, which we call Multi-News, consists of news articles and **human-written summaries** of these articles from the site newser.com. Each summary is professionally written by editors and includes links to the original articles cited." |
|               BillSum               | [BillSum: A Corpus for Automatic Summarization of US Legislation](https://arxiv.org/abs/1910.00523) |              Legislation              | "The BillSum dataset is the first corpus for automatic summarization of US Legislation. The corpus contains **the text of bills** and **human-written summaries** from the US Congress and California Legislature. It was published as a paper at the EMNLP 2019 New Frontiers in Summarization workshop." |
|                 AMI                 | [The AMI meeting corpus: a pre-announcement](http://groups.inf.ed.ac.uk/ami/download/) |                Meeting                | ["As part of the development process, the project is collecting a corpus of **100 hours of meetings** using instrumentation that yields high quality, **synchronized multimodal recording**, with, for technical reasons, a focus on groups of four people." ](http://homepages.inf.ed.ac.uk/jeanc/amidata-MLMI-bookversion.final.pdf )"The AMI Meeting Corpus includes high quality, manually produced orthographic transcription for each individual speaker, including word-level timings that have derived by using a speech recognizer in forced alignment mode. It also contains a wide range of other annotations, not just for linguistic phenomena but also detailing behaviours in other modalities. These include dialogue acts; topic segmentation; **extractive and abstractive summaries**; named entities; the types of head gesture, hand gesture, and gaze direction that are most related to communicative intention; movement around the room; emotional state; and where heads are located on the video frames." |
|                ICSI                 |     [ICSI Corpus](http://groups.inf.ed.ac.uk/ami/icsi/)      |                Meeting                | "The ICSI Meeting Corpus is an audio data set consisting of about 70 hours of meeting recordings."["We will be annotating two different kinds of summaries for this data, both aimed at the external researcher. One is abstractive. The other is extractive."](http://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_plus_NXT.zip ) |
|             Reddit TIFU             | [Abstractive Summarization of Reddit Posts with Multi-level Memory Networks](https://arxiv.org/abs/1811.00783) |           Online discussion           | "We collect data from Reddit, which is a discussion forum platform with a large number of subreddits on diverse topics and interests." "Thus, we regard the body text as source, the title as short summary, and the TL;DR summary as long summary." |
|               SAMSum                | [SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization](https://arxiv.org/abs/1911.12237) |             Conversation              | "We asked linguists to create conversations similar to those they write on a daily basis, reflecting the proportion of topics of their real-life messenger conversations." "After collecting all of the conversations, we asked language experts to annotate them with summaries." |

You can run [summarizationLoader.py](./fastSum/Dataloader/summarizationLoader.py) to download and use them.



## Evaluation

### FastRougeMetric

FastRougeMetric uses the unofficial ROUGE library implemented in python to quickly calculate rouge approximations during training.

[Source code](https://github.com/pltrdy/rouge)

In fastNLP, this method has been packaged into the FastRougeMetric class in Metric.py. You can easily call this API. o(^▽^)o

Firstly, it's necessary for you to install the rouge library using pip.

    pip install rouge

**Note(￣^￣): Due to the difference in implementation details, there is a tiny difference between this result and the official ROUGE result, which means it can only be used as a rough estimation of the optimization trend of the training process.**



### PyRougeMetric

PyRougeMetric uses the official ROUGE 1.5.5 library provided by [ROUGE: A Package for Automatic Evaluation of Summaries](https://www.aclweb.org/anthology/W04-1013).

Since the original ROUGE uses the perl interpreter, [pyrouge](https://github.com/bheinzerling/pyrouge) has carried out python packaging for it. PyRougeMetric packages it into a Metric class. You can use it conveniently. 

In order to use ROUGE 1.5.5, a series of libraries need to be installed with **sudo** privileges.

```shell
sudo apt-get install libxml-perl libxml-dom-perl
pip install git+git://github.com/bheinzerling/pyrouge
export PYROUGE_HOME_DIR=the/path/to/RELEASE-1.5.5
pyrouge_set_rouge_path $PYROUGE_HOME_DIR
chmod +x $PYROUGE_HOME_DIR/ROUGE-1.5.5.pl
```

To avoid download failure, we put all the installation files you need in [resources](./fastSum/resources). You can refer to https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5 for RELEASE-1.5.5. Remember to build Wordnet 2.0 instead of 1.6 in RELEASE-1.5.5/data.

```shell
cd $PYROUGE_HOME_DIR/data/WordNet-2.0-Exceptions/
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ../
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```

**Test whether the installation of rouge is successful:**

```shell
pyrouge_set_rouge_path /absolute/path/to/ROUGE-1.5.5/directory
python -m pyrouge.test
```

Install PyRouge:

```shell
pip install pyrouge
```



### Performance

Dataset: CNN/DailyMail

|              Model              | ROUGE-1 | ROUGE-2 | ROUGE-L |                            Paper                             |
| :-----------------------------: | :-----: | :-----: | :-----: | :----------------------------------------------------------: |
|             LEAD 3              |  40.11  |  17.64  |  36.32  |                     Our data pre-process                     |
|             ORACLE              |  55.24  |  31.14  |  50.96  |                     Our data pre-process                     |
|    LSTM + Sequence Labeling     |  40.72  |  18.27  |  36.98  |                              -                               |
| Transformer + Sequence Labeling |  40.86  |  18.38  |  37.18  |                              -                               |
|     LSTM + Pointer Network      |  39.73  |  39.90  |  36.05  | [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/pdf/1704.04368.pdf)    |
|           BERTSUMEXT            |  42.83  |  19.92  |  39.18  | [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) |
|           TransSUMEXT           |  41.04  |  18.34  |  37.30  | [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) |
|           BERTSUMABS            |  41.17  |  18.72  |  38.16  | [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) |
|           TransSUMABS           |  40.17  |  17.81  |  37.12  | [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf) |



## Dependencies

- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.4.0
- [fastNLP](https://github.com/fastnlp/fastNLP) >= 0.6.0
- [pyrouge](https://github.com/bheinzerling/pyrouge) 0.1.3
  - You should fill your ROUGE path in specified location before running our code.
- [rouge](https://github.com/pltrdy/rouge) 1.0.0
- [transformers](https://github.com/huggingface/transformers) 2.5.1

**All code only supports running on Linux.**



### Install the latest FastNLP

```shell
pip install git+https://gitee.com/fastnlp/fastNLP@dev
```



