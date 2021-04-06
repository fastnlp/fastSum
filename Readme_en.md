# Summarization 

## Extractive Summarization


### Models

The models implemented in FastNLP:

1. Get To The Point: Summarization with Pointer-Generator Networks (See et al. 2017)
2. Searching for Effective Neural Extractive Summarization  What Works and What's Next (Zhong et al. 2019)
3. Fine-tune BERT for Extractive Summarization (Liu et al. 2019)


### Dataset

Summary task data set provided:
|                Name                 |                                                                        Paper                                                                         |                 Type                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| :---------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|            CNN/DailyMail            |               [Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](https://www.aclweb.org/anthology/K16-1028/)                |                 News                  |                                                                                                                                                                                                                                                                       Modified the database originally used for [passage-based question answering](https://arxiv.org/abs/1506.03340) tasks. "Both news providers supplement their articles with a number of bullet points, summarizing aspects of the information contained in the article. Of key importance is that these summary points are **abstractive** and do not simply copy sentences from the documents.""With a simple modification of the script, we restored all the summary bullets of each story in the original order to obtain **a multi-sentence summary**, where each bullet is treated as a sentence."                                                                                                                                                                                                                                                                       |
|                Xsum                 | [Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://arxiv.org/abs/1808.08745) |                 News                  |                                                                                                                                                                                                                                                                                                                                                                                                                           "Our extreme summarization dataset (which we call XSum) consists of BBC articles and accompanying **single sentence summaries**. Specifically, each article is prefaced with an introductory sentence (aka summary) which is professionally written, typically by the author of the article."                                                                                                                                                                                                                                                                                                                                                                                                                           |
| The New York Times Annotated Corpus |                                   [The New York Times Annotated Corpus](https://catalog.ldc.upenn.edu/LDC2008T19)                                    |                 News                  |                                                                                                                                                                                                                                                                                                                                                                                                         "As part of the New York Times' indexing procedures, most articles are manually summarized and tagged by **a staff of library scientists**. This collection contains over **650,000 article-summary pairs** which may prove to be useful in the development and evaluation of algorithms for automated document summarization. "                                                                                                                                                                                                                                                                                                                                                                                                          |
|                 DUC                 |                     [The Effects of Human Variation in DUC Summarization Evaluation](https://www.aclweb.org/anthology/W04-1003/)                     |                 News                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                 2002 Task4 -; 2003/2004 Task1&2:["Tasks 1 and 2 were essentially the same as in DUC 2003; Use the 50 TDT English clusters. **Given each document**, create a very short summary (<= 75 bytes) of the document."](https://duc.nist.gov/duc2004/ )                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|            arXiv PubMed             |                [A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents](https://arxiv.org/abs/1804.05685)                 |           Scientific Paper            |                                                                                                                                                                                                                                                                                                                                                                        "We also introduce two large-scale datasets of long and structured scientific papers obtained from arXiv and PubMed to support both training and evaluating models on the task of **long document summarization**." "Following these works, we take scientific papers as an example of long documents with discourse information, where their abstracts can be used as **ground-truth summaries**."                                                                                                                                                                                                                                                                                                                                                                        |
|               WikiHow               |                                [WikiHow: A Large Scale Text Summarization Dataset](https://arxiv.org/abs/1810.09305)                                 |            Knowledge Base             |                                                                                                                                                                                                                                                                       "The WikiHow knowledge base contains online articles describing a procedural task about various topics (from arts and entertainment to computers and electronics) with multiple methods or steps and new articles are added to it regularly." "Each step description starts with a bold line summarizing that step and is followed by a more detailed explanation." "The concatenation of all the bold lines (the summary sentences) of all the paragraphs to serve as the reference summary. The concatenation of all paragraphs (except the bold lines) to generate the article to be summarized."                                                                                                                                                                                                                                                                        |
|             Multi News              |        [Multi-News: a Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model](https://arxiv.org/abs/1906.01749)         | News&**Multi-Document Summarization** |                                                                                                                                                                                                                                                                                                                                                                                                                                                "Our dataset, which we call Multi-News, consists of news articles and **human-written summaries** of these articles from the site newser.com. Each summary is professionally written by editors and includes links to the original articles cited."                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|               BillSum               |                         [BillSum: A Corpus for Automatic Summarization of US Legislation](https://arxiv.org/abs/1910.00523)                          |              Legislation              |                                                                                                                                                                                                                                                                                                                                                                                              ["The BillSum dataset is the first corpus for automatic summarization of US Legislation. The corpus contains **the text of bills** and **human-written summaries** from the US Congress and California Legislature. It was published as a paper at the EMNLP 2019 New Frontiers in Summarization workshop."](https://www.kaggle.com/akornilo/billsum )                                                                                                                                                                                                                                                                                                                                                                                               |
|                 AMI                 |                                [The AMI meeting corpus: a pre-announcement](http://groups.inf.ed.ac.uk/ami/download/)                                |                Meeting                | ["As part of the development process, the project is collecting a corpus of **100 hours of meetings** using instrumentation that yields high quality, **synchronized multimodal recording**, with, for technical reasons, a focus on groups of four people." ](http://homepages.inf.ed.ac.uk/jeanc/amidata-MLMI-bookversion.final.pdf )["The AMI Meeting Corpus includes high quality, manually produced orthographic transcription for each individual speaker, including word-level timings that have derived by using a speech recognizer in forced alignment mode. It also contains a wide range of other annotations, not just for linguistic phenomena but also detailing behaviours in other modalities. These include dialogue acts; topic segmentation; **extractive and abstractive summaries**; named entities; the types of head gesture, hand gesture, and gaze direction that are most related to communicative intention; movement around the room; emotional state; and where heads are located on the video frames."](http://groups.inf.ed.ac.uk/ami/corpus/overview.shtmlhttp://groups.inf.ed.ac.uk/ami/corpus/overview.shtml ) |
|                ICSI                 |                                                 [ICSI Corpus](http://groups.inf.ed.ac.uk/ami/icsi/)                                                  |                Meeting                |                                                                                                                                                                                                                                                                                                                                                                                                     "The ICSI Meeting Corpus is an audio data set consisting of about 70 hours of meeting recordings."["We will be annotating two different kinds of summaries for this data, both aimed at the external researcher. One is abstractive. The other is extractive."](http://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_plus_NXT.zip )                                                                                                                                                                                                                                                                                                                                                                                                      |
|             Reddit TIFU             |                    [Abstractive Summarization of Reddit Posts with Multi-level Memory Networks](https://arxiv.org/abs/1811.00783)                    |           Online discussion           |                                                                                                                                                                                                                                                                                                                                                                                                                                               "We collect data from Reddit, which is a discussion forum platform with a large number of subreddits on diverse topics and interests." "Thus, we regard the body text as source, the title as short summary, and the TL;DR summary as long summary."                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|               SAMSum                |                 [SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization](https://arxiv.org/abs/1911.12237)                  |             Conversation              |                                                                                                                                                                                                                                                                                                                                                                                                                                "We asked linguists to create conversations similar to those they write on a daily basis, reflecting the proportion of topics of their real-life messenger conversations." "After collecting all of the conversations, we asked language experts to annotate them with summaries."                                                                                                                                                                                                                                                                                                                                                                                                                                 |

<!-- The download address of the public data set (CNN/DailyMail, Newsroom, arXiv, PubMed) after preprocessing:

- [Google Drive](https://drive.google.com/file/d/1uzeSdcLk5ilHaUTeJRNrf-_j59CQGe6r/view?usp=drivesdk)
- [Baidu Cloud Disk](https://pan.baidu.com/s/11qWnDjK9lb33mFZ9vuYlzA) (Key: h1px)

Undisclosed data sets (NYT, NYT 50, DUC) data processing part of the script is placed in the Dataloader folder. -->

You can run summarizationLoader.py to download and use them.



### Evaluation

#### FastRougeMetric

FastRougeMetric uses the unofficial ROUGE library implemented in python to quickly calculate rouge approximations during training.

[Source code](https://github.com/pltrdy/rouge)

In FastNLP, this method has been packaged into the FastRougeMetric class in Metric.py. You can easily call this API. o(^▽^)o

Firstly, it's necessary for you to install the rouge library using pip.

    pip install rouge

**Note(￣^￣): Due to the difference in implementation details, there is a 1-2 point difference between this result and the official ROUGE result, which can only be used as a rough estimate of the optimization trend of the training process.**



#### PyRougeMetric

PyRougeMetric uses the official ROUGE 1.5.5 evaluation library provided by [ROUGE: A Package for Automatic Evaluation of Summaries](https://www.aclweb.org/anthology/W04-1013).

Since the original ROUGE uses the perl interpreter, [pyrouge](https://github.com/bheinzerling/pyrouge) has carried out python packaging for it. PyRougeMetric packages it into a Metric class that can be easily used by trainers.

In order to use ROUGE 1.5.5, a series of dependent libraries need to be installed with **sudo** privileges.

1. You can refer to the [blog](https://blog.csdn.net/Hay54/article/details/78744912) for the installation of ROUGE in Ubuntu
2. Configure wordnet:
```shell
$ cd ~/rouge/RELEASE-1.5.5/data/WordNet-2.0-Exceptions/
$ ./buildExeptionDB.pl . exc WordNet-2.0.exc.db
$ cd ../
$ ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```
3. Install pyrouge:
```shell
$ git clone https://github.com/bheinzerling/pyrouge
$ cd pyrouge
$ python setup.py install
```
4. Test whether ROUGE is installed correctly:
```shell
$ pyrouge_set_rouge_path /absolute/path/to/ROUGE-1.5.5/directory
$ python -m pyrouge.test
```




### Dataset_loader

- **SummarizationLoader** is used to read the processed jsonl format data set and return the following fields:
    - text
    - summary
    - domain (Optional)
    - tag (Optional)
    - labels

- **BertSumLoader** is used to read the data set input as BertSum (Liu 2019) and returns the following field:
  - article (Vocabulary ID&Article length <= 512)
  - segmet_id (0/1)
  - cls_id
  - label



### Train Cmdline

#### [Baseline](./fastsum/Baseline)

LSTM + Sequence Labeling

    python train.py --cuda --gpu <gpuid> --sentence_encoder deeplstm --sentence_decoder SeqLab --save_root <savedir> --log_root <logdir> --lr_descent --grad_clip --max_grad_norm 10

Transformer + Sequence Labeling

    python train.py --cuda --gpu <gpuid> --sentence_encoder transformer --sentence_decoder SeqLab --save_root <savedir> --log_root <logdir> --lr_descent --grad_clip --max_grad_norm 10



#### [BertSum](./fastsum/Bertsum)

```shell
python train_BertSum.py --mode train --save_path save --label_type greedy --batch_size 8
```



### Performance and Hyperparameters

|              Model              | ROUGE-1 | ROUGE-2 | ROUGE-L |                    Paper                    |
| :-----------------------------: | :-----: | :-----: | :-----: | :-----------------------------------------: |
|             LEAD 3              |  40.11  |  17.64  |  36.32  |            our data pre-process             |
|             ORACLE              |  55.24  |  31.14  |  50.96  |            our data pre-process             |
|    LSTM + Sequence Labeling     |  40.72  |  18.27  |  36.98  |                                             |
| Transformer + Sequence Labeling |  40.86  |  18.38  |  37.18  |                                             |
|     LSTM + Pointer Network      |    -    |    -    |    -    |                                             |
|  Transformer + Pointer Network  |    -    |    -    |    -    |                                             |
|             BERTSUM             |  42.71  |  19.76  |  39.03  | Fine-tune BERT for Extractive Summarization |
|         LSTM+PN+BERT+RL         |    -    |    -    |    -    |                                             |



## Abstractive Summarization
Still in Progress...