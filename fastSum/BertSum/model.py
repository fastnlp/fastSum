import torch
from torch import nn

from fastNLP.modules.encoder import BertModel


class Classifier(nn.Module):
    def __init__(self, hidden_size: int):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)  # 抽取式摘要在这里就是对每个句子进行二分类
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor, mask_cls: torch.Tensor) -> torch.Tensor:
        """

        :param inputs: shape: (N, doc_len, hidden_size); doc_len 表示某 doc 有几个句子; inputs[i] 表示第 i 个 doc 中每个句子的句向量
        :param mask_cls: shape: (N, doc_len); 用于指正每个 doc 的实际有几个句子（因为填充了）, 填充部分为 0，真实部分为 1
        :return: shape: (N, doc_len)；表示抽取每个句子的概率大小
        """
        # (N, doc_len, hidden_size=第一层词向量的维度) --linear--> (N, doc_len, 1) --squeeze--> (N, doc_len)
        h = self.linear(inputs).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class BertSum(nn.Module):
    
    def __init__(self, hidden_size: int = 768):
        super(BertSum, self).__init__()
        
        self.hidden_size = hidden_size

        self.encoder = BertModel.from_pretrained('data/uncased_L-12_H-768_A-12')
        self.decoder = Classifier(self.hidden_size)

    def forward(self, article: torch.Tensor, segment_id: torch.Tensor, cls_id: torch.Tensor):
        """

        :param article: shape: (N, seq_len); seq_len 表示最长的文档长度（有几个单词）; 0 填充
        :param segment_id:
        :param cls_id:  [CLS] 的所有位置；shape: (N, doc_len); doc_len 表示某 doc 有几个句子; -1 填充（见 BertSumLoader）
        :return:
        """
         
        # print(article.device)
        # print(segment_id.device)
        # print(cls_id.device)

        input_mask = 1 - torch.eq(article, 0).long()  # 1 有效
        mask_cls = 1 - torch.eq(cls_id, -1).long()  # 1 有效
        # assert input_mask.size() == article.size()
        # assert mask_cls.size() == cls_id.size()

        bert_out = self.encoder(article, token_type_ids=segment_id, attention_mask=input_mask)
        bert_out = bert_out[0][-1]  # last layer; shape: (N, sequence_length, 768=hidden_size)

        # torch.arange(bert_out.size(0)).unsqueeze(1) shape: (N, 1) ; clss 为 (N, doc_len)；配合 broadcast 机制下，选择 cls 的数值（即 bert_out[xxx[i][j], clss[i][j], :]）
        # sents_vec shape: (N, doc_len, 768=hidden_size) 相当于每个句子的句向量
        sent_emb = bert_out[torch.arange(bert_out.size(0)).unsqueeze(1), cls_id]
        sent_emb = sent_emb * mask_cls.unsqueeze(-1).float()
        # assert sent_emb.size() == (article.size(0), cls_id.size(1), self.hidden_size)  # [batch_size, seq_len, hidden_size]

        sent_scores = self.decoder(sent_emb, mask_cls)  # shape: (N, doc_len)；表示抽取每个句子的概率大小
        # assert sent_scores.size() == (article.size(0), cls_id.size(1))

        return {'pred': sent_scores, 'mask': mask_cls}
