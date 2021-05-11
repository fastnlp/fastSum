import copy

import torch
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
# from models.predictor import build_predictor
from others.utils import rouge_results_to_str, test_rouge, tile
from translate.beam import GNMTGlobalScorer


def get_generator(vocab_size, dec_hidden_size):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    # generator.to(device)

    return generator


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if (large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if (self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads,
                                     intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if (args.max_pos > 512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

    def forward(self, text_ids, segment_ids, cls_ids):
        input_mask = 1 - (text_ids == 0).long()
        mask_cls = 1 - (cls_ids == -1).to(torch.uint8)
        assert input_mask.size() == text_ids.size()
        assert mask_cls.size() == cls_ids.size()

        top_vec = self.bert(text_ids, segment_ids, input_mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), cls_ids]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return {'pred': sent_scores, 'mask': mask_cls}


class AbsSummarizer(nn.Module):
    def __init__(self, args, checkpoint=None, bert_from_extractive=None, symbols=None, tokenizer=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        # self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
        self.global_scorer = GNMTGlobalScorer(args.alpha, length_penalty='wu')
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']
        self.vocab = tokenizer

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if (args.max_pos > 512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(args.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size)
        self.generator[0].weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if (args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        # self.to(device)

    def forward(self, text_ids, summary_ids, segment_ids):
        # 默认设定pad id为0
        mask_src = 1 - (text_ids == 0).long()
        top_vec = self.bert(text_ids, segment_ids, mask_src)
        dec_state = self.decoder.init_decoder_state(text_ids, top_vec)
        # print("AbsSummarizer.forward: ")
        # print(summary_ids[:, :-1].size(), top_vec.size())
        pred, state = self.decoder(summary_ids[:, :-1], top_vec, dec_state)
        return {"pred": pred}

    def predict(self, text_ids, segment_ids, summary_ids=None):
        # 当predict在训练的过程中的validation的过程被调用，则返回forward的结果
        if self.args.mode == "train":
            return self.forward(text_ids, summary_ids, segment_ids)

        min_length = self.args.min_length
        max_length = self.args.max_length
        beam_size = self.args.beam_size
        # batch_size = self.args.batch_size
        batch_size = text_ids.size()[0]

        # 默认设定pad id为0
        mask_src = 1 - (text_ids == 0).long()
        top_vec = self.bert(text_ids, segment_ids, mask_src)
        dec_states = self.decoder.init_decoder_state(text_ids, top_vec)
        device = top_vec.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(top_vec, beam_size, dim=0)
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)
        # print("AbsSummarizer.predict: ")
        # print(alive_seq.size(), batch_size, beam_size)
        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        # results["gold_score"] = [0] * batch_size

        for step in range(max_length):
            # print("AbsSummarizer.predict:")
            # print(alive_seq.size())
            # print("alive seq: ", alive_seq.size())
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0, 1)
            # print("decoder input: ", decoder_input.size())

            # print(decoder_input.size(), src_features.size())
            dec_out, dec_states = self.decoder(decoder_input, src_features, dec_states,
                                               step=step)

            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0, 1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if (self.args.block_trigram):
                cur_len = alive_seq.size(1)
                if (cur_len > 3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = ' '.join(words).replace(' ##', '').split()
                        if (len(words) <= 3):
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)
            # print("alive_seq after cat: ", alive_seq.size())

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # print(non_finished)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
                # print("alive_seq predictions.index_select ", alive_seq.size())
            # Reorder states.
            select_indices = batch_index.view(-1)
            # print("select indices: {}".format(select_indices.size()))
            # print("src_features before index_select:", src_features.size())
            src_features = src_features.index_select(0, select_indices)
            # print("src_features: {}".format(src_features.size()))
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results
