# from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util.config import Config
from numpy import random
from data_util.logging import logger
import numpy as np

from data_util import data

from torch.autograd import Variable

config = Config()
use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


def is_infinite_dist(distss):
    is_nan_list = torch.isnan(distss)
    if (is_nan_list.max() == 1).item() == 1:
        return True
    return False


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)


class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        # 一个beam可以理解为存储着一组到当前位置字的“路线”
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token], log_probs=self.log_probs + [log_prob], state=state, context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    # seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)

        _, indices = torch.sort(seq_lens, dim=0, descending=True)
        _, desorted_indices = torch.sort(indices, dim=0)
        embedded = embedded.index_select(0, indices)
        lengths = list(seq_lens[indices])

        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        # print("inside of encoder : encoder outputs size : ",encoder_outputs.size())
        encoder_outputs = encoder_outputs.index_select(0, desorted_indices)
        hn, cn = hidden
        hn = hn.index_select(1, desorted_indices)
        cn = cn.index_select(1, desorted_indices)
        hidden = (hn, cn)
        encoder_outputs = encoder_outputs.contiguous()

        # encoder_outputs, hidden = self.lstm(embedded)
        # encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = 1 x b x hidden_dim


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())
        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1)  # B x hidden_dim * 3
        output = self.out1(output)  # B x hidden_dim
        # output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
            final_dist = vocab_dist_.scatter_add_(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        if is_infinite_dist(final_dist):
            print("catch nan final dist: ", final_dist)
            print("embedding weight:", self.embedding.weight)
            print("y_t_1_embd: ", y_t_1_embd)
            print("y_t_1:", y_t_1)

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage


class Model(torch.nn.Module):
    def __init__(self, vocab=None):
        super(Model, self).__init__()
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()
        self.vocab = vocab

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

    def get_input_from_batch(self, enc_len, enc_input, enc_input_extend_vocab, article_oovs):
        enc_lens = enc_len
        batch_size = len(enc_lens)
        max_enc_seq_len = np.max(np.array(enc_lens.cpu()))
        enc_padding_mask = np.zeros((batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, lens in enumerate(enc_lens):
            for j in range(lens):
                enc_padding_mask[i][j] = 1

        # enc_batch = Variable(torch.from_numpy(np.array(enc_input)).long())
        enc_batch = enc_input
        enc_padding_mask = Variable(torch.from_numpy(enc_padding_mask)).float()
        extra_zeros = None
        enc_batch_extend_vocab = None

        if config.pointer_gen:
            enc_batch_extend_vocab = enc_input_extend_vocab
            # max_art_oovs is the max over all the article oov list in the batch
            # max_art_oovs = max([len(article_oov) for article_oov in article_oovs])
            max_art_oovs = 0
            for article_oov in article_oovs:
                if "N O N E" in article_oov:
                    continue
                else:
                    max_art_oovs = max(max_art_oovs, len(article_oov))
            if max_art_oovs > 0:
                extra_zeros = Variable(torch.zeros((batch_size, max_art_oovs)))

        c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))

        coverage = None
        if config.is_coverage:
            coverage = Variable(torch.zeros(enc_batch.size()))
            # coverage = Variable(torch.zeros(batch_size, max_enc_seq_len))

        if use_cuda:
            enc_batch = enc_batch.cuda()
            enc_padding_mask = enc_padding_mask.cuda()
            if enc_batch_extend_vocab is not None:
                enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
            if extra_zeros is not None:
                extra_zeros = extra_zeros.cuda()
            c_t_1 = c_t_1.cuda()
            if coverage is not None:
                coverage = coverage.cuda()
        return enc_batch, enc_lens, enc_padding_mask, extra_zeros, enc_batch_extend_vocab, c_t_1, coverage

    def get_output_from_batch(self, dec_len, dec_input):
        # get_output_from_batch
        dec_batch = dec_input
        dec_lens = dec_len
        max_dec_len = np.max(np.array(dec_lens.cpu()))

        batch_size = len(dec_lens)

        # dec_padding_mask = np.zeros((batch_size, config.max_dec_steps), dtype=np.float32)
        dec_padding_mask = np.zeros((batch_size, min(max_dec_len, config.max_dec_steps)), dtype=np.float32)

        # Fill in the numpy arrays
        for i, lens in enumerate(dec_lens):
            for j in range(lens):
                dec_padding_mask[i][j] = 1
        dec_padding_mask = Variable(torch.from_numpy(dec_padding_mask)).float()

        dec_lens_var = dec_lens

        # target_batch = Variable(torch.from_numpy(np.array(target))).long()

        if use_cuda:
            dec_batch = dec_batch.cuda()
            dec_padding_mask = dec_padding_mask.cuda()
            dec_lens_var = dec_lens_var.cuda()
            # target_batch = target_batch.cuda()
        # target = target_batch

        return dec_batch, max_dec_len, dec_padding_mask, dec_lens_var

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def forward(self, enc_len, enc_input, dec_input, dec_len, article_oovs, enc_input_extend_vocab):
        # print("start forward",enc_input.size())
        enc_batch, enc_lens, enc_padding_mask, extra_zeros, enc_batch_extend_vocab, c_t_1, coverage = \
            self.get_input_from_batch(enc_len, enc_input, enc_input_extend_vocab, article_oovs)
        dec_batch, max_dec_len, dec_padding_mask, dec_lens_var = self.get_output_from_batch(dec_len, dec_input)

        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(enc_batch, enc_lens)
        s_t_1 = self.reduce_state(encoder_hidden)

        # step_losses = []
        list_final_dist = []
        list_coverage = []
        list_attn = []
        pred = None

        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1,
                                                                                     encoder_outputs,
                                                                                     encoder_feature,
                                                                                     enc_padding_mask, c_t_1,
                                                                                     extra_zeros,
                                                                                     enc_batch_extend_vocab,
                                                                                     coverage, di)

            _, max_position = torch.max(final_dist, 1)
            max_position = max_position.unsqueeze(1)
            if pred is None:
                pred = max_position
            else:
                pred = torch.cat((pred, max_position), 1)

            list_final_dist.append(final_dist)
            list_attn.append(attn_dist)
            if config.is_coverage:
                list_coverage.append(coverage)
                coverage = next_coverage

        return {"pred": pred, "list_final_dist": list_final_dist, "list_coverage": list_coverage,
                "list_attn": list_attn, "max_dec_len": max_dec_len, "dec_padding_mask": dec_padding_mask,
                "dec_lens_var": dec_lens_var}

    def unpadding(self, enc_len, enc_input, enc_input_extend_vocab):
        return enc_input[:enc_len], enc_input_extend_vocab[:enc_len]

    def decode(self, enc_len, enc_input, article_oovs, enc_input_extend_vocab):
        enc_input, enc_input_extend_vocab = self.unpadding(enc_len, enc_input, enc_input_extend_vocab)

        # print("before: ", enc_input.size()," ",enc_len.size()," ",(np.array(article_oovs)).shape," ",enc_input_extend_vocab.size())
        enc_input = enc_input.unsqueeze(0).expand(config.beam_size, list(enc_input.size())[0]).contiguous()
        enc_len = enc_len.unsqueeze(0).expand(config.beam_size).contiguous()
        enc_input_extend_vocab = enc_input_extend_vocab.unsqueeze(0).expand(config.beam_size,
                                                                            list(enc_input_extend_vocab.size())[
                                                                                0]).contiguous()
        # print("after: ", enc_input.size(), " ", enc_len.size(), " ", np.array(article_oovs).shape, " ", enc_input_extend_vocab.size())

        enc_batch, enc_lens, enc_padding_mask, extra_zeros, enc_batch_extend_vocab, c_t_0, coverage_t_0 = \
            self.get_input_from_batch(enc_len, enc_input, enc_input_extend_vocab, [article_oovs])

        # print("-----in decoder: sizeof enc_batch and enc_lens: ",enc_batch.size()," ",enc_lens.size())
        # print("------length infomation: ",enc_lens)
        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(enc_batch, enc_lens)
        # print("******in decoder: sizeof encoder_outputs and encoder_feature: ", encoder_outputs.size(), " ", encoder_feature.size())
        s_t_0 = self.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.to_index(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None)) for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < len(self.vocab) else self.vocab.to_index(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h = []
            all_state_c = []
            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            # print("-"*20)
            # print(article_oovs)
            # print(extra_zeros)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.decoder(y_t_1, s_t_1,
                                                                              encoder_outputs, encoder_feature,
                                                                              enc_padding_mask, c_t_1,
                                                                              extra_zeros, enc_batch_extend_vocab,
                                                                              coverage_t_1, steps)
            # print("final_dist: ", final_dist)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)
            # print("topk_log_probs: ", topk_log_probs)
            # print("topk_ids: ", topk_ids)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    # print("log_probs.size: ",log_probs.size, "topk_ids.size:  ", topk_ids.size())
                    # print("i,j:  ", i, ",", j, "topk_ids: ",topk_ids)
                    # print(topk_ids[i])
                    # print(topk_ids[i,j])
                    # print("topk_ids[i,j].item:  ", topk_ids[i, j].item())
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.to_index(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)
        output_ids = [int(t) for t in beams_sorted[0].tokens[1:]]

        return output_ids

    def predict(self, enc_len, enc_input, article_oovs, enc_input_extend_vocab):
        # batch should have only one example
        # print("start predict",enc_input.size())
        output_ids = []
        batch_size, seq_len = list(enc_input.size())

        enc_len_tmp = enc_len
        enc_input_tmp = enc_input
        article_oovs_tmp = article_oovs
        enc_input_extend_vocab_tmp = enc_input_extend_vocab
        for _num in range(batch_size):
            enc_len = enc_len_tmp[_num]
            enc_input = enc_input_tmp[_num]
            article_oovs = article_oovs_tmp[_num]
            enc_input_extend_vocab = enc_input_extend_vocab_tmp[_num]
            pred = self.decode(enc_len, enc_input, article_oovs, enc_input_extend_vocab)
            output_ids.append(pred)

        return {"prediction": output_ids}
