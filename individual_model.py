import numpy as np
import torch, json
import torch.nn as nn
from graph import Graph
from transformers import BertModel, BertConfig, RobertaModel, XLMRobertaModel
from util import *


def multi2binary(label_idxs):
    mask = (label_idxs == -100)
    binary = (label_idxs > 0).long().masked_fill(mask, -100)
    return binary


def remove_duplicate_rels(relation_preds, max_entity_num):
    batch_size = relation_preds.shape[0]
    batch_mask = []
    for bid in range(batch_size):
        mask = []
        for i in range(max_entity_num):
            for j in range(max_entity_num):
                if relation_preds[bid][i * max_entity_num + j].item() > 0 or relation_preds[bid][
                    j * max_entity_num + i].item() > 0:
                    relation_preds[bid][i * max_entity_num + j] = 1
                    relation_preds[bid][j * max_entity_num + i] = 1

                if j <= i:
                    mask.append(1)
                else:
                    mask.append(0)
        batch_mask.append(mask)

    batch_mask = torch.cuda.LongTensor(batch_mask).bool()
    return relation_preds.masked_fill(batch_mask, 0)


def compute_word_reps_avg(piece_reprs, component_idxs):
    batch_word_reprs = []
    batch_size, _, _ = piece_reprs.shape
    _, num_words, _ = component_idxs.shape
    for bid in range(batch_size):
        word_reprs = []
        for wid in range(num_words):
            wrep = torch.mean(piece_reprs[bid][component_idxs[bid][wid][0]: component_idxs[bid][wid][1]], dim=0)
            word_reprs.append(wrep)
        word_reprs = torch.stack(word_reprs, dim=0)  # [num words, rep dim]
        batch_word_reprs.append(word_reprs)
    batch_word_reprs = torch.stack(batch_word_reprs, dim=0)  # [batch size, num words, rep dim]
    return batch_word_reprs


def compute_span_reprs(word_reprs, span_idxs):
    '''
    word_reprs.shape: [batch size, num words, word dim]
    span_idxs.shape: [batch size, num spans, 2]
    '''
    batch_span_reprs = []
    batch_size, _, _ = word_reprs.shape
    _, num_spans, _ = span_idxs.shape
    for bid in range(batch_size):
        span_reprs = []
        for sid in range(num_spans):
            start, end = span_idxs[bid][sid]
            words = word_reprs[bid][start: end]  # [span size, word dim]
            span_reprs.append(torch.mean(words, dim=0))
        span_reprs = torch.stack(span_reprs, dim=0)  # [num spans, word dim]
        batch_span_reprs.append(span_reprs)
    batch_span_reprs = torch.stack(batch_span_reprs, dim=0)  # [batch size, num spans, word dim]
    return batch_span_reprs


def compute_binary_reprs(obj1_reprs, obj2_reprs):  # note that, (obj1, obj2) != (obj2, obj1)
    batch_size, _, rep_dim = obj1_reprs.shape
    num_obj1 = obj1_reprs.shape[1]
    num_obj2 = obj2_reprs.shape[1]

    cloned_obj1s = obj1_reprs.repeat(1, 1, num_obj2).view(batch_size, -1, rep_dim)
    cloned_obj2s = obj2_reprs.repeat(1, num_obj1, 1).view(batch_size, -1, rep_dim)
    return cloned_obj1s, cloned_obj2s


def log_sum_exp(tensor, dim=0, keepdim: bool = False):
    """LogSumExp operation used by CRF."""
    m, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - m
    else:
        stable_vec = tensor - m.unsqueeze(dim)
    return m + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def sequence_mask(lens, max_len=None):
    """Generate a sequence mask tensor from sequence lengths, used by CRF."""
    batch_size = lens.size(0)
    if max_len is None:
        max_len = lens.max().item()
    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp
    return mask


def token_lens_to_offsets(token_lens):
    """Map token lengths to first word piece indices, used by the sentence
    encoder.
    :param token_lens (list): token lengths (word piece numbers)
    :return (list): first word piece indices (offsets)
    """
    max_token_num = max([len(x) for x in token_lens])
    offsets = []
    for seq_token_lens in token_lens:
        seq_offsets = [0]
        for l in seq_token_lens[:-1]:
            seq_offsets.append(seq_offsets[-1] + l)
        offsets.append(seq_offsets + [-1] * (max_token_num - len(seq_offsets)))
    return offsets


def token_lens_to_idxs(token_lens):
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs = []
    for seq_token_lens in token_lens:
        seq_idxs = []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.append([offset, offset + token_len])
            offset += token_len
        seq_idxs.extend([[-1, 0]] * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
    return idxs, max_token_num, max_token_len


def graphs_to_node_idxs(graphs):
    """
    :param graphs (list): A list of Graph objects.
    :return: entity/trigger index matrix, mask tensor, max number, and max length
    """
    entity_idxs = []
    trigger_idxs = []

    max_entity_num = max(max(graph.entity_num for graph in graphs), 1)
    max_trigger_num = max(max(graph.trigger_num for graph in graphs), 1)
    max_entity_len = max(max([e[1] - e[0] for e in graph.entities] + [1])
                         for graph in graphs)
    max_trigger_len = max(max([t[1] - t[0] for t in graph.triggers] + [1])
                          for graph in graphs)
    num_nodes = max_trigger_num + max_entity_num + max_trigger_num * max_entity_num + max_entity_num ** 2
    batch_node_masks = []
    for bid, graph in enumerate(graphs):
        tmp = np.zeros(num_nodes)
        tmp[:graph.trigger_num] = 1
        tmp[max_trigger_num: max_trigger_num + graph.entity_num] = 1
        for k in range(graph.trigger_num):
            tmp[
            max_trigger_num + max_entity_num + k * max_entity_num: max_trigger_num + max_entity_num + k * max_entity_num + graph.entity_num] = 1
        for k in range(graph.entity_num):
            tmp[
            max_trigger_num + max_entity_num + max_trigger_num * max_entity_num + k * max_entity_num: max_trigger_num + max_entity_num + max_trigger_num * max_entity_num + k * max_entity_num + graph.entity_num] = 1

        node_mask = np.outer(tmp, tmp).tolist()  # [num nodes, 1] x [1, num nodes] -> [num nodes, num nodes]

        for k in range(num_nodes):
            node_mask[k][k] = 1

        batch_node_masks.append(node_mask)

        seq_entity_idxs = []
        seq_trigger_idxs = []

        for entity in graph.entities:
            seq_entity_idxs.append([entity[0], entity[1]])
        seq_entity_idxs.extend([[0, 1]] * (max_entity_num - graph.entity_num))
        entity_idxs.append(seq_entity_idxs)

        for trigger in graph.triggers:
            seq_trigger_idxs.append([trigger[0], trigger[1]])

        seq_trigger_idxs.extend([[0, 1]] * (max_trigger_num - graph.trigger_num))

        trigger_idxs.append(seq_trigger_idxs)

    batch_node_masks = torch.cuda.LongTensor(batch_node_masks).eq(0)
    return (
        entity_idxs, max_entity_num, max_entity_len,
        trigger_idxs, max_trigger_num, max_trigger_len, batch_node_masks
    )


def graphs_to_label_idxs(graphs, max_entity_num=-1, max_trigger_num=-1,
                         relation_directional=False,
                         symmetric_relation_idxs=None):
    """Convert a list of graphs to label index and mask matrices
    :param graphs (list): A list of Graph objects.
    :param max_entity_num (int) Max entity number (default = -1).
    :param max_trigger_num (int) Max trigger number (default = -1).
    """
    if max_entity_num == -1:
        max_entity_num = max(max([g.entity_num for g in graphs]), 1)
    if max_trigger_num == -1:
        max_trigger_num = max(max([g.trigger_num for g in graphs]), 1)
    (
        batch_entity_idxs, batch_entity_mask,
        batch_trigger_idxs, batch_trigger_mask,
        batch_relation_idxs, batch_relation_mask,
        batch_role_idxs, batch_role_mask
    ) = [[] for _ in range(8)]
    for graph in graphs:
        (
            entity_idxs, entity_mask, trigger_idxs, trigger_mask,
            relation_idxs, relation_mask, role_idxs, role_mask,
        ) = graph.to_label_idxs(max_entity_num, max_trigger_num,
                                relation_directional=relation_directional,
                                symmetric_relation_idxs=symmetric_relation_idxs)
        batch_entity_idxs.append(entity_idxs)
        batch_entity_mask.append(entity_mask)
        batch_trigger_idxs.append(trigger_idxs)
        batch_trigger_mask.append(trigger_mask)
        batch_relation_idxs.append(relation_idxs)
        batch_relation_mask.append(relation_mask)
        batch_role_idxs.append(role_idxs)
        batch_role_mask.append(role_mask)
    return (
        batch_entity_idxs, batch_entity_mask,
        batch_trigger_idxs, batch_trigger_mask,
        batch_relation_idxs, batch_relation_mask,
        batch_role_idxs, batch_role_mask
    )


def generate_pairwise_idxs(num1, num2):
    idxs = []
    for i in range(num1):
        for j in range(num2):
            idxs.append(i)
            idxs.append(j + num1)
    return idxs


class Linears(nn.Module):
    """Multiple linear layers with Dropout."""

    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs


class CRF(nn.Module):
    def __init__(self, label_vocab, bioes=False):
        super(CRF, self).__init__()

        self.label_vocab = label_vocab
        self.label_size = len(label_vocab) + 2
        # self.same_type = self.map_same_types()
        self.bioes = bioes

        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        self.transition.data[:, self.end] = -100.0
        self.transition.data[self.start, :] = -100.0

        for label, label_idx in self.label_vocab.items():
            if label.startswith('I-') or label.startswith('E-'):
                self.transition.data[label_idx, self.start] = -100.0
            if label.startswith('B-') or label.startswith('I-'):
                self.transition.data[self.end, label_idx] = -100.0

        for label_from, label_from_idx in self.label_vocab.items():
            if label_from == 'O':
                label_from_prefix, label_from_type = 'O', 'O'
            else:
                label_from_prefix, label_from_type = label_from.split('-', 1)

            for label_to, label_to_idx in self.label_vocab.items():
                if label_to == 'O':
                    label_to_prefix, label_to_type = 'O', 'O'
                else:
                    label_to_prefix, label_to_type = label_to.split('-', 1)

                if self.bioes:
                    is_allowed = any(
                        [
                            label_from_prefix in ['O', 'E', 'S']
                            and label_to_prefix in ['O', 'B', 'S'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix in ['I', 'E']
                            and label_from_type == label_to_type
                        ]
                    )
                else:
                    is_allowed = any(
                        [
                            label_to_prefix in ['B', 'O'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix == 'I'
                            and label_from_type == label_to_type
                        ]
                    )
                if not is_allowed:
                    self.transition.data[
                        label_to_idx, label_from_idx] = -100.0

    def pad_logits(self, logits):
        """Pad the linear layer output with <SOS> and <EOS> scores.
        :param logits: Linear layer output (no non-linear function).
        """
        batch_size, seq_len, _ = logits.size()
        pads = logits.new_full((batch_size, seq_len, 2), -100.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, lens):
        batch_size, seq_len = labels.size()

        # A tensor of size batch_size * (seq_len + 2)
        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        pad_stop = labels.new_full((1,), self.end, requires_grad=False)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, self.label_size,
                                          self.label_size)
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), self.label_size)
        # score of jumping to a tag
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    def calc_unary_score(self, logits, labels, lens):
        """Checked"""
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        return scores

    def calc_gold_score(self, logits, labels, lens):
        """Checked"""
        unary_score = self.calc_unary_score(logits, labels, lens).sum(
            1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        batch_size, _, _ = logits.size()
        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0
        lens_ = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  self.label_size,
                                                  self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (lens_ > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            lens_ = lens_ - 1

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def loglik(self, logits, labels, lens):
        norm_score = self.calc_norm_score(logits, lens)
        gold_score = self.calc_gold_score(logits, labels, lens)
        return gold_score - norm_score

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, _, n_labels = logits.size()
        vit = logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def calc_conf_score_(self, logits, labels):
        batch_size, _, _ = logits.size()

        logits_t = logits.transpose(1, 0)
        scores = [[] for _ in range(batch_size)]
        pre_labels = [self.start] * batch_size
        for i, logit in enumerate(logits_t):
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand(batch_size,
                                                            self.label_size,
                                                            self.label_size)
            score = logit_exp + trans_exp
            score = score.view(-1, self.label_size * self.label_size) \
                .softmax(1)
            for j in range(batch_size):
                cur_label = labels[j][i]
                cur_score = score[j][cur_label * self.label_size + pre_labels[j]]
                scores[j].append(cur_score)
                pre_labels[j] = cur_label
        return scores


class TranscriptNER(nn.Module):
    def __init__(self,
                 config,
                 vocabs):
        super().__init__()
        self.config = config
        self.vocabs = vocabs
        self.entity_label_stoi = vocabs['entity_label']  # BIO tags for [ORG, PER, GPE, ...]
        self.entity_type_stoi = vocabs['entity_type']  # [ORG, PER, GPE, ...]

        self.entity_label_itos = {i: s for s, i in self.entity_label_stoi.items()}
        self.entity_type_itos = {i: s for s, i in self.entity_type_stoi.items()}

        self.entity_label_num = len(self.entity_label_stoi)
        self.entity_type_num = len(self.entity_type_stoi)

        if self.config.bert_model_name.startswith('bert'):
            self.bert = BertModel.from_pretrained(config.bert_model_name,
                                                  cache_dir=config.bert_cache_dir,
                                                  output_hidden_states=True)
        elif self.config.bert_model_name == 'roberta-large':
            self.bert = RobertaModel.from_pretrained(config.bert_model_name,
                                                     cache_dir=config.bert_cache_dir,
                                                     output_hidden_states=True)
        elif self.config.bert_model_name == 'xlm-roberta-large':
            self.bert = XLMRobertaModel.from_pretrained(config.bert_model_name,
                                                        cache_dir=config.bert_cache_dir,
                                                        output_hidden_states=True)

        self.bert_dim = 768 if config.bert_model_name == "bert-base-multilingual-cased" else 1024
        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.bert_dim *= 2

        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy

        self.entity_label_ffn = nn.Linear(self.bert_dim, self.entity_label_num,
                                          bias=config.linear_bias)

        self.entity_type_ffn = Linears([self.config.node_dim, self.config.entity_hidden_num, config.hidden_num,
                                        self.entity_type_num],
                                       dropout_prob=config.linear_dropout,
                                       bias=config.linear_bias,
                                       activation=config.linear_activation)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        self.entity_crf = CRF(self.entity_label_stoi, bioes=False)

        self.cuda()

    def encode(self, piece_idxs, attention_masks, token_lens):
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        bert_outputs = all_bert_outputs[0]

        if self.use_extra_bert:
            extra_bert_outputs = all_bert_outputs[2][self.extra_bert]
            bert_outputs = torch.cat([bert_outputs, extra_bert_outputs], dim=2)

        idxs, token_num, token_len = token_lens_to_idxs(token_lens)
        idxs = piece_idxs.new(idxs) + 1
        bert_outputs = compute_word_reps_avg(bert_outputs, idxs)

        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs

    def forward(self, batch):
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()

        entity_label_scores = self.entity_label_ffn(bert_outputs)
        entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)
        entity_label_loglik = self.entity_crf.loglik(entity_label_scores,
                                                     batch.entity_label_idxs,
                                                     batch.token_nums)

        loss = - entity_label_loglik.mean()
        return loss

    def predict(self, batch):
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()

        entity_label_scores = self.entity_label_ffn(bert_outputs)
        entity_label_scores = self.entity_crf.pad_logits(entity_label_scores)

        _, entity_label_preds = self.entity_crf.viterbi_decode(entity_label_scores,
                                                               batch.token_nums)

        entities = tag_paths_to_spans(entity_label_preds,
                                      batch.token_nums,
                                      self.entity_label_stoi)

        batch_graphs = []

        for i in range(batch_size):
            seq_entities = entities[i]
            graph = self.build_graph(seq_entities)
            batch_graphs.append(graph)

        return batch_graphs

    def build_graph(self, spans):
        graph = Graph.empty_graph(self.vocabs)

        for start, end, entity_type in spans:
            label = self.entity_type_stoi[entity_type]
            graph.add_entity(start, end, label, 0, 0)

        return graph


class IndividualSpeakerIdentifier(nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        self.config = config

        if self.config.bert_model_name.startswith('bert'):
            self.bert = BertModel.from_pretrained(config.bert_model_name,
                                                  cache_dir=config.bert_cache_dir,
                                                  output_hidden_states=True)
        elif self.config.bert_model_name == 'roberta-large':
            self.bert = RobertaModel.from_pretrained(config.bert_model_name,
                                                     cache_dir=config.bert_cache_dir,
                                                     output_hidden_states=True)
        elif self.config.bert_model_name == 'xlm-roberta-large':
            self.bert = XLMRobertaModel.from_pretrained(config.bert_model_name,
                                                        cache_dir=config.bert_cache_dir,
                                                        output_hidden_states=True)

        self.bert_dim = 768 if config.bert_model_name == "bert-base-multilingual-cased" else 1024
        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.bert_dim *= 2

        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.multi_piece = config.multi_piece_strategy

        self.role_idn_ffn = Linears([self.bert_dim * 4, config.hidden_num,
                                     2],
                                    dropout_prob=config.linear_dropout,
                                    bias=config.linear_bias,
                                    activation=config.linear_activation)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        self.cuda()

    def encode(self, piece_idxs, attention_masks, token_lens):
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        bert_outputs = all_bert_outputs[0]

        if self.use_extra_bert:
            extra_bert_outputs = all_bert_outputs[2][self.extra_bert]
            bert_outputs = torch.cat([bert_outputs, extra_bert_outputs], dim=2)

        idxs, token_num, token_len = token_lens_to_idxs(token_lens)
        idxs = piece_idxs.new(idxs) + 1
        bert_outputs = compute_word_reps_avg(bert_outputs, idxs)

        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs

    def scores(self, bert_outputs, graphs, predict=False, batch=None):
        (
            entity_idxs, entity_num, entity_len,
            trigger_idxs, trigger_num, trigger_len, instance_mask
        ) = graphs_to_node_idxs(graphs)

        batch_size, _, bert_dim = bert_outputs.size()

        entity_idxs = bert_outputs.new_tensor(entity_idxs, dtype=torch.long)
        trigger_idxs = bert_outputs.new_tensor(trigger_idxs, dtype=torch.long)

        trigger_reprs = compute_span_reprs(bert_outputs, trigger_idxs)  # vectors for names in current sentence
        entity_reprs = compute_span_reprs(bert_outputs, entity_idxs)  # vectors for surrounding sentences

        # speaker_apperances = batch.speaker_features.unsqueeze(1).repeat(1, trigger_num * entity_num, 1)

        t_reprs, e_reprs = compute_binary_reprs(trigger_reprs, entity_reprs)
        role_reprs = torch.cat(
            [t_reprs, e_reprs, t_reprs * e_reprs, torch.abs(t_reprs - e_reprs)],
            dim=2
        )  # [batch size, num names * window size, 4 * rep dim]

        role_idn_scores = self.role_idn_ffn(role_reprs)  # [batch size, window size, 2]
        return role_idn_scores

    def forward(self, batch):
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()

        role_idn_scores = self.scores(bert_outputs, batch.graphs, predict=False, batch=batch)
        role_idn_scores = role_idn_scores.view(-1, 2)

        loss = self.cross_entropy_loss(role_idn_scores, batch.role_types)

        return loss

    def predict(self, batch):
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()

        role_idn_scores = self.scores(bert_outputs, batch.graphs, predict=False, batch=batch)  # [batch size, 3, 2]
        rile_idn_scores = torch.softmax(role_idn_scores, dim=-1)
        window_size = role_idn_scores.shape[1]

        role_idn_preds = torch.argmax(role_idn_scores, dim=2).masked_fill(batch.role_mask.eq(0),
                                                                          0).data.cpu().numpy().tolist()  # [batch size, 3]
        speaker_scores = role_idn_scores[:, :, 1].data.cpu().numpy().tolist()  # [batch size, 3]
        non_speaker_scores = role_idn_scores[:, :, 0].data.cpu().numpy().tolist()  # [batch size, 3]
        pred_speaker_ids = []
        for bid in range(batch_size):
            
            found_speaker = False

            if self.config.use_patterns:
                text = ''.join([t['text'] for t in batch.tokens[bid]]).lower()
                # self intro patterns
                for pt in SELF_INTRO_PATTERNS:
                    if pt + batch.person_names[bid].lower().replace(' ', '') in text:
                        found_speaker = True
                        pred_speaker_ids.append({'meeting-id': batch.meeting_ids[bid], 'person-name': batch.person_names[bid],
                            'pred-speaker-id': batch.current_speaker_ids[bid], 'gold-speaker-id-relative': batch.relatives[bid],
                                                 'pred-score': 1.0,
                                                 'gold-speaker-id': batch.gold_speaker_ids[bid]})
                        
                        break
                

            if not found_speaker:
                if sum(role_idn_preds[bid]) > 0:  # matches some speaker id
                    # find the best one
                    positive_ids = [(speaker_id, score) for i, speaker_id, score in
                                    zip(list(range(window_size)), batch.speaker_ids[bid], speaker_scores[bid]) if
                                    role_idn_preds[bid][i] != 0]
                    best_speaker_id, best_score = max(positive_ids, key=lambda x: x[1])
                    pred_speaker_ids.append({'meeting-id': batch.meeting_ids[bid], 'person-name': batch.person_names[bid],
                        'pred-speaker-id': best_speaker_id, 'gold-speaker-id-relative': batch.relatives[bid],
                                             'pred-score': best_score,
                                             'gold-speaker-id': batch.gold_speaker_ids[bid]})
                else:
                    pred_speaker_ids.append({'meeting-id': batch.meeting_ids[bid], 'person-name': batch.person_names[bid],
                                             'pred-speaker-id': 'N/A', 'gold-speaker-id-relative': batch.relatives[bid],
                                             'pred-score': non_speaker_scores[bid],
                                             'gold-speaker-id': batch.gold_speaker_ids[bid]})

        return pred_speaker_ids
