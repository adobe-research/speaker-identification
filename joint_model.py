import numpy as np
import torch, json
import torch.nn as nn
from graph import Graph
from gcn_utils import InstanceGCN
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


class JointSpeakerIdentifier(nn.Module):
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

        self.instance_gcn = InstanceGCN(
            trigger_dim=self.bert_dim,
            entity_dim=self.bert_dim,
            num_layers=2
        )

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
            entity_idxs, num_speakers, entity_len,
            trigger_idxs, num_names, trigger_len, instance_mask
        ) = graphs_to_node_idxs(graphs)

        batch_size, _, bert_dim = bert_outputs.size()

        entity_idxs = bert_outputs.new_tensor(entity_idxs, dtype=torch.long)
        trigger_idxs = bert_outputs.new_tensor(trigger_idxs, dtype=torch.long)

        trigger_reprs = compute_span_reprs(bert_outputs, trigger_idxs)  # vectors for names in current sentence
        entity_reprs = compute_span_reprs(bert_outputs, entity_idxs)  # vectors for surrounding sentences
        # trigger_reprs.shape = [batch size, num names, bert dim]
        # entity_reprs.shape = [batch size, num speakers, bert dim]

        # speaker_apperances = batch.speaker_features.unsqueeze(1).repeat(1, trigger_num * entity_num, 1)

        node_reprs = torch.cat([trigger_reprs, entity_reprs], dim=1) # [batch size, num names + num speakers, bert dim]
        num_nodes = node_reprs.shape[1]

        similarity_scores = node_reprs.bmm(node_reprs.transpose(1, 2)) # [batch size, num nodes, num nodes]
        similarity_graph = torch.softmax(similarity_scores, dim=-1).view(batch_size, num_nodes, num_nodes)

        trigger_reprs, entity_reprs = self.instance_gcn(
            similarity_graph,
            trigger_reprs,
            entity_reprs,
            num_names,
            num_speakers
        )

        t_reprs, e_reprs = compute_binary_reprs(trigger_reprs, entity_reprs)

        role_reprs = torch.cat(
            [t_reprs, e_reprs, t_reprs * e_reprs, torch.abs(t_reprs - e_reprs)],
            dim=2
        )  # [batch size, num names * num speakers, 4 * rep dim]

        role_idn_scores = self.role_idn_ffn(role_reprs)  # [batch size, num names * window size, 2]
        return role_idn_scores, num_names, num_speakers

    def forward(self, batch):
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()

        role_idn_scores, trigger_num, entity_num = self.scores(bert_outputs, batch.graphs, predict=False, batch=batch)
        role_idn_scores = role_idn_scores.view(-1, 2)

        loss = self.cross_entropy_loss(role_idn_scores, batch.role_types)

        return loss

    def predict(self, batch):
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()

        role_idn_scores, num_names, num_speakers = self.scores(bert_outputs, batch.graphs, predict=False,
                                                               batch=batch)  # [batch size, num names * num speakers, 2]
        role_idn_scores = torch.softmax(role_idn_scores, dim=-1)

        role_idn_preds = torch.argmax(role_idn_scores, dim=2).masked_fill(batch.role_mask.eq(0),
                                                                          0).data.cpu().numpy().tolist()  # [batch size, num names * num speakers]
        speaker_scores = role_idn_scores[:, :, 1].data.cpu().numpy().tolist()  # [batch size, num names * num speakers]
        non_speaker_scores = role_idn_scores[:, :,
                             0].data.cpu().numpy().tolist()  # [batch size, num names * num speakers]
        pred_speaker_ids = []
        for bid in range(batch_size):
            text = ''.join([t['text'] for t in batch.tokens[bid]]).lower()

            for i in range(len(batch.person_names[bid])):
                found_speaker = False

                if self.config.use_patterns:
                    cur_name = batch.person_names[bid][i].lower().replace(' ', '')

                    for pt in SELF_INTRO_PATTERNS:
                        if pt + cur_name in text:
                            found_speaker = True
                            pred_speaker_ids.append({'meeting-id': batch.meeting_ids[bid], 'person-name': batch.person_names[bid],
                                'pred-speaker-id': batch.current_speaker_ids[bid], 'gold-speaker-id-relative': batch.relatives[bid],
                                                     'pred-score': 1.0,
                                                     'gold-speaker-id': batch.gold_speaker_ids[bid]})
                            
                            break


                if not found_speaker:
                    if sum(role_idn_preds[bid][i * num_speakers: (i+1) * num_speakers]) > 0:  # matches some speaker id
                        # find the best one
                        positive_ids = [(speaker_id, score) for ix, speaker_id, score in
                                        zip(list(range(num_speakers)), batch.speaker_ids[bid][i], speaker_scores[bid][i * num_speakers: (i+1) * num_speakers]) if
                                        role_idn_preds[bid][i * num_speakers + ix] != 0]
                        best_speaker_id, best_score = max(positive_ids, key=lambda x: x[1])
                        pred_speaker_ids.append({'meeting-id': batch.meeting_ids[bid], 'person-name': batch.person_names[bid][i],
                                                 'pred-speaker-id': best_speaker_id,
                                                 'gold-speaker-id-relative': batch.relatives[bid][i],
                                                 'pred-score': best_score,
                                                 'gold-speaker-id': batch.gold_speaker_ids[bid][i]})
                    else:
                        pred_speaker_ids.append({'meeting-id': batch.meeting_ids[bid], 'person-name': batch.person_names[bid][i],
                                                 'pred-speaker-id': 'N/A', 'gold-speaker-id-relative': batch.relatives[bid][i],
                                                 'pred-score': non_speaker_scores[bid][i * num_speakers],
                                                 'gold-speaker-id': batch.gold_speaker_ids[bid][i]})

        return pred_speaker_ids
