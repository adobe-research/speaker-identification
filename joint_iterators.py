import os
import copy
import itertools
import json, random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter, namedtuple, defaultdict
from graph import Graph
from util import *
from copy import deepcopy

os.makedirs('resource/precomputed_features/CoNLL03-English', exist_ok=True)
os.makedirs('resource/precomputed_features/CoNLL03-English-synthetic', exist_ok=True)

instance_fields = [
    'sent_id', 'tokens', 'pieces', 'piece_idxs', 'token_lens', 'attention_mask',
    'entity_label_idxs',
    'graph', 'entity_num'
]

batch_fields = [
    'sent_ids', 'tokens', 'piece_idxs', 'token_lens', 'attention_masks',
    'entity_label_idxs',
    'graphs', 'token_nums'
]

Instance = namedtuple('Instance', field_names=instance_fields)
Batch = namedtuple('Batch', field_names=batch_fields)

instance_fields_pred = [
    'sent_id', 'text', 'tokens', 'pieces', 'piece_idxs', 'token_lens', 'attention_mask',
    'entity_label_idxs',
    'graph', 'entity_num'
]

batch_fields_pred = [
    'sent_ids', 'texts', 'tokens', 'piece_idxs', 'token_lens', 'attention_masks',
    'entity_label_idxs',
    'graphs', 'token_nums'
]

Instance_pred = namedtuple('Instance', field_names=instance_fields_pred)
Batch_pred = namedtuple('Batch', field_names=batch_fields_pred)


def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    for entity in entities:
        start, end = entity['start'], entity['end']
        for i in range(start, end):
            if tokens[i]:
                id_map[entity['id']] = tokens[i]  # tokens[i] returns the id of the overlapping entity
                continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity['id']
    return entities_, id_map


def get_entity_labels(entities, token_num):
    """Convert entity mentions in a sentence to an entity label sequence with
    the length of token_num
    CHECKED
    :param entities (list): a list of entity mentions.
    :param token_num (int): the number of tokens.
    :return:a sequence of BIO format labels.
    """
    labels = ['O'] * token_num
    for entity in entities:
        start, end = entity['start'], entity['end']
        entity_type = entity['entity_type']
        if any([labels[i] != 'O' for i in range(start, end)]):
            continue
        labels[start] = 'B-{}'.format(entity_type)
        for i in range(start + 1, end):
            labels[i] = 'I-{}'.format(entity_type)
    return labels


def get_trigger_labels(events, token_num):
    """Convert event mentions in a sentence to a trigger label sequence with the
    length of token_num.
    :param events (list): a list of event mentions.
    :param token_num (int): the number of tokens.
    :return: a sequence of BIO format labels.
    """
    labels = ['O'] * token_num
    for event in events:
        trigger = event['trigger']
        start, end = trigger['start'], trigger['end']
        event_type = event['event_type']
        labels[start] = 'B-{}'.format(event_type)
        for i in range(start + 1, end):
            labels[i] = 'I-{}'.format(event_type)
    return labels


def get_relation_types(entities, relations, id_map, directional=False,
                       symmetric=None):
    """Get relation type labels among all entities in a sentence.
    :param entities (list): a list of entity mentions.
    :param relations (list): a list of relation mentions.
    :param id_map (dict): a dict of entity ID mapping.
    :param symmetric (set): a set of symmetric relation types.
    :return: a matrix of relation type labels.
    """
    entity_num = len(entities)
    labels = [['O'] * entity_num for _ in range(
        entity_num)]  # a matrix of labels: L, L[i][j] tells the relation type between entity i and entity j.
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    for relation in relations:
        entity_1 = entity_2 = -1
        for arg in relation['arguments']:
            entity_id = arg['entity_id']
            entity_id = id_map.get(entity_id, entity_id)
            if arg['role'] == 'Arg-1':
                entity_1 = entity_idxs[entity_id]
            elif arg['role'] == 'Arg-2':
                entity_2 = entity_idxs[entity_id]
        if entity_1 == -1 or entity_2 == -1:  # skip this relation
            continue
        labels[entity_1][entity_2] = relation['relation_type']
        if not directional:  # similar to adjacency matrix where we consider i-> j and j -> i, default: directional=false
            labels[entity_2][entity_1] = relation['relation_type']
        if symmetric and relation['relation_type'] in symmetric:  # always symmetrix, cannot set this directional
            labels[entity_2][entity_1] = relation['relation_type']
    return labels


def get_relation_list(entities, relations, id_map, vocab, directional=False,
                      symmetric=None):
    """Get the relation list (used for Graph objects)
    :param entities (list): a list of entity mentions.
    :param relations (list): a list of relation mentions.
    :param id_map (dict): a dict of entity ID mapping.
    :param vocab (dict): a dict of label to label index mapping.
    """
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(entities))]
    relation_list = []
    for relation in relations:
        arg_1 = arg_2 = None
        for arg in relation['arguments']:
            if arg['role'] == 'Arg-1':
                arg_1 = entity_idxs[id_map.get(
                    arg['entity_id'], arg['entity_id'])]  # if this entity is not obmitted
            elif arg['role'] == 'Arg-2':
                arg_2 = entity_idxs[id_map.get(
                    arg['entity_id'], arg['entity_id'])]
        if arg_1 is None or arg_2 is None:
            continue
        relation_type = relation['relation_type']
        # sort arg1, arg2 in the ascending order of their positions in the sentence.
        if (not directional and arg_1 > arg_2) or \
                (directional and symmetric and relation_type in symmetric and arg_1 > arg_2):
            arg_1, arg_2 = arg_2, arg_1
        if visited[arg_1][arg_2] == 0:
            relation_list.append((arg_1, arg_2, vocab[relation_type]))
            visited[arg_1][arg_2] = 1

    relation_list.sort(key=lambda x: (x[0], x[1]))
    return relation_list


def get_role_types(entities, events, id_map):
    labels = [['O'] * len(entities) for _ in
              range(len(events))]  # each event has its own argument sequence over entities, NOT tokens!
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    for event_idx, event in enumerate(events):
        for arg in event['arguments']:
            entity_id = arg['entity_id']
            entity_id = id_map.get(entity_id, entity_id)
            entity_idx = entity_idxs[entity_id]
            # if labels[event_idx][entity_idx] != 'O':
            #     print('Conflict argument role {} {} {}'.format(event['trigger']['text'], arg['text'], arg['role']))
            labels[event_idx][entity_idx] = arg['role']
    return labels


def get_role_list(entities, events, id_map, vocab):
    entity_idxs = {entity['id']: i for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(events))]
    role_list = []
    for i, event in enumerate(events):
        for arg in event['arguments']:
            entity_idx = entity_idxs[id_map.get(
                arg['entity_id'], arg['entity_id'])]
            if visited[i][entity_idx] == 0:
                role_list.append((i, entity_idx, vocab[arg['role']]))
                visited[i][entity_idx] = 1
    role_list.sort(key=lambda x: (x[0], x[1]))
    return role_list


def get_coref_types(entities):
    entity_num = len(entities)
    labels = [['O'] * entity_num for _ in range(entity_num)]
    clusters = defaultdict(list)
    for i, entity in enumerate(entities):
        entity_id = entity['entity_id']
        cluster_id = entity_id[:entity_id.rfind('-')]
        clusters[cluster_id].append(i)
    for _, entities in clusters.items():
        for i, j in itertools.combinations(entities, 2):
            labels[i][j] = 'COREF'
            labels[j][i] = 'COREF'
    return labels


def get_coref_list(entities, vocab):
    clusters = defaultdict(list)
    coref_list = []
    for i, entity in enumerate(entities):
        entity_id = entity['entity_id']
        cluster_id = entity_id[:entity_id.rfind('-')]
        clusters[cluster_id].append(i)
    for _, entities in clusters.items():
        for i, j in itertools.combinations(entities, 2):
            if i < j:
                coref_list.append((i, j, vocab['COREF']))
            else:
                coref_list.append((j, i, vocab['COREF']))
    coref_list.sort(key=lambda x: (x[0], x[1]))
    return coref_list


def merge_coref_relation_lists(coref_list, relation_list, entity_num):
    visited = [[0] * entity_num for _ in range(entity_num)]
    merge_list = []
    for i, j, l in coref_list:
        visited[i][j] = 1
        visited[j][i] = 1
        merge_list.append((i, j, l))
    for i, j, l in relation_list:
        assert visited[i][j] == 0 and visited[j][i] == 0
        merge_list.append((i, j, l))
    merge_list.sort(key=lambda x: (x[0], x[1]))


def merge_coref_relation_types(coref_types, relation_types):
    entity_num = len(coref_types)
    labels = copy.deepcopy(coref_types)
    for i in range(entity_num):
        for j in range(entity_num):
            label = relation_types[i][j]
            if label != 0:
                assert labels[i][j] == 0
                labels[i][j] = label
    return labels


speaker_instance_fields = [
    'meeting_id', 'person_names', 'gold_speaker_ids', 'speaker_ids', 'relatives',
    'current_speaker_id',
    'tokens', 'piece_idxs', 'token_lens', 'attention_mask',
    'graph', 'trigger_num', 'entity_num', 'role_types',
    'speaker_feature',
]

speaker_batch_fields = [
    'meeting_ids', 'person_names', 'gold_speaker_ids', 'speaker_ids', 'relatives',
    'current_speaker_ids',
    'tokens', 'piece_idxs', 'token_lens', 'attention_masks',
    'graphs', 'role_mask', 'role_types',
    'speaker_features',
]

speaker_Instance = namedtuple('speaker_Instance', field_names=speaker_instance_fields)
speaker_Batch = namedtuple('speaker_Batch', field_names=speaker_batch_fields)


class JointSpeakerDataset(Dataset):
    def __init__(self, config, path):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        """
        self.config = config

        self.path = path
        self.data = []
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        with open(self.path) as f:
            self.data = json.load(f)

        self.meetings = deepcopy(self.data)
        print('Loaded {} meetings from {}'.format(len(self), self.path))

    def numberize(self, tokenizer):

        data = []

        for meeting in self.data:
            meeting_id = meeting['doc_id']
            sentences = meeting['sentences']
            normalize_face = meeting['normalize_face'] if 'normalize_face' in meeting else {}
            face2id = meeting['face2id'] if 'face2id' in meeting else {}
            for face in face2id:
                face2id[face] = str(face2id[face])

            for sent_id, cur_sent in enumerate(sentences):
                if len(cur_sent['person_names']) > 0:
                    sent_speaker_id = str(face2id[normalize_face[cur_sent['face']]]) if 'face' in cur_sent else \
                    cur_sent['speakerFaceId']
                    prev_sent_id = sent_id - 1
                    next_sent_id = sent_id + 1
                    prev_sent = None
                    next_sent = None

                    while prev_sent_id >= 0:
                        prev_tmp = face2id[normalize_face[sentences[prev_sent_id]['face']]] if 'face' in sentences[
                            prev_sent_id] else sentences[prev_sent_id]['speakerFaceId']
                        if prev_tmp == sent_speaker_id:
                            prev_sent_id -= 1
                        else:
                            prev_sent = sentences[prev_sent_id]
                            break
                    while next_sent_id < len(sentences):
                        next_tmp = face2id[normalize_face[sentences[next_sent_id]['face']]] if 'face' in sentences[
                            next_sent_id] else sentences[next_sent_id]['speakerFaceId']
                        if next_tmp == sent_speaker_id:
                            next_sent_id += 1
                        else:
                            next_sent = sentences[next_sent_id]
                            break

                    all_gold_speaker_ids = []
                    all_person_names = []
                    all_speaker_ids = []
                    all_trigger_list = []
                    all_entity_list = []
                    all_role_types = []
                    all_relatives = []
                    all_tokens = []
                    for name in cur_sent['person_names']:
                        gold_speaker_id = name['speakerId']
                        person_name = name['text']
                        speaker_ids = [sent_speaker_id]
                        entity_list = []
                        role_types = []
                        relative = 'N/A'
                        tokens = []

                        if prev_sent is not None:
                            if 'face' in prev_sent:
                                speaker_ids = [face2id[normalize_face[prev_sent['face']]]] + speaker_ids
                            else:
                                speaker_ids = [prev_sent['speakerFaceId']] + speaker_ids

                            start_token = len(tokens)
                            tokens += deepcopy(prev_sent['tokens'])
                            end_token = len(tokens)

                            entity_list.append([start_token, end_token, 'previous-speaker-sentence'])

                            if 'face' in prev_sent:
                                if face2id[normalize_face[prev_sent['face']]] == gold_speaker_id:
                                    role_types.append(1)
                                    relative = 'prev'
                                else:
                                    role_types.append(0)
                            else:
                                if prev_sent['speakerFaceId'] == gold_speaker_id:
                                    role_types.append(1)
                                    relative = 'prev'
                                else:
                                    role_types.append(0)

                        if prev_sent is None and sent_id > 0 or prev_sent is not None and sent_id - 1 > prev_sent_id >= 0:
                            tokens += deepcopy(sentences[sent_id - 1]['tokens'])

                        start_token = len(tokens)
                        tmp = deepcopy(cur_sent['tokens'])
                        tokens += tmp
                        end_token = len(tokens)
                        entity_list.append([start_token, end_token, 'current-speaker-sentence'])

                        if 'face' in cur_sent:
                            if face2id[normalize_face[cur_sent['face']]] == gold_speaker_id:
                                role_types.append(1)
                                relative = 'cur'
                            else:
                                role_types.append(0)
                        else:
                            if cur_sent['speakerFaceId'] == gold_speaker_id:
                                role_types.append(1)
                                relative = 'cur'
                            else:
                                role_types.append(0)

                        trigger_list = [
                            [start_token + name['start_token'], start_token + name['end_token'], 'person-name']]

                        if next_sent is not None:
                            if 'face' in next_sent:
                                speaker_ids += [face2id[normalize_face[next_sent['face']]]]
                            else:
                                speaker_ids += [next_sent['speakerFaceId']]

                            start_token = len(tokens)
                            tokens += deepcopy(next_sent['tokens'])
                            end_token = len(tokens)

                            entity_list.append([start_token, end_token, 'next-speaker-sentence'])

                            if 'face' in next_sent:
                                if face2id[normalize_face[next_sent['face']]] == gold_speaker_id:
                                    role_types.append(1)
                                    relative = 'next'
                                else:
                                    role_types.append(0)
                            else:
                                if next_sent['speakerFaceId'] == gold_speaker_id:
                                    relative = 'next'
                                    role_types.append(1)
                                else:
                                    role_types.append(0)

                        if next_sent is None and sent_id + 1 < len(
                                sentences) or next_sent is not None and sent_id + 1 < next_sent_id:
                            tokens += deepcopy(sentences[sent_id + 1]['tokens'])

                        all_gold_speaker_ids.append(gold_speaker_id)
                        all_person_names.append(person_name)
                        all_entity_list.append(entity_list)
                        all_tokens.append(tokens)
                        all_speaker_ids.append(speaker_ids)
                        all_role_types.append(role_types)
                        all_relatives.append(relative)
                        all_trigger_list.extend(trigger_list)

                    speaker_feature = [0] * 4

                    graph = Graph(
                        entities=all_entity_list[0],
                        triggers=all_trigger_list,
                        relations=[],
                        roles=[],
                        mentions=[],
                        vocabs={},
                    )

                    group_pieces = [[p for p in tokenizer.tokenize(w['text']) if p != '▁'] for w in all_tokens[0]]
                    for ps in group_pieces:
                        if len(ps) == 0:
                            ps += ['-']
                    pieces = [p for ps in group_pieces for p in ps]
                    token_lens = [len(x) for x in group_pieces]

                    # Pad word pieces with special tokens
                    piece_idxs = tokenizer.encode(pieces,
                                                  add_special_tokens=True,
                                                  max_length=500,
                                                  truncation=True)
                    attn_mask = [1] * len(piece_idxs)

                    instance = speaker_Instance(
                        meeting_id=meeting_id,
                        person_names=all_person_names,
                        gold_speaker_ids=all_gold_speaker_ids,
                        relatives=all_relatives,
                        speaker_ids=all_speaker_ids,
                        current_speaker_id=sent_speaker_id,
                        tokens=all_tokens[0],
                        piece_idxs=piece_idxs,
                        token_lens=token_lens,
                        attention_mask=attn_mask,
                        graph=graph,
                        trigger_num=len(all_trigger_list),
                        entity_num=len(all_entity_list[0]),
                        role_types=all_role_types,
                        speaker_feature=speaker_feature
                    )
                    data.append(instance)

        if self.config.augment_lowercase and 'train' in self.path:
            for meeting in self.data:
                meeting_id = meeting['doc_id']
                sentences = meeting['sentences']
                normalize_face = meeting['normalize_face'] if 'normalize_face' in meeting else {}
                face2id = meeting['face2id'] if 'face2id' in meeting else {}
                for face in face2id:
                    face2id[face] = str(face2id[face])

                for sent_id, cur_sent in enumerate(sentences):
                    if len(cur_sent['person_names']) > 0:
                        sent_speaker_id = str(face2id[normalize_face[cur_sent['face']]]) if 'face' in cur_sent else \
                            cur_sent['speakerFaceId']
                        prev_sent_id = sent_id - 1
                        next_sent_id = sent_id + 1
                        prev_sent = None
                        next_sent = None

                        while prev_sent_id >= 0:
                            prev_tmp = face2id[normalize_face[sentences[prev_sent_id]['face']]] if 'face' in sentences[
                                prev_sent_id] else sentences[prev_sent_id]['speakerFaceId']
                            if prev_tmp == sent_speaker_id:
                                prev_sent_id -= 1
                            else:
                                prev_sent = sentences[prev_sent_id]
                                break
                        while next_sent_id < len(sentences):
                            next_tmp = face2id[normalize_face[sentences[next_sent_id]['face']]] if 'face' in sentences[
                                next_sent_id] else sentences[next_sent_id]['speakerFaceId']
                            if next_tmp == sent_speaker_id:
                                next_sent_id += 1
                            else:
                                next_sent = sentences[next_sent_id]
                                break

                        all_gold_speaker_ids = []
                        all_person_names = []
                        all_speaker_ids = []
                        all_trigger_list = []
                        all_entity_list = []
                        all_role_types = []
                        all_relatives = []
                        all_tokens = []
                        for name in cur_sent['person_names']:
                            gold_speaker_id = name['speakerId']
                            person_name = name['text']
                            speaker_ids = [sent_speaker_id]
                            entity_list = []
                            role_types = []
                            relative = 'N/A'
                            tokens = []

                            if prev_sent is not None:
                                if 'face' in prev_sent:
                                    speaker_ids = [face2id[normalize_face[prev_sent['face']]]] + speaker_ids
                                else:
                                    speaker_ids = [prev_sent['speakerFaceId']] + speaker_ids

                                start_token = len(tokens)
                                tokens += deepcopy(prev_sent['tokens'])
                                end_token = len(tokens)

                                entity_list.append([start_token, end_token, 'previous-speaker-sentence'])

                                if 'face' in prev_sent:
                                    if face2id[normalize_face[prev_sent['face']]] == gold_speaker_id:
                                        role_types.append(1)
                                        relative = 'prev'
                                    else:
                                        role_types.append(0)
                                else:
                                    if prev_sent['speakerFaceId'] == gold_speaker_id:
                                        role_types.append(1)
                                        relative = 'prev'
                                    else:
                                        role_types.append(0)

                            if prev_sent is None and sent_id > 0 or prev_sent is not None and sent_id - 1 > prev_sent_id >= 0:
                                tokens += deepcopy(sentences[sent_id - 1]['tokens'])

                            start_token = len(tokens)
                            tmp = deepcopy(cur_sent['tokens'])
                            tokens += tmp
                            end_token = len(tokens)
                            entity_list.append([start_token, end_token, 'current-speaker-sentence'])

                            if 'face' in cur_sent:
                                if face2id[normalize_face[cur_sent['face']]] == gold_speaker_id:
                                    role_types.append(1)
                                    relative = 'cur'
                                else:
                                    role_types.append(0)
                            else:
                                if cur_sent['speakerFaceId'] == gold_speaker_id:
                                    role_types.append(1)
                                    relative = 'cur'
                                else:
                                    role_types.append(0)

                            trigger_list = [
                                [start_token + name['start_token'], start_token + name['end_token'], 'person-name']]

                            if next_sent is not None:
                                if 'face' in next_sent:
                                    speaker_ids += [face2id[normalize_face[next_sent['face']]]]
                                else:
                                    speaker_ids += [next_sent['speakerFaceId']]

                                start_token = len(tokens)
                                tokens += deepcopy(next_sent['tokens'])
                                end_token = len(tokens)

                                entity_list.append([start_token, end_token, 'next-speaker-sentence'])

                                if 'face' in next_sent:
                                    if face2id[normalize_face[next_sent['face']]] == gold_speaker_id:
                                        role_types.append(1)
                                        relative = 'next'
                                    else:
                                        role_types.append(0)
                                else:
                                    if next_sent['speakerFaceId'] == gold_speaker_id:
                                        relative = 'next'
                                        role_types.append(1)
                                    else:
                                        role_types.append(0)

                            if next_sent is None and sent_id + 1 < len(
                                    sentences) or next_sent is not None and sent_id + 1 < next_sent_id:
                                tokens += deepcopy(sentences[sent_id + 1]['tokens'])

                            all_gold_speaker_ids.append(gold_speaker_id)
                            all_person_names.append(person_name)
                            all_entity_list.append(entity_list)
                            all_tokens.append(tokens)
                            all_speaker_ids.append(speaker_ids)
                            all_role_types.append(role_types)
                            all_relatives.append(relative)
                            all_trigger_list.extend(trigger_list)

                        speaker_feature = [0] * 4

                        graph = Graph(
                            entities=all_entity_list[0],
                            triggers=all_trigger_list,
                            relations=[],
                            roles=[],
                            mentions=[],
                            vocabs={},
                        )

                        group_pieces = [[p for p in tokenizer.tokenize(w['text']) if p != '▁'] for w in all_tokens[0]]
                        for ps in group_pieces:
                            if len(ps) == 0:
                                ps += ['-']
                        pieces = [p for ps in group_pieces for p in ps]
                        token_lens = [len(x) for x in group_pieces]

                        # Pad word pieces with special tokens
                        piece_idxs = tokenizer.encode(pieces,
                                                      add_special_tokens=True,
                                                      max_length=500,
                                                      truncation=True)
                        attn_mask = [1] * len(piece_idxs)

                        instance = speaker_Instance(
                            meeting_id=meeting_id,
                            person_names=all_person_names,
                            gold_speaker_ids=all_gold_speaker_ids,
                            relatives=all_relatives,
                            speaker_ids=all_speaker_ids,
                            current_speaker_id=sent_speaker_id,
                            tokens=all_tokens[0],
                            piece_idxs=piece_idxs,
                            token_lens=token_lens,
                            attention_mask=attn_mask,
                            graph=graph,
                            trigger_num=len(all_trigger_list),
                            entity_num=len(all_entity_list[0]),
                            role_types=all_role_types,
                            speaker_feature=speaker_feature
                        )
                        data.append(instance)

        print('Numberized {} examples'.format(len(data)))
        self.data = data

    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_graphs = []
        batch_token_lens = []
        batch_attention_masks = []

        max_num_pieces = max([len(inst.piece_idxs) for inst in batch])
        max_trigger_num = max([inst.trigger_num for inst in batch])
        max_entity_num = max([inst.entity_num for inst in batch])
        batch_role_mask = []
        batch_role_types = []

        for inst in batch:
            batch_piece_idxs.append(inst.piece_idxs + [0] * (max_num_pieces - len(inst.piece_idxs)))
            batch_attention_masks.append(inst.attention_mask + [0] * (max_num_pieces - len(inst.piece_idxs)))
            batch_token_lens.append(inst.token_lens)
            batch_graphs.append(inst.graph)

            tmp = []
            for _ in range(inst.trigger_num):
                tmp.extend([1] * inst.entity_num + [0] * (max_entity_num - inst.entity_num))

            tmp.extend([0] * max_entity_num * (max_trigger_num - inst.trigger_num))
            batch_role_mask.append(tmp)

            for i in range(inst.trigger_num):
                batch_role_types.extend(inst.role_types[i] + [-100] * (max_entity_num - inst.entity_num))

            batch_role_types.extend([-100] * max_entity_num * (max_trigger_num - inst.trigger_num))

        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(
            batch_attention_masks)

        batch_role_mask = torch.cuda.FloatTensor(batch_role_mask)
        batch_role_types = torch.cuda.LongTensor(batch_role_types)

        speaker_features = torch.cuda.FloatTensor([inst.speaker_feature for inst in batch])

        return speaker_Batch(
            meeting_ids=[inst.meeting_id for inst in batch],
            person_names=[inst.person_names for inst in batch],
            gold_speaker_ids=[inst.gold_speaker_ids for inst in batch],
            speaker_ids=[inst.speaker_ids for inst in batch],
            current_speaker_ids=[inst.current_speaker_id for inst in batch],
            tokens=[inst.tokens for inst in batch],
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            graphs=[inst.graph for inst in batch],
            role_mask=batch_role_mask,
            relatives=[inst.relatives for inst in batch],
            role_types=batch_role_types,
            speaker_features=speaker_features,
        )
