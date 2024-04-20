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


class NERDatasetPred(Dataset):
    def __init__(self, config, path, max_length=500, gpu=False):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        """
        self.config = config

        self.path = path
        self.data = []
        self.gpu = gpu
        self.max_length = max_length
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def entity_type_set(self):
        type_set = set()
        for inst in self.data:
            for entity in inst['entity_mentions']:
                type_set.add(entity['entity_type'])
        return type_set

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def relation_type_set(self):
        type_set = set()
        for inst in self.data:
            for relation in inst['relation_mentions']:
                type_set.add(relation['relation_type'])
        return type_set

    @property
    def role_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['role'])
        return type_set

    def load_data(self):
        """Load data from file."""
        with open(self.path) as r:
            for line in r:
                line = line.strip()

                if line:
                    self.data.append({'text': line})

        print('Loaded {} sentences from {}'.format(len(self), self.path))

    def numberize(self, tokenizer, vocabs):
        """Numberize word pieces, labels, etcs.
        :param tokenizer: Bert tokenizer.
        :param vocabs (dict): a dict of vocabularies.
        """
        entity_type_stoi = vocabs['entity_type']
        entity_label_stoi = vocabs['entity_label']

        data = []
        lengthy = 0
        w2subwords = {}

        progress = tqdm.tqdm(total=len(self.data), ncols=75,
                             desc='Numberize')
        for inst in self.data:
            progress.update(1)
            raw_tokens = self.config.trankit_pipeline.tokenize(inst['text'], is_sent=True)['tokens']
            tokens = [t['text'] for t in raw_tokens]

            group_pieces = []
            for w in tokens:
                if w not in w2subwords:
                    subwords = tokenizer.tokenize(w)
                    w2subwords[w] = subwords
                else:
                    subwords = w2subwords[w]
                group_pieces.append(subwords)

            for ps in group_pieces:
                if len(ps) == 0:
                    ps += ['-']
            pieces = [p for ps in group_pieces for p in ps]
            token_lens = [len(x) for x in group_pieces]

            sent_id = len(data)
            token_num = len(tokens)

            # Pad word pieces with special tokens
            piece_idxs = tokenizer.encode(pieces,
                                          add_special_tokens=True,
                                          max_length=self.max_length,
                                          truncation=True)
            attn_mask = [1] * len(piece_idxs)
            if len(piece_idxs) > self.config.max_subwords:
                lengthy += 1
                continue

            # Entity

            entity_label_idxs = [0] * token_num
            entity_type_idxs = []
            entity_list = []
            # entity_num = len(entity_list)
            mention_type_idxs = []
            mention_list = []

            # Trigger

            trigger_label_idxs = [0] * token_num
            event_type_idxs = []
            trigger_list = []

            # Relation
            relation_type_idxs = []
            relation_list = []

            # Argument role
            role_type_idxs = []
            role_list = []

            # Graph
            graph = Graph(
                entities=entity_list,
                triggers=trigger_list,
                relations=relation_list,
                roles=role_list,
                mentions=mention_list,
                vocabs=vocabs,
            )

            instance = Instance_pred(
                sent_id=sent_id,
                text=inst['text'],
                tokens=raw_tokens,
                pieces=pieces,
                piece_idxs=piece_idxs,
                token_lens=token_lens,
                attention_mask=attn_mask,
                entity_label_idxs=entity_label_idxs,
                graph=graph,
                entity_num=0
            )
            data.append(instance)

        if lengthy > 0:
            print('{}: skipped {} examples with more than {} subwords'.format(self.path, lengthy,
                                                                              self.config.max_subwords))

        progress.close()
        self.data = data

    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_tokens = []
        batch_entity_labels = []
        batch_graphs = []
        batch_token_lens = []
        batch_attention_masks = []

        sent_ids = [inst.sent_id for inst in batch]
        token_nums = [len(inst.tokens) for inst in batch]
        max_token_num = max(token_nums)
        max_num_pieces = max([len(inst.piece_idxs) for inst in batch])
        max_entity_num = max([inst.entity_num for inst in batch] + [1])

        for inst in batch:
            token_num = len(inst.tokens)
            batch_piece_idxs.append(inst.piece_idxs + [0] * (max_num_pieces - len(inst.piece_idxs)))
            batch_attention_masks.append(inst.attention_mask + [0] * (max_num_pieces - len(inst.piece_idxs)))
            batch_token_lens.append(inst.token_lens)
            batch_graphs.append(inst.graph)
            batch_tokens.append(inst.tokens)
            # for identification
            batch_entity_labels.append(inst.entity_label_idxs +
                                       [0] * (max_token_num - token_num))

        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(
            batch_attention_masks)

        batch_entity_labels = torch.cuda.LongTensor(batch_entity_labels)

        token_nums = torch.cuda.LongTensor(token_nums)

        return Batch_pred(
            sent_ids=sent_ids,
            texts=[inst.text for inst in batch],
            tokens=[inst.tokens for inst in batch],
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            entity_label_idxs=batch_entity_labels,
            graphs=batch_graphs,
            token_nums=token_nums
        )


class NERDataset(Dataset):
    def __init__(self, config, path, max_length=500, gpu=False):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        """
        self.config = config

        self.path = path
        self.data = []
        self.gpu = gpu
        self.max_length = max_length

        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def entity_type_set(self):
        type_set = set()
        for inst in self.data:
            for entity in inst['entity_mentions']:
                type_set.add(entity['entity_type'])
        return type_set

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def role_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['role'])
        return type_set

    def load_data(self):
        self.data = get_examples_from_bio_fpath(self.path)
        print('Loaded {} sentences from {}'.format(len(self), self.path))

    def numberize(self, tokenizer, vocabs):
        """Numberize word pieces, labels, etcs.
        :param tokenizer: Bert tokenizer.
        :param vocabs (dict): a dict of vocabularies.
        """
        entity_type_stoi = vocabs['entity_type']
        entity_label_stoi = vocabs['entity_label']

        data = []
        lengthy = 0
        for inst in self.data:
            tokens = inst['tokens']

            group_pieces = [[p for p in tokenizer.tokenize(w) if p != '▁'] for w in tokens]
            for ps in group_pieces:
                if len(ps) == 0:
                    ps += ['-']
            pieces = [p for ps in group_pieces for p in ps]
            token_lens = [len(x) for x in group_pieces]

            sent_id = inst['sent_id']
            entities = inst['entity_mentions']
            entities.sort(key=lambda x: x['start'])

            token_num = len(tokens)

            # Pad word pieces with special tokens
            piece_idxs = tokenizer.encode(pieces,
                                          add_special_tokens=True,
                                          max_length=self.max_length,
                                          truncation=True)
            attn_mask = [1] * len(piece_idxs)
            if len(piece_idxs) > self.config.max_subwords:
                lengthy += 1
                continue

            # Entity

            entity_labels = get_entity_labels(entities, token_num)
            entity_label_idxs = [entity_label_stoi[l] for l in entity_labels]
            entity_types = [e['entity_type'] for e in entities]
            entity_type_idxs = [entity_type_stoi[l] for l in entity_types]
            entity_list = [(e['start'], e['end'], entity_type_stoi[e['entity_type']])
                           for e in entities]
            # entity_num = len(entity_list)

            # Graph
            graph = Graph(
                entities=entity_list,
                triggers=[],
                relations=[],
                roles=[],
                mentions=[],
                vocabs=vocabs,
            )

            instance = Instance(
                sent_id=sent_id,
                tokens=tokens,
                pieces=pieces,
                piece_idxs=piece_idxs,
                token_lens=token_lens,
                attention_mask=attn_mask,
                entity_label_idxs=entity_label_idxs,
                graph=graph,
                entity_num=len(entities)
            )
            data.append(instance)

        if lengthy > 0:
            print('{}: skipped {} examples with more than {} subwords'.format(self.path, lengthy,
                                                                              self.config.max_subwords))

        self.data = data

    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_tokens = []
        batch_entity_labels = []
        batch_graphs = []
        batch_token_lens = []
        batch_attention_masks = []

        sent_ids = [inst.sent_id for inst in batch]
        token_nums = [len(inst.tokens) for inst in batch]
        max_token_num = max(token_nums)
        max_num_pieces = max([len(inst.piece_idxs) for inst in batch])

        for inst in batch:
            token_num = len(inst.tokens)
            batch_piece_idxs.append(inst.piece_idxs + [0] * (max_num_pieces - len(inst.piece_idxs)))
            batch_attention_masks.append(inst.attention_mask + [0] * (max_num_pieces - len(inst.piece_idxs)))
            batch_token_lens.append(inst.token_lens)
            batch_graphs.append(inst.graph)
            batch_tokens.append(inst.tokens)
            # for identification
            batch_entity_labels.append(inst.entity_label_idxs +
                                       [0] * (max_token_num - token_num))

        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(
            batch_attention_masks)

        batch_entity_labels = torch.cuda.LongTensor(batch_entity_labels)

        token_nums = torch.cuda.LongTensor(token_nums)

        return Batch(
            sent_ids=sent_ids,
            tokens=[inst.tokens for inst in batch],
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            entity_label_idxs=batch_entity_labels,
            graphs=batch_graphs,
            token_nums=token_nums
        )


speaker_instance_fields = [
    'meeting_id', 'person_name', 'gold_speaker_id', 'speaker_ids', 'relative',
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


class IndividualSpeakerDataset(Dataset):
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
        lengthy = 0
        for meeting in self.data:
            meeting_id = meeting['doc_id']
            sentences = meeting['sentences']
            normalize_face = meeting['normalize_face'] if 'normalize_face' in meeting else {}
            face2id = meeting['face2id'] if 'face2id' in meeting else {}
            for face in face2id:
                face2id[face] = str(face2id[face])

            for sent_id, cur_sent in enumerate(sentences):
                if len(cur_sent['person_names']) > 0:
                    sent_speaker_id = str(face2id[normalize_face[cur_sent['face']]]) if 'face' in cur_sent else cur_sent['speakerFaceId']
                    prev_sent_id = sent_id - 1
                    next_sent_id = sent_id + 1
                    prev_sent = None
                    next_sent = None

                    while prev_sent_id >= 0:
                        prev_tmp = face2id[normalize_face[sentences[prev_sent_id]['face']]] if 'face' in sentences[prev_sent_id] else sentences[prev_sent_id]['speakerFaceId']
                        if prev_tmp == sent_speaker_id:
                            prev_sent_id -= 1
                        else:
                            prev_sent = sentences[prev_sent_id]
                            break
                    while next_sent_id < len(sentences):
                        next_tmp = face2id[normalize_face[sentences[next_sent_id]['face']]] if 'face' in sentences[next_sent_id] else sentences[next_sent_id]['speakerFaceId']
                        if next_tmp == sent_speaker_id:
                            next_sent_id += 1
                        else:
                            next_sent = sentences[next_sent_id]
                            break

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
                        tmp = deepcopy(cur_sent['tokens'])[max(0, name['start_token'] - self.config.context_size):]

                        offset = len(cur_sent['tokens']) - len(tmp)
                        name['start_token'] -= offset
                        name['end_token'] -= offset

                        tmp_len = len(tmp)
                        tmp = tmp[:min(tmp_len, name['end_token'] + self.config.context_size)]

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

                        if next_sent is None and sent_id + 1 < len(sentences) or next_sent is not None and sent_id + 1 < next_sent_id:
                            tokens += deepcopy(sentences[sent_id + 1]['tokens'])
                        
                        speaker_feature  = [0] * 4

                        graph = Graph(
                            entities=entity_list,
                            triggers=trigger_list,
                            relations=[],
                            roles=[],
                            mentions=[],
                            vocabs={},
                        )

                        group_pieces = [[p for p in tokenizer.tokenize(w['text']) if p != '▁'] for w in tokens]
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
                            person_name=person_name,
                            gold_speaker_id=gold_speaker_id,
                            relative=relative,
                            speaker_ids=speaker_ids,
                            current_speaker_id=sent_speaker_id,
                            tokens=tokens,
                            piece_idxs=piece_idxs,
                            token_lens=token_lens,
                            attention_mask=attn_mask,
                            graph=graph,
                            trigger_num=1,
                            entity_num=len(entity_list),
                            role_types=role_types,
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
                        sent_speaker_id = str(face2id[normalize_face[cur_sent['face']]]) if 'face' in cur_sent else cur_sent['speakerFaceId']
                        prev_sent_id = sent_id - 1
                        next_sent_id = sent_id + 1
                        prev_sent = None
                        next_sent = None

                        while prev_sent_id >= 0:
                            prev_tmp = face2id[normalize_face[sentences[prev_sent_id]['face']]] if 'face' in sentences[prev_sent_id] else sentences[prev_sent_id]['speakerFaceId']
                            if prev_tmp == sent_speaker_id:
                                prev_sent_id -= 1
                            else:
                                prev_sent = sentences[prev_sent_id]
                                break
                        while next_sent_id < len(sentences):
                            next_tmp = face2id[normalize_face[sentences[next_sent_id]['face']]] if 'face' in sentences[next_sent_id] else sentences[next_sent_id]['speakerFaceId']
                            if next_tmp == sent_speaker_id:
                                next_sent_id += 1
                            else:
                                next_sent = sentences[next_sent_id]
                                break

                        for name in cur_sent['person_names']:
                            gold_speaker_id = name['speakerId']
                            person_name = name['text']
                            speaker_ids = [sent_speaker_id]
                            entity_list = []
                            role_types = []
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
                                    else:
                                        role_types.append(0)
                                else:
                                    if prev_sent['speakerFaceId'] == gold_speaker_id:
                                        role_types.append(1)
                                    else:
                                        role_types.append(0)


                            if prev_sent is None and sent_id > 0 or prev_sent is not None and sent_id - 1 > prev_sent_id >= 0:
                                tokens += deepcopy(sentences[sent_id - 1]['tokens'])

                            start_token = len(tokens)
                            tmp = deepcopy(cur_sent['tokens'])[max(0, name['start_token'] - self.config.context_size):]

                            offset = len(cur_sent['tokens']) - len(tmp)
                            name['start_token'] -= offset
                            name['end_token'] -= offset

                            tmp_len = len(tmp)
                            tmp = tmp[:min(tmp_len, name['end_token'] + self.config.context_size)]

                            tokens += tmp

                            end_token = len(tokens)
                            entity_list.append([start_token, end_token, 'current-speaker-sentence'])

                            if 'face' in cur_sent:
                                if face2id[normalize_face[cur_sent['face']]] == gold_speaker_id:
                                    role_types.append(1)
                                else:
                                    role_types.append(0)
                            else:
                                if cur_sent['speakerFaceId'] == gold_speaker_id:
                                    role_types.append(1)
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
                                    else:
                                        role_types.append(0)
                                else:
                                    if next_sent['speakerFaceId'] == gold_speaker_id:
                                        role_types.append(1)
                                    else:
                                        role_types.append(0)

                            if next_sent is None and sent_id + 1 < len(sentences) or next_sent is not None and sent_id + 1 < next_sent_id:
                                tokens += deepcopy(sentences[sent_id + 1]['tokens'])
                            
                            speaker_feature  = [0] * 4

                            graph = Graph(
                                entities=entity_list,
                                triggers=trigger_list,
                                relations=[],
                                roles=[],
                                mentions=[],
                                vocabs={},
                            )

                            group_pieces = [[p for p in tokenizer.tokenize(w['text'].lower()) if p != '▁'] for w in tokens]
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
                                person_name=person_name,
                                gold_speaker_id=gold_speaker_id,
                                speaker_ids=speaker_ids,
                                current_speaker_id=sent_speaker_id,
                                tokens=tokens,
                                piece_idxs=piece_idxs,
                                token_lens=token_lens,
                                attention_mask=attn_mask,
                                graph=graph,
                                trigger_num=1,
                                entity_num=len(entity_list),
                                role_types=role_types,
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
        max_entity_num = max([inst.entity_num for inst in batch])
        batch_role_mask = []
        batch_role_types = []

        for inst in batch:
            batch_piece_idxs.append(inst.piece_idxs + [0] * (max_num_pieces - len(inst.piece_idxs)))
            batch_attention_masks.append(inst.attention_mask + [0] * (max_num_pieces - len(inst.piece_idxs)))
            batch_token_lens.append(inst.token_lens)
            batch_graphs.append(inst.graph)

            batch_role_mask.append([1] * inst.entity_num + [0] * (max_entity_num - inst.entity_num))

            batch_role_types.extend(inst.role_types + [-100] * (max_entity_num - inst.entity_num))

        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(
            batch_attention_masks)

        batch_role_mask = torch.cuda.FloatTensor(batch_role_mask)
        batch_role_types = torch.cuda.LongTensor(batch_role_types)

        speaker_features = torch.cuda.FloatTensor([inst.speaker_feature for inst in batch])

        return speaker_Batch(
            meeting_ids=[inst.meeting_id for inst in batch],
            person_names=[inst.person_name for inst in batch],
            gold_speaker_ids=[inst.gold_speaker_id for inst in batch],
            speaker_ids=[inst.speaker_ids for inst in batch],
            current_speaker_ids=[inst.current_speaker_id for inst in batch],
            tokens=[inst.tokens for inst in batch],
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            graphs=[inst.graph for inst in batch],
            role_mask=batch_role_mask,
            relatives=[inst.relative for inst in batch],
            role_types=batch_role_types,
            speaker_features=speaker_features,
        )
