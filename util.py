import os
import random, tqdm, csv
import json
import glob
from nltk import word_tokenize, sent_tokenize
from copy import deepcopy
from thefuzz import process
from xlsxwriter.workbook import Workbook

NAME_MATCHING_THRESHOLD = 80 # min: 0, max: 100

SELF_INTRO_PATTERNS = {'iam', "i'm", 'mynameis', "myname's"}

def ensure_dir(dir_fpath):
    os.makedirs(dir_fpath, exist_ok=True)

def tag_paths_to_spans(paths, token_nums, vocab):
    """Convert predicted tag paths to a list of spans (entity mentions or event
    triggers).
    :param paths: predicted tag paths.
    :return (list): a list (batch) of lists (sequence) of spans.
    """
    batch_mentions = []
    itos = {i: s for s, i in vocab.items()}
    for i, path in enumerate(paths):
        mentions = []
        cur_mention = None
        path = path.tolist()[:token_nums[i].item()]
        for j, tag in enumerate(path):
            tag = itos[tag]
            if tag == 'O':
                prefix = tag = 'O'
            else:
                prefix, tag = tag.split('-', 1)

            if prefix == 'B':
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = [j, j + 1, tag]
            elif prefix == 'I':
                if cur_mention is None:
                    # treat it as B-*
                    cur_mention = [j, j + 1, tag]
                elif cur_mention[-1] == tag:
                    cur_mention[1] = j + 1
                else:
                    # treat it as B-*
                    mentions.append(cur_mention)
                    cur_mention = [j, j + 1, tag]
            else:
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = None
        if cur_mention:
            mentions.append(cur_mention)
        mentions.sort(key=lambda x: (x[0], x[1]))
        batch_mentions.append(mentions)
    return batch_mentions


def tag_path_to_spans(path):
    mentions = []
    cur_mention = None

    for j, tag in enumerate(path):
        if tag == 'O':
            prefix = tag = 'O'
        else:
            prefix, tag = tag.split('-', 1)

        if prefix == 'B':
            if cur_mention:
                mentions.append(cur_mention)
            cur_mention = [j, j + 1, tag]
        elif prefix == 'I':
            if cur_mention is None:
                # treat it as B-*
                cur_mention = [j, j + 1, tag]
            elif cur_mention[-1] == tag:
                cur_mention[1] = j + 1
            else:
                # treat it as B-*
                mentions.append(cur_mention)
                cur_mention = [j, j + 1, tag]
        else:
            if cur_mention:
                mentions.append(cur_mention)
            cur_mention = None
    if cur_mention:
        mentions.append(cur_mention)
    mentions.sort(key=lambda x: (x[0], x[1]))
    return mentions


def convert_to_bio2(ori_tags):
    bio2_tags = []
    for i, tag in enumerate(ori_tags):
        if tag == 'O':
            bio2_tags.append(tag)
        elif tag[0] == 'I':
            if i == 0 or ori_tags[i - 1] == 'O' or ori_tags[i - 1][1:] != tag[1:]:
                bio2_tags.append('B' + tag[1:])
            else:
                bio2_tags.append(tag)
        else:
            bio2_tags.append(tag)
    return bio2_tags


def get_example_from_lines(sent_lines):
    tokens = []
    ner_tags = []
    for line in sent_lines:
        array = line.split()
        assert len(array) >= 4
        tokens.append(array[0])
        ner_tags.append(array[3])
    ner_tags = convert_to_bio2(ner_tags)
    entities = tag_path_to_spans(ner_tags)
    inst = {
        'tokens': tokens,
        'entity_mentions': [{'start': x[0], 'end': x[1], 'entity_type': x[2]} for x in entities]
    }
    return inst


def get_examples_from_bio_fpath(bio_fpath):
    sent_lines = []
    bio2_examples = []
    nlines = 0
    with open(bio_fpath) as infile:
        for line in infile:
            nlines += 1
            line = line.strip()
            if '-DOCSTART-' in line or '-docstart-' in line:
                continue
            if len(line) > 0:
                array = line.split()
                if len(array) < 4:
                    continue
                else:
                    sent_lines.append(line)
            elif len(sent_lines) > 0:
                example = get_example_from_lines(sent_lines)
                example['sent_id'] = len(bio2_examples)
                bio2_examples.append(example)
                sent_lines = []
        if len(sent_lines) > 0:
            example = get_example_from_lines(sent_lines)
            example['sent_id'] = len(bio2_examples)
            bio2_examples.append(example)

    return bio2_examples


def generate_vocabs(datasets, coref=False,
                    relation_directional=False,
                    symmetric_relations=None):
    """Generate vocabularies from a list of data sets
    :param datasets (list): A list of data sets
    :return (dict): A dictionary of vocabs
    """
    entity_type_set = set()
    event_type_set = set()
    relation_type_set = set()
    role_type_set = set()
    for dataset in datasets:
        entity_type_set.update(dataset.entity_type_set)
        event_type_set.update(dataset.event_type_set)
        relation_type_set.update(dataset.relation_type_set)
        role_type_set.update(dataset.role_type_set)

    # add inverse relation types for non-symmetric relations
    if relation_directional:
        if symmetric_relations is None:
            symmetric_relations = []
        relation_type_set_ = set()
        for relation_type in relation_type_set:
            relation_type_set_.add(relation_type)
            if relation_directional and relation_type not in symmetric_relations:
                relation_type_set_.add(relation_type + '_inv')

    # entity and trigger labels
    prefix = ['B', 'I']
    entity_label_stoi = {'O': 0}
    trigger_label_stoi = {'O': 0}
    for t in entity_type_set:
        for p in prefix:
            entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)
    for t in event_type_set:
        for p in prefix:
            trigger_label_stoi['{}-{}'.format(p, t)] = len(trigger_label_stoi)

    entity_type_stoi = {k: i for i, k in enumerate(entity_type_set, 1)}
    entity_type_stoi['O'] = 0

    event_type_stoi = {k: i for i, k in enumerate(event_type_set, 1)}
    event_type_stoi['O'] = 0

    relation_type_stoi = {k: i for i, k in enumerate(relation_type_set, 1)}
    relation_type_stoi['O'] = 0
    if coref:
        relation_type_stoi['COREF'] = len(relation_type_stoi)

    role_type_stoi = {k: i for i, k in enumerate(role_type_set, 1)}
    role_type_stoi['O'] = 0

    mention_type_stoi = {'NAM': 0, 'NOM': 1, 'PRO': 2, 'UNK': 3}

    return {
        'entity_type': entity_type_stoi,
        'event_type': event_type_stoi,
        'relation_type': relation_type_stoi,
        'role_type': role_type_stoi,
        'mention_type': mention_type_stoi,
        'entity_label': entity_label_stoi,
        'trigger_label': trigger_label_stoi,
    }


def load_valid_patterns(path, vocabs):
    event_type_vocab = vocabs['event_type']
    entity_type_vocab = vocabs['entity_type']
    relation_type_vocab = vocabs['relation_type']
    role_type_vocab = vocabs['role_type']

    # valid event-role
    valid_event_role = set()
    event_role = json.load(
        open(os.path.join(path, 'event_role.json'), 'r', encoding='utf-8'))
    for event, roles in event_role.items():
        if event not in event_type_vocab:
            continue
        event_type_idx = event_type_vocab[event]
        for role in roles:
            if role not in role_type_vocab:
                continue
            role_type_idx = role_type_vocab[role]
            valid_event_role.add(event_type_idx * 100 + role_type_idx)

    # valid relation-entity
    valid_relation_entity = set()
    relation_entity = json.load(
        open(os.path.join(path, 'relation_entity.json'), 'r', encoding='utf-8'))
    for relation, entities in relation_entity.items():
        relation_type_idx = relation_type_vocab[relation]
        for entity in entities:
            entity_type_idx = entity_type_vocab[entity]
            valid_relation_entity.add(
                relation_type_idx * 100 + entity_type_idx)

    # valid role-entity
    valid_role_entity = set()
    role_entity = json.load(
        open(os.path.join(path, 'role_entity.json'), 'r', encoding='utf-8'))
    for role, entities in role_entity.items():
        if role not in role_type_vocab:
            continue
        role_type_idx = role_type_vocab[role]
        for entity in entities:
            entity_type_idx = entity_type_vocab[entity]
            valid_role_entity.add(role_type_idx * 100 + entity_type_idx)

    return {
        'event_role': valid_event_role,
        'relation_entity': valid_relation_entity,
        'role_entity': valid_role_entity
    }


def read_ltf(path):
    root = et.parse(path, et.XMLParser(
        dtd_validation=False, encoding='utf-8')).getroot()
    doc_id = root.find('DOC').get('id')
    doc_tokens = []
    for seg in root.find('DOC').find('TEXT').findall('SEG'):
        seg_id = seg.get('id')
        seg_tokens = []
        seg_start = int(seg.get('start_char'))
        seg_text = seg.find('ORIGINAL_TEXT').text
        for token in seg.findall('TOKEN'):
            token_text = token.text
            start_char = int(token.get('start_char'))
            end_char = int(token.get('end_char'))
            assert seg_text[start_char - seg_start:
                            end_char - seg_start + 1
                   ] == token_text, 'token offset error'
            seg_tokens.append((token_text, start_char, end_char))
        doc_tokens.append((seg_id, seg_tokens))

    return doc_tokens, doc_id


def read_txt(path, language='english'):
    doc_id = os.path.basename(path)
    data = open(path, 'r', encoding='utf-8').read()
    data = [s.strip() for s in data.split('\n') if s.strip()]
    sents = [l for ls in [sent_tokenize(line, language=language) for line in data]
             for l in ls]
    doc_tokens = []
    offset = 0
    for sent_idx, sent in enumerate(sents):
        sent_id = '{}-{}'.format(doc_id, sent_idx)
        tokens = word_tokenize(sent)
        tokens = [(token, offset + i, offset + i + 1)
                  for i, token in enumerate(tokens)]
        offset += len(tokens)
        doc_tokens.append((sent_id, tokens))
    return doc_tokens, doc_id


def read_json(path):
    with open(path, 'r', encoding='utf-8') as r:
        data = [json.loads(line) for line in r]
    doc_id = data[0]['doc_id']
    offset = 0
    doc_tokens = []

    for inst in data:
        tokens = inst['tokens']
        tokens = [(token, offset + i, offset + i + 1)
                  for i, token in enumerate(tokens)]
        offset += len(tokens)
        doc_tokens.append((inst['sent_id'], tokens))
    return doc_tokens, doc_id


def read_json_single(path):
    with open(path, 'r', encoding='utf-8') as r:
        data = [json.loads(line) for line in r]
    doc_id = os.path.basename(path)
    doc_tokens = []
    for inst in data:
        tokens = inst['tokens']
        tokens = [(token, i, i + 1) for i, token in enumerate(tokens)]
        doc_tokens.append((inst['sent_id'], tokens))
    return doc_tokens, doc_id


def save_result(output_file, gold_graphs, pred_graphs, sent_ids, tokens=None):
    with open(output_file, 'w', encoding='utf-8') as w:
        for i, (gold_graph, pred_graph, sent_id) in enumerate(
                zip(gold_graphs, pred_graphs, sent_ids)):
            output = {'sent_id': sent_id,
                      'gold': gold_graph.to_dict(),
                      'pred': pred_graph.to_dict()}
            if tokens:
                output['tokens'] = tokens[i]
            w.write(json.dumps(output) + '\n')


def mention_to_tab(start, end, entity_type, mention_type, mention_id, tokens, token_ids, score=1):
    tokens = tokens[start:end]
    token_ids = token_ids[start:end]
    span = '{}:{}-{}'.format(token_ids[0].split(':')[0],
                             token_ids[0].split(':')[1].split('-')[0],
                             token_ids[1].split(':')[1].split('-')[1])
    mention_text = tokens[0]
    previous_end = int(token_ids[0].split(':')[1].split('-')[1])
    for token, token_id in zip(tokens[1:], token_ids[1:]):
        start, end = token_id.split(':')[1].split('-')
        start, end = int(start), int(end)
        mention_text += ' ' * (start - previous_end) + token
        previous_end = end
    return '\t'.join([
        'json2tab',
        mention_id,
        mention_text,
        span,
        'NIL',
        entity_type,
        mention_type,
        str(score)
    ])


def json_to_mention_results(input_dir, output_dir, file_name,
                            bio_separator=' '):
    mention_type_list = ['nam', 'nom', 'pro', 'nam+nom+pro']
    file_type_list = ['bio', 'tab']
    writers = {}
    for mention_type in mention_type_list:
        for file_type in file_type_list:
            output_file = os.path.join(output_dir, '{}.{}.{}'.format(file_name,
                                                                     mention_type,
                                                                     file_type))
            writers['{}_{}'.format(mention_type, file_type)
            ] = open(output_file, 'w')

    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    for f in json_files:
        with open(f, 'r', encoding='utf-8') as r:
            for line in r:
                result = json.loads(line)
                doc_id = result['doc_id']
                tokens = result['tokens']
                token_ids = result['token_ids']
                bio_tokens = [[t, tid, 'O']
                              for t, tid in zip(tokens, token_ids)]
                # separate bio output
                for mention_type in ['NAM', 'NOM', 'PRO']:
                    tokens_tmp = deepcopy(bio_tokens)
                    for start, end, enttype, mentype in result['graph']['entities']:
                        if mention_type == mentype:
                            tokens_tmp[start] = 'B-{}'.format(enttype)
                            for token_idx in range(start + 1, end):
                                tokens_tmp[token_idx] = 'I-{}'.format(
                                    enttype)
                    writer = writers['{}_bio'.format(mention_type.lower())]
                    for token in tokens_tmp:
                        writer.write(bio_separator.join(token) + '\n')
                    writer.write('\n')
                # combined bio output
                tokens_tmp = deepcopy(bio_tokens)
                for start, end, enttype, _ in result['graph']['entities']:
                    tokens_tmp[start] = 'B-{}'.format(enttype)
                    for token_idx in range(start + 1, end):
                        tokens_tmp[token_idx] = 'I-{}'.format(enttype)
                writer = writers['nam+nom+pro_bio']
                for token in tokens_tmp:
                    writer.write(bio_separator.join(token) + '\n')
                writer.write('\n')
                # separate tab output
                for mention_type in ['NAM', 'NOM', 'PRO']:
                    writer = writers['{}_tab'.format(mention_type.lower())]
                    mention_count = 0
                    for start, end, enttype, mentype in result['graph']['entities']:
                        if mention_type == mentype:
                            mention_id = '{}-{}'.format(doc_id, mention_count)
                            tab_line = mention_to_tab(
                                start, end, enttype, mentype, mention_id, tokens, token_ids)
                            writer.write(tab_line + '\n')
                # combined tab output
                writer = writers['nam+nom+pro_tab']
                mention_count = 0
                for start, end, enttype, mentype in result['graph']['entities']:
                    mention_id = '{}-{}'.format(doc_id, mention_count)
                    tab_line = mention_to_tab(
                        start, end, enttype, mentype, mention_id, tokens, token_ids)
                    writer.write(tab_line + '\n')
    for w in writers:
        w.close()


def normalize_score(scores):
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        return [0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def best_score_by_task(log_file, task, max_epoch=1000):
    with open(log_file, 'r', encoding='utf-8') as r:
        config = r.readline()

        best_scores = []
        best_dev_score = 0
        for line in r:
            record = json.loads(line)
            dev = record['dev']
            test = record['test']
            epoch = record['epoch']
            if epoch > max_epoch:
                break
            if dev[task]['f'] > best_dev_score:
                best_dev_score = dev[task]['f']
                best_scores = [dev, test, epoch]

        print('Epoch: {}'.format(best_scores[-1]))
        tasks = ['entity', 'mention', 'relation', 'trigger_id', 'trigger',
                 'role_id', 'role']
        for t in tasks:
            print('{}: dev: {:.2f}, test: {:.2f}'.format(t,
                                                         best_scores[0][t][
                                                             'f'] * 100.0,
                                                         best_scores[1][t][
                                                             'f'] * 100.0))


def prepare_sample_blink(data_dir):
    from trankit import Pipeline
    nlp = Pipeline('english')

    fpaths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.tsv')]

    random.seed(2022)
    random.shuffle(fpaths)

    selected_fpaths = fpaths
    total_meetings, total_sent, total_speakers, total_names = 0, 0, 0, 0
    found_speakers = 0

    all_sentences = []
    all_meetings = []
    for fpath in selected_fpaths:
        meeting = read_blink_file(nlp, fpath)
        total_speakers += len(set([s['speakerFaceId'] for s in meeting['sentences']]))
        
        speaker_ids = set()
        for s in meeting['sentences']:
            for ent in s['person_names']:
                if ent['is_speaker']:
                    speaker_ids.add(ent['speakerId'])
        found_speakers += len(speaker_ids)

        all_meetings.append(meeting)
        all_sentences.extend(meeting['sentences'])

        total_meetings += 1

    ensure_dir('datasets/blink')

    total_sent = len(all_sentences)

    for s in all_sentences:
        total_names += len(s['person_names'])

    final_meetings = []
    for m in all_meetings:
        final_meetings.append(m)

    print('statistics:')
    print('total meetings: {}, total sentences: {}, total speakers: {}, found speakers: {}, recall bound: {}, total names: {}'.format(total_meetings, total_sent, total_speakers, found_speakers, 100. * found_speakers/total_speakers,  total_names))

    with open('datasets/blink/all_meetings.json', 'w') as f:
        json.dump(final_meetings, f)

    dev = final_meetings

    print('-' * 20)
    print('generating data for model development')
    print('dev size: {} meetings, {} sentences, {} names'.format(len(dev), sum([len(m['sentences']) for m in dev]), sum([sum([len(s['person_names']) for s in m['sentences']]) for m in dev])))

    with open('datasets/blink/dev-meetings.blink.json', 'w') as f:
        json.dump(dev, f)

def sample_mediasum(data_dir):
    fpaths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.json')]

    random.seed(2022)
    random.shuffle(fpaths)

    selected_fpaths = fpaths[:200]
    total_meetings, total_sent, total_speakers, total_names = 0, 0, 0, 0

    all_sentences = []
    all_meetings = []
    ida2idb = {}
    for fpath in selected_fpaths:
        meeting = read_mediasum_file(fpath)
        for local_id, s in enumerate(meeting['sentences']):
            ida = len(all_sentences)
            idb = [len(all_meetings), local_id]

            ida2idb[ida] = idb

            all_sentences.append(s)

        all_meetings.append(meeting)


        total_meetings += 1

    ensure_dir('datasets/mediasum')

    total_sent = len(all_sentences)

    with open('datasets/mediasum/all_sentences.json', 'w') as f:
        for s in all_sentences:
            f.write(json.dumps(s) + '\n')

    # create input for NER model
    model_identity = 'CoNLL03-English-synthetic'
    input_file = 'datasets/mediasum/all_sentences.txt'
    output_file = 'datasets/mediasum/all_sentences.person-entities.json'

    with open(input_file, 'w') as f:
        for s in all_sentences:
            f.write(s['displayText'] + '\n')
    if not os.path.exists(output_file):
        os.system('python train.py --predict --dataset {} --pred_file {} --output {}'.format(model_identity, input_file, output_file))
        with open(output_file) as f:
            tagged_sentences = [json.loads(line.strip()) for line in f if line.strip()]
    else:
        with open(output_file) as f:
            tagged_sentences = [json.loads(line.strip()) for line in f if line.strip()]

        if len(tagged_sentences) < len(all_sentences):
            os.system('python train.py --predict --dataset {} --pred_file {} --output {}'.format(model_identity, input_file, output_file))

            with open(output_file) as f:
                tagged_sentences = [json.loads(line.strip()) for line in f if line.strip()]

    for ida in range(len(tagged_sentences)):
        l = tagged_sentences[ida]
        assert ida in ida2idb

        idb = ida2idb[ida]

        s = all_meetings[idb[0]]['sentences'][idb[1]]

        assert s['displayText'].strip() == l['text'].strip(), '{}\n{}\n'.format(s['displayText'], l['text'])

        s['tokens'] = l['tokens']
        s['person_names'] = [ent for ent in l['entity_mentions'] if ent['entity_type'] == 'PER']
        if len(s['person_names']) > 0:
            s['has_names'] = 'True'

        total_names += len(s['person_names'])

    meetings_without_names = 0
    final_meetings = []
    for m in all_meetings:
        names = 0
        for s in m['sentences']:
            if 'has_names' in s:
                names += 1
        if names == 0:
            meetings_without_names += 1
        else:
            final_meetings.append(m)

    print('statistics:')
    print('total meetings: {}, meetings without names: {}, total sentences: {}, total speakers: {}, total names: {}'.format(total_meetings, meetings_without_names, total_sent, total_speakers, total_names))

    with open('datasets/mediasum/all_meetings.json', 'w') as f:
        json.dump(final_meetings, f)

    train_size = int(0.8 * len(final_meetings))
    dev_size = int(0.1 * len(final_meetings))
    train = final_meetings[:train_size]
    test = final_meetings[train_size: train_size + dev_size]
    dev = final_meetings[train_size + dev_size:]

    print('-' * 20)
    print('generating data for model development')
    print('train size: {} meetings, {} sentences, {} names'.format(len(train), sum([len(m['sentences']) for m in train]), sum([sum([len(s['person_names']) for s in m['sentences']]) for m in train])))
    print('dev size: {} meetings, {} sentences, {} names'.format(len(dev), sum([len(m['sentences']) for m in dev]), sum([sum([len(s['person_names']) for s in m['sentences']]) for m in dev])))
    print('test size: {} meetings, {} sentences, {} names'.format(len(dev), sum([len(m['sentences']) for m in dev]), sum([sum([len(s['person_names']) for s in m['sentences']]) for m in test])))

    process_speaker_data(input_data=train, output_dir='datasets/mediasum/train/')
    process_speaker_data(input_data=dev, output_dir='datasets/mediasum/dev/')
    process_speaker_data(input_data=test, output_dir='datasets/mediasum/test/')

    with open('datasets/mediasum/train-meetings.mediasum.json', 'w') as f:
        json.dump(train, f)

    with open('datasets/mediasum/dev-meetings.mediasum.json', 'w') as f:
        json.dump(dev, f)

    with open('datasets/mediasum/test-meetings.mediasum.json', 'w') as f:
        json.dump(test, f)

def process_speaker_data(input_data, output_dir):
    ensure_dir(output_dir)
    print('-' * 20)
    total_speakers = 0
    total_matched_speakers = 0
    total_names = 0
    total_matched_names = 0
    for meeting in input_data:
        multiple_word_speaker_faces = set()
        one_word_speaker_faces = set()
        for sent in meeting['sentences']:
            face = sent['speakerFaceId'].strip().split(',')[0].lower().split('(')[0].strip()
            sent['face'] = face

            if len(face.split()) >= 2:
                multiple_word_speaker_faces.add(face)
            else:
                one_word_speaker_faces.add(face)

        multiple_word_speaker_faces = list(multiple_word_speaker_faces)
        multiple_word_speaker_faces.sort()
        
        one_word_speaker_faces = list(one_word_speaker_faces)
        one_word_speaker_faces.sort()

        normalize_face = {}

        for face in multiple_word_speaker_faces:
            normalize_face[face] = face

        for face in one_word_speaker_faces:
            matching = process.extractOne(face, multiple_word_speaker_faces)
            if matching is not None:
                m_face, m_score = matching[0], matching[1]
            else:
                m_score = 0

            if m_score >= NAME_MATCHING_THRESHOLD or matching is not None and (face in m_face or m_face in face):
                normalize_face[face] = m_face
            else:
                normalize_face[face] = face

        face2id = {}
        for face in normalize_face:
            n_face = normalize_face[face]
            if n_face not in face2id:
                face2id[n_face] = len(face2id)

        meeting['normalize_face'] = normalize_face
        meeting['face2id'] = face2id

        total_speakers += len(face2id)

        available_faces = list(face2id.keys())

        matched_speaker_ids = set()

        for sent in meeting['sentences']:
            sent_text = sent['displayText']
            sent_tokens = sent['tokens']
            total_names += len(sent['person_names'])

            for entity in sent['person_names']:
                start = sent_tokens[entity['start_token']]['span'][0]
                end = sent_tokens[entity['end_token'] - 1]['span'][1]
                entity['text'] = sent_text[start: end]
                name_lowered = entity['text'].lower()

                matching = process.extractOne(name_lowered, available_faces)
                m_face, m_score = matching[0], matching[1]

                if m_score >= NAME_MATCHING_THRESHOLD or matching is not None and (name_lowered in m_face or m_face in name_lowered):
                    entity['is_speaker'] = True
                    entity['speakerId'] = str(face2id[m_face])
                    matched_speaker_ids.add(m_face)
                    total_matched_names += 1
                else:
                    entity['is_speaker'] = False
                    entity['speakerId'] = 'N/A'

        total_matched_speakers += len(matched_speaker_ids)

    print('num meetings: {}, total_sentences: {}, total speakers: {}, found speakers: {}/{} ~ {}%, total_names: {}, names matching speakers: {}/{} ~ {}%'.format(
        len(input_data),
        sum([len(x['sentences']) for x in input_data]),
        total_speakers,
        total_matched_speakers,
        total_speakers,
        total_matched_speakers * 100. / total_speakers,
        total_names,
        total_matched_names,
        total_names,
        total_matched_names * 100./total_names))

    print('writing to xlsx files...')
    for meeting in input_data:
        out_file = os.path.join(output_dir, meeting['doc_id'].rstrip('.json') + '.tsv')
        out = []
        out.append('speakerFaceName\tspeakerFaceId\tsentenceId\tdisplayText\tpersonNames\tspeakerIds')
        face2id = meeting['face2id']
        normalize_face = meeting['normalize_face']

        for sid, sent in enumerate(meeting['sentences']):
            cols = []
            cols.append(sent['speakerFaceId'])
            cols.append(str(face2id[normalize_face[sent['face']]]))
            cols.append(str(sid))

            cols.append(sent['displayText'])
            cols.append(','.join([entity['text'] for entity in sent['person_names']]))
            cols.append(','.join([str(entity['speakerId']) for entity in sent['person_names']]))

            out.append('\t'.join(cols))
        with open(out_file, 'w') as f:
            f.write('\n'.join(out))

        tsv_to_xlsx(out_file, out_file.rstrip('.tsv') + '.xlsx')

def tsv_to_xlsx(tsv_file, xlsx_file):
    # Creating an XlsxWriter workbook object and adding 
    # a worksheet.
    workbook = Workbook(xlsx_file)
    worksheet = workbook.add_worksheet()
    
    # Reading the tsv file.
    read_tsv = csv.reader(open(tsv_file, 'r', encoding='utf-8'), delimiter='\t')
      
    # We'll use a loop with enumerate to pass the data 
    # together with its row position number, which we'll
    # use as the cell number in the write_row() function.
    for row, data in enumerate(read_tsv):
        worksheet.write_row(row, 0, data)

    # Closing the xlsx file.
    workbook.close()
    os.system('rm -rf {}'.format(tsv_file))


def read_mediasum_file(fpath):
    with open(fpath) as f:
        data = json.load(f)

    return {'doc_id': os.path.basename(fpath), 'sentences': [{'sentence_id': '{}-{}'.format(os.path.basename(fpath), i), 'displayText': sent['displayText'].replace('\n', ' '), 'speakerFaceId': sent['speakerFaceId']} for i, sent in enumerate(data['sentences'])]}


def read_blink_file(nlp, fpath):
    with open(fpath) as f:
        lines = [line.strip() for line in f if line.strip()][1:]
    doc_id = os.path.basename(fpath)
    data = {'doc_id': doc_id, 'sentences': []}

    progress = tqdm.tqdm(total=len(lines), ncols=75,
                         desc='processing')
    for line in lines:
        progress.update(1)
        feats = line.split('\t')
        assert len(feats) >= 3

        sent = {
                'sentence_id': '{}-{}'.format(doc_id, feats[1]),
                'displayText': feats[2].strip(),
                'speakerFaceId': feats[0].strip()
        }

        if sent['speakerFaceId'].startswith('off'):
            continue

        sent['tokens'] = nlp.tokenize(sent['displayText'], is_sent=True)['tokens']

        cid2tid = {}
        for tid in range(len(sent['tokens'])):
            token = sent['tokens'][tid]
            for cid in range(token['span'][0], token['span'][1]):
                cid2tid[cid] = tid

        sent['person_names'] = []
        sent['has_names'] = False

        data['sentences'].append(sent)

        if len(feats) > 3:
            if len(feats) == 4:
                feats.append(','.join(['N/A'] * len(feats[3].split(','))))

            assert len(feats) == 5 and len(feats[3].split(',')) == len(feats[4].split(',')), 'file_name: {}, sentence id: {}, {}'.format(doc_id, feats[1], feats)

            sent['has_names'] = True

            offset = 0
            for name, speaker_id in zip(feats[3].split(','), feats[4].split(',')):
                new_ent = {'text': name}

                start_token = 1000000
                end_token = -1

                start_char = offset + sent['displayText'][offset:].index(name)
                assert start_char >= 0

                end_char = start_char + len(name)
                offset = end_char

                for cid in range(start_char, end_char):
                    if cid not in cid2tid:
                        continue
                    start_token = min(start_token, cid2tid[cid])
                    end_token = max(end_token, cid2tid[cid])

                end_token += 1

                assert end_token > start_token >= 0, '{}'.format(sent)

                new_ent['start_token'] = start_token
                new_ent['end_token'] = end_token

                if speaker_id == 'N/A':
                    new_ent['is_speaker'] = False
                    new_ent['speakerId'] = 'N/A'
                else:
                    new_ent['is_speaker'] = True
                    new_ent['speakerId'] = speaker_id

                new_ent['entity_id'] = 'entity-{}'.format(len(sent['person_names']) + 1)

                sent['person_names'].append(new_ent)

    progress.close()

    return data

if __name__ == '__main__':
    sample_mediasum('datasets/mediaSum/transcripts/')
