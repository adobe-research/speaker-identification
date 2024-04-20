import os
import json
import random
import time
from argparse import ArgumentParser
import logging
import sys
from datetime import datetime
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, BertConfig, AdamW,
                          get_linear_schedule_with_warmup)

from joint_model import JointSpeakerIdentifier
from graph import Graph
from joint_iterators import JointSpeakerDataset

from individual_model import *
from individual_iterators import *
from scorer import *
from vocabs_util import data_vocabs

# configuration
parser = ArgumentParser()
parser.add_argument('--bert_model_name', default='roberta-large', type=str,
                    choices=['bert-large-cased', "bert-base-multilingual-cased", 'roberta-large', 'xlm-roberta-large'])
parser.add_argument('--bert_cache_dir', default='resource/bert', type=str)
parser.add_argument('--multi_piece_strategy', default='average', type=str)
parser.add_argument('--use_extra_bert', default=1, type=int)
parser.add_argument('--extra_bert', default=-3, type=int)
parser.add_argument('--bert_dropout', default=0.5, type=float)
parser.add_argument('--context_size', default=1000, type=int)

parser.add_argument('--linear_dropout', default=0.4, type=float)
parser.add_argument('--linear_bias', default=1, type=int)
parser.add_argument('--linear_activation', default='sigmoid', type=str)
parser.add_argument('--node_dim', default=200, type=int)
parser.add_argument('--hidden_num', default=300, type=int)
parser.add_argument('--entity_hidden_num', default=450, type=int)
parser.add_argument('--event_hidden_num', default=450, type=int)

parser.add_argument('--dataset', default='CoNLL03-English-synthetic', type=str)
parser.add_argument('--model_type', default='individual', type=str, choices=['individual', 'joint'])

parser.add_argument('--accumulate_step', default=1, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--eval_batch_size', default=5, type=int)
parser.add_argument('--augment_lowercase', default=0, type=int)
parser.add_argument('--max_epoch', default=100, type=int)
parser.add_argument('--max_subwords', default=350, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--bert_learning_rate', default=1e-5, type=float)
parser.add_argument('--weight_decay', default=1e-3, type=float)
parser.add_argument('--bert_weight_decay', default=1e-5, type=float)
parser.add_argument('--grad_clipping', default=5.0, type=float)
parser.add_argument('--seed', default=3456, type=int)
parser.add_argument('--output_dir', default='', type=str)

parser.add_argument('--debug', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--use_patterns', action='store_true')
parser.add_argument('--predict', action='store_true')
parser.add_argument('--pred_file', default='input.txt', type=str)
parser.add_argument('--output', default='output.json', type=str)
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--gpu_device', default=0, type=int)

config = parser.parse_args()

os.environ['PYTHONHASHSEED'] = str(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if config.bert_model_name == 'roberta-large':
    config.accumulate_step = 2
    config.bert_dropout = 0.1
    config.bert_learning_rate = 5e-6
elif config.bert_model_name == 'xlm-roberta-large':
    config.batch_size = 16
    config.accumulate_step = 8
    config.bert_dropout = 0.1
    config.bert_learning_rate = 5e-6

if config.dataset.startswith('CoNLL03'):
    tasks = ['entity']
    config.tasks = set(tasks)

    config.train_file = "datasets/{}/train.bio".format(config.dataset)
    config.dev_file = "datasets/{}/dev.bio".format(config.dataset)

    if len(config.output_dir.strip()) == 0:
        output_dir = 'logs/{}'.format(config.dataset)
    else:
        output_dir = config.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # logging
    for name in logging.root.manager.loggerDict:
        if 'transformers' in name:
            logging.getLogger(name).setLevel(logging.CRITICAL)

    logging.basicConfig(format='%(message)s', level=logging.INFO,
                        filename=os.path.join(output_dir, '{}.log'.format(config.dataset)),
                        filemode='w')
    logger = logging.getLogger(__name__)


    def printlog(message, printout=True):
        if printout:
            print(message)
        logger.info(message)


    running_command = 'python ' + ' '.join([x for x in sys.argv])
    printlog('Running command: {}'.format(running_command))

    # set GPU device
    use_gpu = config.use_gpu
    if use_gpu and config.gpu_device >= 0:
        torch.cuda.set_device(config.gpu_device)

    # output
    best_model_fpath = os.path.join(output_dir, 'best-model.mdl')
    vocabs = data_vocabs[config.dataset]
    config.vocabs = vocabs
    # datasets
    model_name = config.bert_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=config.bert_cache_dir)
    config.tokenizer = tokenizer

    # initialize the model
    model = TranscriptNER(config, vocabs)

    if config.predict:
        from trankit import Pipeline

        config.trankit_pipeline = Pipeline('english')
        pred_set = NERDatasetPred(config, config.pred_file, gpu=use_gpu)
        pred_set.numberize(tokenizer, vocabs)
        pred_batch_num = len(pred_set) // config.eval_batch_size + \
                         (len(pred_set) % config.eval_batch_size != 0)
        torch.cuda.empty_cache()
        model.eval()

        saved_ckpt = torch.load(best_model_fpath)

        model.load_state_dict(saved_ckpt['model'])

        # pred set
        printlog('=' * 20)
        progress = tqdm.tqdm(total=pred_batch_num, ncols=75,
                             desc='Predict')

        with open(config.output, 'w') as f:
            f.write('')

        outputs = []
        for batch in DataLoader(pred_set, batch_size=config.eval_batch_size,
                                shuffle=False, collate_fn=pred_set.collate_fn):
            progress.update(1)
            graphs = model.predict(batch)

            for sent_id, text, tokens, graph in zip(batch.sent_ids, batch.texts, batch.tokens, graphs):
                res = {
                    'sent_id': sent_id, 'text': text, 'tokens': tokens,
                    'entity_mentions': [], 'entities': []
                }

                for entity in graph.entities:
                    entity_type = model.entity_type_itos.get(entity[2], 'O')
                    if entity_type != 'O':
                        res['entity_mentions'].append({
                            'entity_id': 'entity-{}'.format(len(res['entity_mentions']) + 1),
                            'start_token': entity[0], 'end_token': entity[1], 'entity_type': entity_type
                        })
                        start = entity[0]
                        end = entity[1]

                        res['entities'].append({'text': text[tokens[start]['span'][0]: tokens[end - 1]['span'][1]],
                                                'entity_type': entity_type})

                with open(config.output, 'a') as f:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')
        progress.close()
        print('Writing predictions to {} ...'.format(config.output))
        print('Done!')
    elif config.eval:
        train_set = NERDataset(config, config.train_file, gpu=use_gpu)
                               
        dev_set = NERDataset(config, config.dev_file, gpu=use_gpu)

        train_set.numberize(tokenizer, vocabs)
        dev_set.numberize(tokenizer, vocabs)

        batch_num = len(train_set) // config.batch_size
        dev_batch_num = len(dev_set) // config.eval_batch_size + \
                        (len(dev_set) % config.eval_batch_size != 0)

        torch.cuda.empty_cache()
        model.eval()

        saved_ckpt = torch.load(best_model_fpath)

        model.load_state_dict(saved_ckpt['model'])

        # dev set
        printlog('=' * 20)
        progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                             desc='Dev')
        dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
        for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                                shuffle=False, collate_fn=dev_set.collate_fn):
            progress.update(1)
            graphs = model.predict(batch)

            dev_gold_graphs.extend(batch.graphs)
            dev_pred_graphs.extend(graphs)
            dev_sent_ids.extend(batch.sent_ids)
            dev_tokens.extend(batch.tokens)
        progress.close()
        torch.cuda.empty_cache()

        dev_scores = score_graphs(logger, tasks, dev_gold_graphs, dev_pred_graphs)
                                  

        printlog('=' * 20)
    else:

        train_set = NERDataset(config, config.train_file, gpu=use_gpu)
                               
        dev_set = NERDataset(config, config.dev_file, gpu=use_gpu)

        train_set.numberize(tokenizer, vocabs)
        dev_set.numberize(tokenizer, vocabs)

        batch_num = len(train_set) // config.batch_size
        dev_batch_num = len(dev_set) // config.eval_batch_size + \
                        (len(dev_set) % config.eval_batch_size != 0)

        logger.info('================= Trainable params ({:.2f}M) ================='.format(
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.))
        for n, p in model.named_parameters():
            if p.requires_grad:
                logger.info('{}\t\t{}'.format(n, list(p.shape)))

        logger.info('==============================================================')

        # optimizer
        param_groups = [
            {
                'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
                'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                           and 'crf' not in n],
                'lr': config.learning_rate, 'weight_decay': config.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                           and 'crf' in n],
                'lr': config.learning_rate, 'weight_decay': 0
            }
        ]
        optimizer = AdamW(params=param_groups)
        schedule = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=batch_num * 5,
                                                   num_training_steps=batch_num * 80)

        # model state
        state = dict(model=model.state_dict(),
                     vocabs=vocabs)

        best_dev = {}
        best_test = {}
        for epoch in range(config.max_epoch):
            printlog('*' * 50)
            printlog('Epoch: {}'.format(epoch))
            printlog('Running: {}'.format(running_command))

            # training set
            progress = tqdm.tqdm(total=batch_num, ncols=75,
                                 desc='Train {}'.format(epoch))
            model.train()
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(DataLoader(
                    train_set, batch_size=config.batch_size // config.accumulate_step,
                    shuffle=True, drop_last=True, collate_fn=train_set.collate_fn)):

                loss = model(batch)
                loss = loss * (1 / config.accumulate_step)
                loss.backward()

                if (batch_idx + 1) % config.accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clipping)
                    optimizer.step()
                    schedule.step()
                    optimizer.zero_grad()

                if (batch_idx + 1) % 20 == 0:
                    printlog('{}: step: {}/{}, loss: {}'.format(datetime.now(), batch_idx + 1, batch_num, loss.item()),
                             printout=False)

            progress.close()
            torch.cuda.empty_cache()
            model.eval()

            # dev set
            printlog('=' * 20)
            progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                                 desc='Dev {}'.format(epoch))
            dev_gold_graphs, dev_pred_graphs, dev_sent_ids, dev_tokens = [], [], [], []
            for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                                    shuffle=False, collate_fn=dev_set.collate_fn):
                progress.update(1)
                graphs = model.predict(batch)

                dev_gold_graphs.extend(batch.graphs)
                dev_pred_graphs.extend(graphs)
                dev_sent_ids.extend(batch.sent_ids)
                dev_tokens.extend(batch.tokens)
            progress.close()
            torch.cuda.empty_cache()

            dev_scores = score_graphs(logger, tasks, dev_gold_graphs, dev_pred_graphs)
                                     

            printlog('=' * 20)
            if epoch == 0 or np.mean([dev_scores[task]['f'] for task in tasks]) > np.mean(
                    [best_dev[task]['f'] for task in tasks]):
                best_dev = dev_scores

                printlog('-' * 10)
                printlog('New best model: mean(dev[task]) = {:.2f}'.format(
                    np.mean([dev_scores[task]['f'] for task in tasks]) * 100.))
                torch.save(state, best_model_fpath)

            if len(best_dev) == 0:
                continue

            printlog('=' * 20)
            printlog('Current best results:')
            printlog('Dev:')
            for task in tasks:
                if task + '_id' in best_dev:
                    printlog('{}: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(task + '_id',
                                                                          best_dev[task + '_id']['prec'] * 100.0,
                                                                          best_dev[task + '_id']['rec'] * 100.0,
                                                                          best_dev[task + '_id']['f'] * 100.0))

                printlog('{}: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(task,
                                                                      best_dev[task]['prec'] * 100.0,
                                                                      best_dev[task]['rec'] * 100.0,
                                                                      best_dev[task]['f'] * 100.0))

            printlog('-' * 5)
            printlog('mean(dev[task]) = {:.2f}'.format(np.mean([best_dev[task]['f'] for task in tasks]) * 100.))

            printlog('-' * 10)
elif config.dataset in ['mediasum']:
    config.train_file = "datasets/{}/train-meetings.{}.json".format(config.dataset, config.dataset)
    config.dev_file = "datasets/{}/dev-meetings.{}.json".format(config.dataset, config.dataset)
    config.test_file = "datasets/{}/test-meetings.{}.json".format(config.dataset, config.dataset)

    if len(config.output_dir.strip()) == 0:
        output_dir = 'logs/{}'.format(config.dataset + '-{}'.format(config.model_type))
    else:
        output_dir = config.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # logging
    for name in logging.root.manager.loggerDict:
        if 'transformers' in name:
            logging.getLogger(name).setLevel(logging.CRITICAL)

    logging.basicConfig(format='%(message)s', level=logging.INFO,
                        filename=os.path.join(output_dir, '{}.log'.format(config.dataset)),
                        filemode='w')
    logger = logging.getLogger(__name__)


    def printlog(message, printout=True):
        if printout:
            print(message)
        logger.info(message)


    running_command = 'python ' + ' '.join([x for x in sys.argv])
    printlog('Running command: {}'.format(running_command))

    # set GPU device
    use_gpu = config.use_gpu
    if use_gpu and config.gpu_device >= 0:
        torch.cuda.set_device(config.gpu_device)

    # output
    best_model_fpath = os.path.join(output_dir, 'best-model.mdl')
    # datasets
    model_name = config.bert_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=config.bert_cache_dir)
    config.tokenizer = tokenizer

    # initialize the model
    model = JointSpeakerIdentifier(config) if config.model_type == 'joint' else IndividualSpeakerIdentifier(config)

    if config.eval:
        dev_set = JointSpeakerDataset(config, config.dev_file) if config.model_type == 'joint' else IndividualSpeakerDataset(config, config.dev_file)
        dev_set.numberize(tokenizer)

        dev_batch_num = len(dev_set) // config.eval_batch_size + \
                        (len(dev_set) % config.eval_batch_size != 0)

        best_dev = {}
        torch.cuda.empty_cache()
        model.eval()
        model.load_state_dict(torch.load(best_model_fpath)['model'])

        # dev set
        printlog('=' * 20)
        progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                             desc='Dev')
        dev_predictions = []
        for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                                shuffle=False, collate_fn=dev_set.collate_fn):
            progress.update(1)
            preds = model.predict(batch)

            dev_predictions.extend(preds)

        progress.close()
        torch.cuda.empty_cache()

        dev_scores = compute_speaker_scores(logger, dev_set.meetings, dev_predictions)

        printlog('=' * 20)
        best_dev = dev_scores

        printlog('=' * 20)
        printlog('Current best results:')
        printlog('Dev:')
        printlog('precision: {:.2f}, recall: {:.2f}, f1: {:.2f}, accuracy: {:.2f}'.format(best_dev['precision'],
                                                                                          best_dev['recall'],
                                                                                          best_dev['f1'],
                                                                                          best_dev['accuracy']))

        printlog('-' * 10)
    else:
        train_set = JointSpeakerDataset(config, config.train_file) if config.model_type == 'joint' else IndividualSpeakerDataset(config, config.train_file)
        dev_set = JointSpeakerDataset(config, config.dev_file) if config.model_type == 'joint' else IndividualSpeakerDataset(config, config.dev_file)
        test_set = JointSpeakerDataset(config, config.test_file) if config.model_type == 'joint' else IndividualSpeakerDataset(config, config.test_file)

        train_set.numberize(tokenizer)
        dev_set.numberize(tokenizer)
        test_set.numberize(tokenizer)

        batch_num = len(train_set) // config.batch_size
        dev_batch_num = len(dev_set) // config.eval_batch_size + \
                        (len(dev_set) % config.eval_batch_size != 0)
        test_batch_num = len(test_set) // config.eval_batch_size + \
                        (len(test_set) % config.eval_batch_size != 0)

        logger.info('================= Trainable params ({:.2f}M) ================='.format(
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.))
        for n, p in model.named_parameters():
            if p.requires_grad:
                logger.info('{}\t\t{}'.format(n, list(p.shape)))

        logger.info('==============================================================')

        # optimizer
        param_groups = [
            {
                'params': [p for n, p in model.named_parameters() if n.startswith('bert')],
                'lr': config.bert_learning_rate, 'weight_decay': config.bert_weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                           and 'crf' not in n],
                'lr': config.learning_rate, 'weight_decay': config.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if not n.startswith('bert')
                           and 'crf' in n],
                'lr': config.learning_rate, 'weight_decay': 0
            }
        ]
        optimizer = AdamW(params=param_groups)
        schedule = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=batch_num * 5,
                                                   num_training_steps=batch_num * 80)

        # model state
        state = dict(model=model.state_dict())

        best_dev = {}
        best_test = {}
        for epoch in range(config.max_epoch):
            printlog('*' * 50)
            printlog('Epoch: {}'.format(epoch))
            printlog('Running: {}'.format(running_command))

            # training set
            progress = tqdm.tqdm(total=batch_num, ncols=75,
                                 desc='Train {}'.format(epoch))
            model.train()
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(DataLoader(
                    train_set, batch_size=config.batch_size // config.accumulate_step,
                    shuffle=True, drop_last=True, collate_fn=train_set.collate_fn)):

                loss = model(batch)
                loss = loss * (1 / config.accumulate_step)
                loss.backward()

                if (batch_idx + 1) % config.accumulate_step == 0:
                    progress.update(1)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clipping)
                    optimizer.step()
                    schedule.step()
                    optimizer.zero_grad()

                if (batch_idx + 1) % 20 == 0:
                    printlog('{}: step: {}/{}, loss: {}'.format(datetime.now(), batch_idx + 1, batch_num, loss.item()),
                             printout=False)

            progress.close()
            torch.cuda.empty_cache()
            model.eval()

            # dev set
            printlog('=' * 20)
            progress = tqdm.tqdm(total=dev_batch_num, ncols=75,
                                 desc='Dev {}'.format(epoch))
            dev_predictions = []
            for batch in DataLoader(dev_set, batch_size=config.eval_batch_size,
                                    shuffle=False, collate_fn=dev_set.collate_fn):
                progress.update(1)
                preds = model.predict(batch)

                dev_predictions.extend(preds)

            progress.close()
            torch.cuda.empty_cache()

            dev_scores = compute_speaker_scores(logger, dev_set.meetings, dev_predictions)

            printlog('=' * 20)
            # test set
            printlog('=' * 20)
            progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                                 desc='test {}'.format(epoch))
            test_predictions = []
            for batch in DataLoader(test_set, batch_size=config.eval_batch_size,
                                    shuffle=False, collate_fn=test_set.collate_fn):
                progress.update(1)
                preds = model.predict(batch)

                test_predictions.extend(preds)

            progress.close()
            torch.cuda.empty_cache()

            test_scores = compute_speaker_scores(logger, test_set.meetings, test_predictions)

            printlog('=' * 20)
            if epoch == 0 or (dev_scores['f1'] + dev_scores['accuracy']) > (best_dev['f1'] + best_dev['accuracy']):
                best_dev = dev_scores
                best_test = test_scores

                printlog('-' * 10)
                printlog('New best model...')
                torch.save(state, best_model_fpath)

            if len(best_dev) == 0:
                continue

            printlog('=' * 20)
            printlog('Current best results:')
            printlog('Dev:')
            printlog('precision: {:.2f}, recall: {:.2f}, f1: {:.2f}, accuracy: {:.2f}'.format(best_dev['precision'],
                                                                                              best_dev['recall'],
                                                                                              best_dev['f1'],
                                                                                              best_dev['accuracy']))

            printlog('Test:')
            printlog('precision: {:.2f}, recall: {:.2f}, f1: {:.2f}, accuracy: {:.2f}'.format(best_test['precision'],
                                                                                              best_test['recall'],
                                                                                              best_test['f1'],
                                                                                              best_test['accuracy']))

            printlog('-' * 10)
