from collections import defaultdict
from util import *


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def convert_arguments(triggers, entities, roles):
    args = set()
    for trigger_idx, entity_idx, role in roles:
        arg_start, arg_end, _ = entities[entity_idx]
        trigger_label = triggers[trigger_idx][-1]
        args.add((arg_start, arg_end, trigger_label, role))
    return args


def compute_speaker_scores(logger, meetings, predictions):
    output_scores = None
    for mode in ['micro', 'N/A', 'prev', 'cur', 'next']:
        if mode == 'micro':
            correct = 0
            meeting2preds = defaultdict(list)
            meeting_name2speaker_id = {}
            for pred in predictions:
                if pred['pred-speaker-id'] != 'N/A':
                    meeting2preds[pred['meeting-id']].append(pred)

                meeting_name2speaker_id['{}.{}'.format(pred['meeting-id'], pred['person-name'])] = [pred['pred-speaker-id'], pred['pred-score']]
                if pred['pred-speaker-id'] == pred['gold-speaker-id']:
                    correct += 1
            print('-' * 20)
            logger.info('-' * 20)
            accuracy = correct * 100. / len(predictions)
            print('Accuracy: {}/{} = {:.2f}'.format(correct, len(predictions), accuracy))
            logger.info('Accuracy: {}/{} = {:.2f}'.format(correct, len(predictions), accuracy))

            true_positives = 0
            false_positives = 0
            total_positives = 0

            for meeting in meetings:
                face2id = meeting['face2id'] if 'face2id' in meeting else {}
                normalize_face = meeting['normalize_face'] if 'normalize_face' in meeting else {}

                preds = meeting2preds[meeting['doc_id']]
                speaker_id2pred_names = defaultdict(list)
                
                for p in preds:
                    speaker_id = p['pred-speaker-id']
                    speaker_id2pred_names[speaker_id].append([p['gold-speaker-id'], p['pred-score']])

                for speaker_id in speaker_id2pred_names:
                    gold_speaker_id, final_score = max(speaker_id2pred_names[speaker_id], key=lambda x: x[1])

                    if speaker_id == gold_speaker_id: 
                        true_positives += 1
                    else:
                        false_positives += 1
                if 'face2id' in meeting:
                    total_positives += len(set(meeting['face2id'].values()))
                else:
                    total_positives += len(set([s['speakerFaceId'] for s in meeting['sentences']]))

            precision = true_positives * 100. / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
            recall = true_positives * 100. / total_positives
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            output_scores = {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
        elif mode == 'N/A':
            correct = 0
            meeting2preds = defaultdict(list)
            meeting_name2speaker_id = {}
            for pred in predictions:
                if pred['gold-speaker-id-relative'] != 'N/A':
                    continue

                if pred['pred-speaker-id'] != 'N/A':
                    meeting2preds[pred['meeting-id']].append(pred)

                meeting_name2speaker_id['{}.{}'.format(pred['meeting-id'], pred['person-name'])] = [pred['pred-speaker-id'], pred['pred-score']]
                if pred['pred-speaker-id'] == pred['gold-speaker-id']:
                    correct += 1
            print('-' * 20)
            logger.info('-' * 20)
            accuracy = correct * 100. / len([p for p in predictions if p['gold-speaker-id-relative'] == 'N/A'])
            print('N/A Accuracy: {}/{} = {:.2f}'.format(correct, len([p for p in predictions if p['gold-speaker-id-relative'] == 'N/A']), accuracy))
            logger.info('N/A Accuracy: {}/{} = {:.2f}'.format(correct, len([p for p in predictions if p['gold-speaker-id-relative'] == 'N/A']), accuracy))
        elif mode == 'prev':
            correct = 0
            meeting2preds = defaultdict(list)
            meeting_name2speaker_id = {}
            for pred in predictions:
                if pred['gold-speaker-id-relative'] != 'prev':
                    continue

                if pred['pred-speaker-id'] != 'N/A':
                    meeting2preds[pred['meeting-id']].append(pred)

                meeting_name2speaker_id['{}.{}'.format(pred['meeting-id'], pred['person-name'])] = [pred['pred-speaker-id'], pred['pred-score']]
                if pred['pred-speaker-id'] == pred['gold-speaker-id']:
                    correct += 1
            print('-' * 20)
            logger.info('-' * 20)
            accuracy = correct * 100. / len([p for p in predictions if p['gold-speaker-id-relative'] == 'prev'])
            print('Prev Accuracy: {}/{} = {:.2f}'.format(correct, len([p for p in predictions if p['gold-speaker-id-relative'] == 'prev']), accuracy))
            logger.info('Prev Accuracy: {}/{} = {:.2f}'.format(correct, len([p for p in predictions if p['gold-speaker-id-relative'] == 'prev']), accuracy))
        elif mode == 'cur':
            correct = 0
            meeting2preds = defaultdict(list)
            meeting_name2speaker_id = {}
            for pred in predictions:
                if pred['gold-speaker-id-relative'] != 'cur':
                    continue

                if pred['pred-speaker-id'] != 'N/A':
                    meeting2preds[pred['meeting-id']].append(pred)

                meeting_name2speaker_id['{}.{}'.format(pred['meeting-id'], pred['person-name'])] = [pred['pred-speaker-id'], pred['pred-score']]
                if pred['pred-speaker-id'] == pred['gold-speaker-id']:
                    correct += 1
            print('-' * 20)
            logger.info('-' * 20)
            accuracy = correct * 100. / len([p for p in predictions if p['gold-speaker-id-relative'] == 'cur'])
            print('Cur Accuracy: {}/{} = {:.2f}'.format(correct, len([p for p in predictions if p['gold-speaker-id-relative'] == 'cur']), accuracy))
            logger.info('Cur Accuracy: {}/{} = {:.2f}'.format(correct, len([p for p in predictions if p['gold-speaker-id-relative'] == 'cur']), accuracy))
        elif mode == 'next':
            correct = 0
            meeting2preds = defaultdict(list)
            meeting_name2speaker_id = {}
            for pred in predictions:
                if pred['gold-speaker-id-relative'] != 'next':
                    continue

                if pred['pred-speaker-id'] != 'N/A':
                    meeting2preds[pred['meeting-id']].append(pred)

                meeting_name2speaker_id['{}.{}'.format(pred['meeting-id'], pred['person-name'])] = [pred['pred-speaker-id'], pred['pred-score']]
                if pred['pred-speaker-id'] == pred['gold-speaker-id']:
                    correct += 1
            print('-' * 20)
            logger.info('-' * 20)
            accuracy = correct * 100. / len([p for p in predictions if p['gold-speaker-id-relative'] == 'next'])
            print('Next Accuracy: {}/{} = {:.2f}'.format(correct, len([p for p in predictions if p['gold-speaker-id-relative'] == 'next']), accuracy))
            logger.info('Next Accuracy: {}/{} = {:.2f}'.format(correct, len([p for p in predictions if p['gold-speaker-id-relative'] == 'next']), accuracy))

    return output_scores


def score_graphs(logger, tasks, gold_graphs, pred_graphs):
                 
    gold_arg_num = pred_arg_num = arg_idn_num = arg_class_num = 0
    gold_trigger_num = pred_trigger_num = trigger_idn_num = trigger_class_num = 0
    gold_ent_num = pred_ent_num = ent_match_num = 0
    gold_rel_num = pred_rel_num = rel_match_num = 0
    gold_men_num = pred_men_num = men_match_num = 0

    for gold_graph, pred_graph in zip(gold_graphs, pred_graphs):
        # Entity
        gold_entities = gold_graph.entities
        pred_entities = pred_graph.entities
        gold_ent_num += len(gold_entities)
        pred_ent_num += len(pred_entities)
        ent_match_num += len([entity for entity in pred_entities
                              if entity in gold_entities])

    entity_prec, entity_rec, entity_f = compute_f1(
        pred_ent_num, gold_ent_num, ent_match_num)

    if 'entity' in tasks:
        print('entity: P: {:.2f}, R: {:.2f}, F: {:.2f} | TP: {}, FP: {}, FN: {}'.format(
            entity_prec * 100.0, entity_rec * 100.0, entity_f * 100.0, ent_match_num, pred_ent_num - ent_match_num,
            gold_ent_num - ent_match_num))

        logger.info('entity: P: {:.2f}, R: {:.2f}, F: {:.2f} | TP: {}, FP: {}, FN: {}'.format(
            entity_prec * 100.0, entity_rec * 100.0, entity_f * 100.0, ent_match_num, pred_ent_num - ent_match_num,
            gold_ent_num - ent_match_num))

    scores = {
        'entity': {'prec': entity_prec, 'rec': entity_rec, 'f': entity_f}
    }
    return scores
