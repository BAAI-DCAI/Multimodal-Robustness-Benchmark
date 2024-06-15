import os
import json
import argparse


def calculate_mc(pred_list, label_list, question_type, question_id):
    posrhtnegrht_record = {'pos_rht_neg_rht': 0, 'character_character': 0, 'attribute_posture': 0, 'attribute_color&texture': 0, 'attribute_existential': 0, 
              'attribute_position': 0, 'attribute_number': 0, 'attribute_shape': 0, 'context_abstract': 0, 'context_activity': 0,
              'context_concrete': 0, 'context_expert': 0, 'context_relation': 0}
    total_record = {'character_character': 0, 'attribute_existential': 0, 'attribute_color&texture': 0, 'attribute_number': 0, 
                    'attribute_shape': 0, 'attribute_posture': 0,  'attribute_position': 0,   'context_abstract': 0, 'context_concrete': 0,
                    'context_expert': 0, 'context_activity': 0, 'context_relation': 0}
    posghtnegwro_record = {'pos_rht_neg_wro': 0, 'character_character': 0, 'attribute_posture': 0, 'attribute_color&texture': 0, 'attribute_existential': 0, 
              'attribute_position': 0, 'attribute_number': 0, 'attribute_shape': 0, 'context_abstract': 0, 'context_activity': 0,
              'context_concrete': 0, 'context_expert': 0, 'context_relation': 0}
    total = len(pred_list) / 2

    for i in range(0, len(pred_list), 2):
        if pred_list[i] == label_list[i] and pred_list[i+1] == label_list[i+1]:
            posrhtnegrht_record['pos_rht_neg_rht']+=1
            posrhtnegrht_record[question_id[i]]+=1
        elif (pred_list[i] == label_list[i] and pred_list[i+1] != label_list[i+1] and question_type[i] == 'POS') or (pred_list[i] != label_list[i] and pred_list[i+1] == label_list[i+1] and question_type[i+1] == 'POS'):
            posghtnegwro_record['pos_rht_neg_wro']+=1
            posghtnegwro_record[question_id[i]]+=1
        total_record[question_id[i]]+=1

    avg_acc = posrhtnegrht_record['pos_rht_neg_rht'] / total
    misleading_acc = posghtnegwro_record['pos_rht_neg_wro'] / (posrhtnegrht_record['pos_rht_neg_rht'] + posghtnegwro_record['pos_rht_neg_wro'])
    print('================MC Accuracy: {:.2f}%'.format(avg_acc * 100))
    for key, value in total_record.items():
        if value == 0:
            print(f'{key} Accuracy:0%')
        else:
            print(f'{key} Accuracy:{(posrhtnegrht_record[key]/(value))*100}%')
    print('================MISLEAD Rate: {:.2f}%'.format(misleading_acc * 100))
    for key, value in total_record.items():
        if (posrhtnegrht_record[key] + posghtnegwro_record[key]) == 0:
            print(f'{key} Rate:0%')
        else:
            print(f'{key} Rate:{(posghtnegwro_record[key]/(posrhtnegrht_record[key] + posghtnegwro_record[key]))*100}%')


def eval_mma(args):

    with open(args.result_file, 'r') as f:
        result_file = json.load(f)
    with open(args.groudtruth, 'r') as f:
        data = json.load(f)
    print(f"before process predict number: {len(result_file)}")
    print(f"before process label number: {len(data)}")
    pred_list = []
    label_list = []
    question_type_list = []
    question_id = []

    for result, item in zip(result_file, data):
        try:
            text = result[1]['text']
        except Exception as e:
            text = 'E'
        question_type_list.append(result[2])
        question_id.append(result[0].split('/')[0]+'_'+result[0].split('/')[1])
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', ' ')
        words = text.split(' ')

        if 'A' in words:
            pred_list.append('A')
        elif 'B' in words:
            pred_list.append('B')
        elif 'C' in words:
            pred_list.append('C')
        elif 'D' in words:
            pred_list.append('D')
        else:
            pred_list.append('E')

        if item['ground_truth']=='A':
            label_list.append('A')
        elif item['ground_truth']=='B':
            label_list.append('B')
        elif item['ground_truth']=='C':
            label_list.append('C')
        elif item['ground_truth']=='D':
            label_list.append('D')

    print(f"after process predict number: {len(pred_list)}")
    print(f"after process label number: {len(label_list)}")
    print('================multi-choice================')
    calculate_mc(pred_list, label_list, question_type_list, question_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default='result.json')
    parser.add_argument("--groundtruth", type=str, default='MMR-benchmark.json')
    args = parser.parse_args()

    eval_mma(args)
