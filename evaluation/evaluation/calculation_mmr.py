import os
import json
import argparse


def calculate_mc(pred_list, label_list, question_type, question_id):
    posrhtnegrht_record = {'pos_rht_neg_rht': 0, 'character': 0, 'semantics_attitude': 0, 'semantics_color&texture': 0, 'semantics_existential': 0, 
              'semantics_position': 0, 'semantics_quantity': 0, 'semantics_shape': 0, 'understanding_abstract_knowledge': 0, 'understanding_activity': 0,
              'understanding_embodied_knowledge': 0, 'understanding_professional_knowledge': 0, 'understanding_relation': 0}
    total_record = {'character': 0, 'semantics_existential': 0, 'semantics_color&texture': 0, 'semantics_quantity': 0, 
                    'semantics_shape': 0, 'semantics_attitude': 0,  'semantics_position': 0,   'understanding_abstract_knowledge': 0, 'understanding_embodied_knowledge': 0,
                    'understanding_professional_knowledge': 0, 'understanding_activity': 0, 'understanding_relation': 0}
    posghtnegwro_record = {'pos_rht_neg_wro': 0, 'character': 0, 'semantics_attitude': 0, 'semantics_color&texture': 0, 'semantics_existential': 0, 
              'semantics_position': 0, 'semantics_quantity': 0, 'semantics_shape': 0, 'understanding_abstract_knowledge': 0, 'understanding_activity': 0,
              'understanding_embodied_knowledge': 0, 'understanding_professional_knowledge': 0, 'understanding_relation': 0}
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


def eval_mma(result_file):

    result_file = [json.loads(line) for line in open(result_file)]
    pred_list = []
    label_list = []
    question_type_list = []
    question_id = []

    print(f"before process item number: {len(result_file)}")
    for result in result_file:
        text = result['text']
        question_type_list.append(result['question_type'])
        question_id.append(result['question_id'].split('/')[0]+'_'+result['question_id'].split('/')[1])
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
            pred_list.append('E') # default choice is E if no answer is found in the text.

        if result['groundtruth']=='A':
            label_list.append('A')
        elif result['groundtruth']=='B':
            label_list.append('B')
        elif result['groundtruth']=='C':
            label_list.append('C')
        elif result['groundtruth']=='D':
            label_list.append('D')

    print(f"after process pred number: {len(pred_list)}")
    print(f"after process label number: {len(label_list)}")
    print('================multi-choice================')
    calculate_mc(pred_list, label_list, question_type_list, question_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default="bunny-lora-phi-2-L2M-llava24k.jsonl")
    args = parser.parse_args()

    eval_mma(args.result_file)
