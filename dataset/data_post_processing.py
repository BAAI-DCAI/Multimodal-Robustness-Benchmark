import json
from tqdm import tqdm
import argparse


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def remove_options(conversations):
    for conversation in conversations:
        if 'options' in conversation:
            del conversation['options']
    return conversations


def process_files(input_file, output_pos_file, output_neg_file):
    data = read_json_file(input_file)

    data_pos = []
    data_neg = []

    for item in data:
        base_item = {k: v for k, v in item.items() if
                     k not in ['conversations', 'conversations-neg', 'conversations-pos']}

        pos_item = base_item.copy()
        pos_item['conversations'] = remove_options(item.get('conversations-pos', []))
        data_pos.append(pos_item)

        neg_item = base_item.copy()
        neg_item['conversations'] = remove_options(item.get('conversations-neg', []))
        data_neg.append(neg_item)

    save_to_json(data_pos, output_pos_file)
    save_to_json(data_neg, output_neg_file)


def validate_and_filter_json(json_file_path):
    valid_samples = []

    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

            if not isinstance(data, list):
                print("The top-level element of the JSON file should be a list")
                return

            for item in data:
                if not isinstance(item, dict):
                    print(f"Error: {item} should be a dictionary")
                    continue

                required_keys = ["id", "image", "conversations"]
                if not all(key in item for key in required_keys):
                    print(f"Error: Missing keys in item {item}")
                    continue

                if not isinstance(item['conversations'], list):
                    print(f"Error: 'conversations' should be a list, item {item}")
                    continue

                conversations = item['conversations']
                if (len(conversations) % 2 != 0):
                    print(f"Error: 'conversations' should have an even number of items, item {item}")
                    continue

                valid = True
                for i, conversation in enumerate(conversations):
                    if not isinstance(conversation, dict):
                        print(f"Error: Conversation {conversation} should be a dictionary")
                        valid = False
                        break

                    if conversation['value'] is None:
                        print(f"Error: Conversation {conversation} value is empty")
                        valid = False
                        break

                    conversation_required_keys = ["from", "value"]
                    if not all(key in conversation for key in conversation_required_keys):
                        print(f"Error: Missing keys in conversation item {conversation}")
                        valid = False
                        break

                    if len(conversation.keys()) != 2:
                        print(f"Error: Conversation {conversation} should only contain 'from' and 'value' keys")
                        valid = False
                        break

                    if (i % 2 == 0 and conversation['from'] != "human") or (
                            i % 2 == 1 and conversation['from'] != "gpt"):
                        print(f"Error: The value of the 'from' key should alternate between 'human' and 'gpt' in sequence, item {conversation}")
                        valid = False
                        break

                    if i > 0 and '<image>' in conversation['value']:
                        conversation['value'] = conversation['value'].replace('<image>\n', '')
                        print(f"Error: Multiple image tokens in item {conversation}")
                        valid = False
                        break

                if valid:
                    valid_samples.append(item)

        with open(json_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(valid_samples, output_file, ensure_ascii=False, indent=4)
        print(f"Valid samples have been saved to {json_file_path}")
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")


def merge_json_files_list(input_file, output_file):
    all_data = []

    for file in input_file:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and validate JSON files.')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output_pos', type=str, required=True, help='Output positive JSON file path')
    parser.add_argument('--output_neg', type=str, required=True, help='Output negative JSON file path')
    parser.add_argument('--output_merge', type=str, required=True, help='Output negative JSON file path')
    args = parser.parse_args()

    # Reformat the JSON files
    process_files(args.input, args.output_pos, args.output_neg)

    # Validate and filter the JSON files
    validate_and_filter_json(args.output_pos)
    validate_and_filter_json(args.output_neg)

    # Merge the positive and negative JSON files
    merge_json_files([args.output_pos, args.output_neg], args.output_merge)