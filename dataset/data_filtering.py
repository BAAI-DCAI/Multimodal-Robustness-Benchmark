import json
import argparse
from tqdm import tqdm


def contains_invalid_word(text, invalid_words):
    return any(word in text for word in invalid_words)


def filter_conversations(json_file, output_file, invalid_words):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = []
    for sample in tqdm(data, desc="Filtering conversations"):
        filtered_conversations = []
        skip_next = False  # Flag to skip the next conversation if current one is to be filtered

        conversations = sample['conversations']
        for i, conversation in enumerate(conversations):
            if skip_next:
                skip_next = False
                continue

            from_human = conversation['from'] == 'human'
            from_gpt = conversation['from'] == 'gpt'

            if from_human:
                gpt_value = conversations[i + 1]['value'] if i + 1 < len(conversations) else ""
                if contains_invalid_word(gpt_value, invalid_words):
                    skip_next = True
                    continue

            if from_gpt:
                if contains_invalid_word(conversation['value'], invalid_words):
                    continue

            filtered_conversations.append(conversation)

        sample['conversations'] = filtered_conversations
        filtered_data.append(sample)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter conversations in a JSON file based on invalid words.')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file path')
    args = parser.parse_args()

    invalid_words = [
        "sorry", "not provided", "is not visible", "Based on the data provided", "I cannot",
        "I am unable to see", "I can't provide", "I can't determine",
        "I can't specify", "I cannot directly see", "I can't identify", "I can't confirm", "I can't tell",
        "I can't confirm", "I can't verify", "The question can't be answered", "I can't ascertain",
        "I can't visually describe", "Apologies, but", "I can't visually confirm", "I can't accurately answer",
        "not discernible", "is not specified in the image.", "I am unable to see", "described in the image",
        "available in the text or image", "given in the image", "can be found in the image", "I'm unable"
    ]

    filter_conversations(args.input, args.output, invalid_words)
