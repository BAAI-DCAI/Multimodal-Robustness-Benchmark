from tqdm import tqdm
from openai import OpenAI
import json
import re
from multiprocessing import Process, Manager
import os
import base64
import argparse


def constructed_prompt(json):
    instruct = """
Please accurately and faithfully understand and execute the following instructions step by step.


STEP-1: (Do not output) Parse and extract information based on the Input JSON and 
Information Topics:
1. Determine if there is any very clear text or numbers (You must make sure the identification is right) in the image. If yes, extract the corresponding text or numbers along with its position.
2. Determine if there is any object in the image, including the properties of that object, such as name/category/color/texture/shape/pose (The properties of an object may not be unique, such as color, it may be a mixture of different colours), position, and quantity (You must make sure the identification is right).
3. Determine if there is any person in the image, including attributes such as name, position, gender, and facial expression (You must make sure the identification is right,You must differentiate between a portrait in a photo or poster and a real person, don't confuse them.).
4. Understand the events occurring in the image at a global level, and assess whether they relate to culture, emotions, or knowledge (You must make sure the identification is right).
5. Understand the relationships between objects within the image, such as relative positions or hierarchical relationships (You must make sure the identification is right).
6. Understand the relationships between people and objects, such as someone riding a bike or a person's actions (You must make sure the identification is right).
7. If there are two or more people present, extract activities they are engaged in.


STEP-2: (Do not output) Generate questions and corresponding correct answers based on the extracted information in STEP-1, Input Image and Question-Answer Types:
Question-Answer Types:
Types-1: (1) Ask if a certain object exists, the question is defined as ques-pos. (2) Operation-pos (3) Replace the object mentioned in the ques-pos with visually similar nonexistent object. Ask if the model identifies the replaced object, the question is defined as ques-neg. Rhetorical questions are preferred. (4) Operation-neg

Types-2: (1) Ask a question about the name/category/color/texture/shape/pose of an object in the image, the question is defined as ques-pos. (2) Operation-pos (3) Replace the name/category/color/texture/shape/pose of the object mentioned in the ques-pos. Ask a question about if the modified the name/category/color/texture/shape/pose of an object is correct. Rhetorical questions are preferred. The new question is defined as ques-neg. (4) Operation-neg

Types-3: (1) Ask a question about the number/position of an object in the image, the question is defined as ques-pos. (2) Operation-pos (3) Replace the number/position of the object mentioned in the ques-pos with a similar number/position as interference. Ask a question about if the modified number/position of the object is correct. Rhetorical questions are preferred. The new question is defined as ques-neg. (4) Operation-neg

Types-4: (1) Ask a question about the topic of abstract-knowledge/concrete- knowledge/professional-knowledge in the image, the question is defined as ques-pos. (2) Operation-pos (3) Replace the knowledge related to the topic mentioned in the ques-pos with similar knowledge as interference. Ask a distractor question about if the modified knowledge related to the topic is correct. Rhetorical questions are preferred. The new question is defined as ques-neg. (4) Operation-neg
- Candidate topics of abstract-knowledge: emotions, aesthetics, connotations, symbols, culture, news, allusions, legends, common sense, functions and so on.
- Candidate topics of concrete-knowledge: landmarks, celebrities, well-known objects and so on.
- Candidate topics of professional-knowledge: specific knowledge in various vertical fields, industries, disciplines and so on.

Types-5: (1) Ask a question about the activity of an object and/or interaction/relationship between objects or person in the image, the question is defined as ques-pos. (2) Operation-pos (3) Replace the activity/interaction/relationship mentioned in the ques-pos with similar activity/interaction/relationship as interference. Ask a distractor question about if the modified activity/interaction/relationship is correct. Rhetorical questions are preferred. The new question is defined as ques-neg. (4) Operation-neg

NOTE: All (2) Operation-pos is the same type, which is: Design and provide four options based on ques-pos, including opt-pos-0: right answer with the correct reason, opt-pos-1: right answer with an incorrect reason, opt-pos-2: wrong answer with the correct reason, opt-pos-3: wrong answer with an incorrect reason. The opt-pos-0 is defined as ans-pos. 
NOTE: All (4) Operation-neg is the same type, which is: Design and provide four options based on ques-neg, including opt-neg-0: right answer with the correct reason, opt-neg-1: right answer with an incorrect reason, opt-neg-2: wrong answer with the correct reason, opt-neg-3: wrong answer with an incorrect reason. The opt-neg-0 is defined as ans-neg. 
NOTE: The correct reason is the only one, while the incorrect reason can stem from various distortions and fabrications of facts, etc. 
NOTE: Pose questions that differ slightly from true understanding but are very similar, or add a distractor sentence to questions, so that the model is guided to make incorrect inferences or answers.
NOTE: For ques-neg, Rhetorical questions are preferred.


STEP-3: Organize and output JSON sample based on the generated questions, operations and answers in STEP-2, Output Requirements and Output JSON Format:
Output Requirements:
!!! ONLY output the JSON.
1. Ask 7-10 questions with answers for each of the pos and neg samples. The greater the diversity of different questions within the same type, the better. pos and neg questions need to correspond to each other.
2. Ask directly, without adding any information such as "in the conversation," "mentioned in the conversation," or "in the image."
3. Answer the question with explanation directly and succinctly. Do not exceeding 15 words.
4. Questions must be real and answerable. Answers must be correct, definitely, and not speculative.
5. Answers should NOT contain words such as "uncertain/unclear/not clear".
6. The format of the options list is: 'options': ['XXX.', 'XXX.', 'XXX.', 'XXX.']
7. The format of the generated ques-pos and ans-pos is: 'conversations-pos': [{'from': 'human', 'value': 'ques-pos'}, {'from': 'gpt', 'value': 'ans-pos'}, {'from': 'human', 'value': 'ques-pos'}, {'from': 'gpt', 'value': 'ans-pos'}, ..., {'from': 'human', 'value': 'ques-pos'}, {'from': 'gpt', 'value': 'ans-pos'}]
8. The format of the generated ques-neg and ans-neg is: 'conversations-neg': [{'from': 'human', 'value': 'ques-neg'}, {'from': 'gpt', 'value': 'ans-neg'}, {'from': 'human', 'value': 'ques-neg'}, {'from': 'gpt', 'value': 'ans-neg'}, ..., {'from': 'human', 'value': 'ques-neg'}, {'from': 'gpt', 'value': 'ans-neg'}]

Output JSON Format:
{'id': '{json['id']}', 'image': '{json['image']}', 'conversations-pos': [{'from': 'human', 'value': 'ques-pos', 'options': ['XXX.', 'XXX.', 'XXX.', 'XXX.']}, {'from': 'gpt', 'value': 'ans-pos'}, {'from': 'human', 'value': 'ques-pos','options': ['XXX.', 'XXX.', 'XXX.', 'XXX.']}, {'from': 'gpt', 'value': 'ans-pos'}, ..., {'from': 'human', 'value': 'ques-pos','options':['XXX.', 'XXX.', 'XXX.', 'XXX.']}, {'from': 'gpt', 'value': 'ans-pos'}], 'conversations-neg': [{'from': 'human', 'value': 'ques-neg','options': ['XXX.', 'XXX.', 'XXX.', 'XXX.']}, {'from': 'gpt', 'value': 'ans-neg'}, {'from': 'human', 'value': 'ques-neg','options': ['XXX.', 'XXX.', 'XXX.', 'XXX.']}, {'from': 'gpt', 'value': 'ans-neg'}, ..., {'from': 'human', 'value': 'ques-neg','options': ['XXX.', 'XXX.', 'XXX.', 'XXX.']}, {'from': 'gpt', 'value': 'ans-neg'}]}
    """
    prompt ="Input JSON:"+ "\n" + f"{json}" + instruct
    return prompt


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def generate_conversation(json_str,image_folder,api_key):
    json_dict = json.loads(json_str)
    image_path=os.path.join(image_folder, json_dict["image"])
    image_path= image_path.replace('\\', '/')
    base64_image = encode_image(image_path)

    chat_client = OpenAI(api_key=api_key)
    conversation_response = chat_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user",
             "content": [
                 {
                     "type": "image_url",
                     "image_url": {
                         "url": f"data:image/jpeg;base64,{base64_image}"
                     }
                 },
                 {
                     "type": "text",
                     "text": constructed_prompt(json_dict)
                 }
             ]
             }
        ]
    )
    return conversation_response.choices[0].message.content


def check_conversations(generated_sample, sample):
    if generated_sample['image'] != sample['image']:
        generated_sample['image'] = sample['image']
    if generated_sample['id'] != sample['id']:
        generated_sample['id'] = sample['id']

    conversations = generated_sample['conversations-pos']

    for i, conv in enumerate(conversations):
        if i % 2 == 0:
            if conv['from'] != 'human':
                assert False
        else:
            if conv['from'] != 'gpt':
                assert False

        if i > 0 and '<image>\n' in conv['value']:
            conv['value'] = conv['value'].replace('<image>\n', '')

        if not conv['value'].strip() and i != 0:
            assert False

    conversations = generated_sample['conversations-neg']

    for i, conv in enumerate(conversations):
        if i % 2 == 0:
            if conv['from'] != 'human':
                assert False
        else:
            if conv['from'] != 'gpt':
                assert False

        if i > 0 and '<image>\n' in conv['value']:
            conv['value'] = conv['value'].replace('<image>\n', '')

        if not conv['value'].strip() and i != 0:
            assert False


def process_json(data, beg, end,image_folder, output_file,api_key):
    for sample in tqdm(data[beg:end], desc="Processing samples", unit="sample"):
        json_str = json.dumps(sample)
        retry_count = 3
        while retry_count > 0:
            generated_json = generate_conversation(json_str,image_folder,api_key)
            pattern = r'```json(.*?)```'
            matches = re.search(pattern, generated_json, re.DOTALL)

            if matches:
                generated_json = matches.group(1).strip()
            try:
                generated_sample = json.loads(generated_json)
                check_conversations(generated_sample, sample)
                sample.update(generated_sample)
                with open(output_file, 'a', encoding='utf-8') as outfile:
                    json.dump(sample, outfile, ensure_ascii=False)
                    outfile.write('\n')
                break
            except json.JSONDecodeError:
                retry_count -= 1
                print(f"Failed to decode JSON after {retry_count} attempts.")
                if retry_count == 0:
                    print("Failed to decode JSON after 3 attempts, skipping sample.")
                    break


def check_output_file(output_file):
    if not os.path.isfile(output_file):
        return None

    existing_ids = set()
    with open(output_file, 'r', encoding='utf-8') as f:
        data =[]
        for line in f:
            sample = json.loads(line)
            data.append(sample)
        for sample in tqdm(data, desc="check samples", unit="sample"):
            existing_ids.add(sample['id'])
    return existing_ids


def find_min_remainder(x, start, end):
    min_remainder = x
    min_y = start
    for y in range(start, end):
        remainder = x % y
        if remainder < min_remainder:
            min_remainder = remainder
            min_y = y
    return min_y, min_remainder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process JSON files with multiprocessing.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the image folder.")
    parser.add_argument('--api_key', type=str, required=True, help="api_key.")
    args = parser.parse_args()

    existing_ids = check_output_file(args.output_file)
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    num_samples = len(data)

    if existing_ids is not None:
        data = [sample for sample in data if sample['id'] not in existing_ids]

    num_samples = len(data)
    start = 80
    end = 110
    min_y, min_remainder = find_min_remainder(num_samples, start, end)
    print("num_samples needed for process:", num_samples)
    print("num_samples per process:", min_y)
    num_process=min_y

    num_per_process = num_samples // num_process

    manager = Manager()
    result_list = manager.list()

    processes = []
    for idx in range(num_process):
        start_idx = idx * num_per_process
        end_idx = start_idx + num_per_process if idx < num_process - 1 else num_samples
        p = Process(target=process_json, args=(data, start_idx, end_idx,args.image_folder, args.output_file,args.api_key))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()