from http import HTTPStatus
import dashscope
import json
from tqdm import tqdm
import time


# export your dashscope api key in shell
answer = []

def simple_multimodal_conversation_call(image, text, id_, type):
    """Simple single round multimodal conversation call.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image},
                {"text": text+"\nAnswer with the option's letter from the given choices directly."}
            ]
        }
    ]
    
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max',
                                                     messages=messages)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        answer.append([id_, response.output.choices[0].message.content[0], type])
        print(response.output.choices[0].message.content[0])
    else:
        answer.append([id_, response.code, type])


if __name__ == '__main__':
    with open('question_mma_v5_bak.json', 'r') as f:
        data = json.load(f)
    for item in tqdm(data):
        simple_multimodal_conversation_call(item['image'], item['question'], item['id'] ,item['question_type'])
        time.sleep(5)
    with open('result/qwen_ni.json', 'w') as f:
        json.dump(answer, f)
