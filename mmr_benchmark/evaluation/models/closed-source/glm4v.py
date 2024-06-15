from http import HTTPStatus
import dashscope
import json
from tqdm import tqdm
import time
from zhipuai import ZhipuAI
import base64

# input your api key here
api = ""
answer = []

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def simple_multimodal_conversation_call(image, text, id_, type, client):
    img = encode_image(image)

    try:
        response = client.chat.completions.create(
            model="glm-4v",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                            "url": img
                            }
                        }
                    ]
                }
            ],
        )
        output = response.choices[0].message.content
    except Exception as e:
        output = "E"
    print(output)
    answer.append([id_, {"text": output}, type])

if __name__ == '__main__':
    client = ZhipuAI(api_key=api)
    with open('MMR-benchmark.json', 'r') as f:
        data = json.load(f)
    for item in tqdm(data):
        simple_multimodal_conversation_call(item['image'], item['question'], item['id'] ,item['question_type'], client)
    with open('result/glm_result.json', 'w') as f:
        json.dump(answer, f)
