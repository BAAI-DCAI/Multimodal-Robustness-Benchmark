from openai import OpenAI
import base64
from PIL import Image
import json
from tqdm import tqdm
import time

# input your api key here
client = OpenAI(api_key='', base_url="https://api.stepfun.com/v1")

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

answer = []

def simple_multimodal_conversation_call(image, text, id_, type):
    """Simple single round multimodal conversation call.
    """
    base64_image = encode_image(image)

    messages = text+"\nAnswer with the option's letter from the given choices directly."
    try:
      completion = client.chat.completions.create(
                  model="step-1v-32k",
                  messages=[
                      # {"role": "system",
                      #  "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                      {"role": "user",
                      "content": [
                      {
                        "type": "text",
                        "text": messages
                      },
                      {
                        "type": "image_url",
                        "image_url": {
                          "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                      }
                    ]
                      }
                  ]
              )
      print(completion.choices[0].message.content)
      answer.append([id_, {'text': completion.choices[0].message.content}, type])
    except Exception as e:
      print(id_)
      answer.append([id_, {'text': 'E'}, type])

if __name__ == '__main__':
    with open('MMR-benchmark.json', 'r') as f:
        data = json.load(f)
    for item in tqdm(data):
        simple_multimodal_conversation_call(item['image'], item['question'], item['id'] ,item['question_type'])
        time.sleep(10)
    with open('result/step_result.json', 'w') as f:
        json.dump(answer, f)


