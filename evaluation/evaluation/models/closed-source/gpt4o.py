from openai import OpenAI
import base64
from PIL import Image
import json
from tqdm import tqdm

# input your api key here
client = OpenAI(api_key='')

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
                  model="gpt-4o",
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
      print(e)
      print(id_)
      answer.append([id_, {'text': 'E'}, type])

if __name__ == '__main__':
    with open('MMR-benchmark.json', 'r') as f:
        data = json.load(f)
    for item in tqdm(data):
        simple_multimodal_conversation_call(item['image'], item['question'], item['id'] ,item['question_type'])
    with open('openai-MMR-result.json', 'w') as f:
        json.dump(answer, f)


