import anthropic
import base64
import json
from tqdm import tqdm

answer = []
client = anthropic.Anthropic(
        # input your api key here
        api_key="",
    )

def read_image(path):
    with open(path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    return image_data

def simple_multimodal_conversation_call(image, text, id_, type):
    """Simple single round multimodal conversation call.
    """
    img = read_image(image)

    messages = [
                {
                    "type": "text",
                    "text": text+"\nAnswer with the option's letter from the given choices directly."
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": 'image/jpeg',
                        "data": img,
                    },
                }
            ]
    try:
        message = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                system="Select the option that best answers the question.",
                messages=[
                    {
                        "role": "user",
                        "content": messages
                    }
                ],
            )
        response = message.content[0].text
    except Exception as e:
        print(e)
        response = "E"
    answer.append([id_, {'text': response}, type])
    print(response)

if __name__ == '__main__':
    with open('MMR-benchmark.json', 'r') as f:
        data = json.load(f)
    for item in tqdm(data):
        simple_multimodal_conversation_call(item['image'], item['question'], item['id'] ,item['question_type'])
    with open('claude-MMR-result.json', 'w') as f:
        json.dump(answer, f)

