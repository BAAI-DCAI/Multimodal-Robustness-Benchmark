import google.generativeai as genai
from PIL import Image
import json
from tqdm import tqdm
import time

# input your api key here
GOOGLE_API_KEY=''
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name = "models/gemini-pro-vision")

answer = []

def simple_multimodal_conversation_call(image, text, id_, type):
    """Simple single round multimodal conversation call.
    """
    img = Image.open(image)
    messages = [text+"\nAnswer with the option's letter from the given choices directly.", img]
    response = model.generate_content(messages)
    response.resolve()
    answer.append([id_, {'text': response.text}, type])
    print(response.text)

if __name__ == '__main__':
    with open('MMR-benchmark.json', 'r') as f:
        data = json.load(f)
    for item in tqdm(data):
        simple_multimodal_conversation_call(item['image'], item['question'], item['id'] ,item['question_type'])
        time.sleep(2)
    with open('gemini-MMR-result.json', 'w') as f:
        json.dump(answer, f)
