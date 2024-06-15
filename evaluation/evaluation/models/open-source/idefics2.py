import requests
import torch
from PIL import Image
from io import BytesIO
import json
import os
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda"

# Note that passing the image urls (instead of the actual pil images) to the processor is also possible

processor = AutoProcessor.from_pretrained("/model_cache/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "/model_cache/idefics2-8b",
    torch_dtype=torch.float16,
).to(DEVICE)

with open('MMR-benchmark.json', 'r') as f:
    questions = json.load(f)

answer = []
for item in tqdm(questions):
    qs = item['question'] + "\nAnswer with the option's letter from the given choices directly."

    url = os.path.join('MMR-benchmark-images', item['image'])
    image = load_image(url)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": qs},
            ]
        }    
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}


    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": qs},
    #         ]
    #     }    
    # ]

    # prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    # inputs = processor(text=prompt, return_tensors="pt")
    # inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generate_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
    generated_texts = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

    print(generated_texts)
    answer.append([item['id'], {"text": generated_texts}, item['question_type']])

with open('result/idefics2_result.json', 'w') as f:
    json.dump(answer, f)
