import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

import json
import os
from tqdm import tqdm

model_id = "/share/liangzy/model_cache/Llama-3.2-90B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)



with open('/share/liangzy/mmr/hf_mmr/MMR-benchmark/MMR-benchmark_modify.json', "r") as f:
        questions = json.load(f)

answer = []
for item in tqdm(questions):
    image_path = os.path.join('/share/liangzy/mmr/hf_mmr/MMR-benchmark', item['image'])
    image = Image.open(image_path)


    qs = item['question']
    ins = "\nAnswer with the option's letter from the given choices directly.\n"
    question = ins + qs
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=30)
    outputs = processor.decode(output[0])
    outputs = outputs.split('<|start_header_id|>assistant<|end_header_id|>')[-1]
    outputs = outputs.split('<|eot_id|>')[0]
    print(outputs)
    
    answer.append([item['id'], {'text': outputs}, item['question_type']])
with open('/share/liangzy/mmr/hf_mmr/MMR-benchmark/results/llama3.2_90_mmr_results.json', "w") as f:
    json.dump(answer, f)
