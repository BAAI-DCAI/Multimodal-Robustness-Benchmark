import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

import json
import os 
from tqdm import tqdm

model = AutoModel.from_pretrained('/share/liangzy/model_cache/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('/share/liangzy/model_cache/MiniCPM-V-2_6', trust_remote_code=True)


with open('/share/liangzy/mmr/hf_mmr/MMR-benchmark/MMR-benchmark_modify.json', "r") as f:
        questions = json.load(f)

answer = []
for item in tqdm(questions):
    image_path = os.path.join('/share/liangzy/mmr/hf_mmr/MMR-benchmark', item['image'])
    image = Image.open(image_path).convert('RGB')


    qs = item['question']
    ins = "\nAnswer with the option's letter from the given choices directly.\n"
    question = ins + qs

    msgs = [{'role': 'user', 'content': [image, question]}]

    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    print(res)

    answer.append([item['id'], {'text': res}, item['question_type']])
with open('/share/liangzy/mmr/hf_mmr/MMR-benchmark/results/minicpm2.6_results.json', "w") as f:
    json.dump(answer, f)