# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import json
import os
from tqdm import tqdm

model = AutoModel.from_pretrained('/model_cache/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('/model_cache/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

with open('MMR-benchmark.json', 'r') as f:
    questions = json.load(f)


answer = []
for item in tqdm(questions):
    qs = item['question'] + "\nAnswer with the option's letter from the given choices directly."
    url = os.path.join('MMR-benchmark-images', item['image'])
    image = Image.open(url).convert('RGB')

    msgs = [{'role': 'user', 'content': qs}]

    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True, # if sampling=False, beam_search will be used by default
        temperature=0.7
        # system_prompt='' # pass system_prompt if needed
    )
    print(res)
    answer.append([item['id'], {"text": res}, item['question_type']])

with open('result/minicpm_result.json', 'w') as f:
    json.dump(answer, f)
## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
# res = model.chat(
#     image=image,
#     msgs=msgs,
#     tokenizer=tokenizer,
#     sampling=True,
#     temperature=0.7,
#     stream=True
# )

# generated_text = ""
# for new_text in res:
#     generated_text += new_text
#     print(new_text, flush=True, end='')
