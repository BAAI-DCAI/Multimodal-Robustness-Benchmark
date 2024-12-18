# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings
from tqdm import tqdm
import json
import os
import numpy as np

warnings.filterwarnings("ignore")
pretrained = "/share/liangzy/model_cache/llava-onevision-qwen2-7b-si"
# pretrained = "/share/liangzy/model_cache/llava-onevision-qwen2-0.5b-si"
model_name = "llava_qwen"
device = "cuda:0"
# device_map = "auto"
device_map = "cuda:0"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
# model.config.image_aspect_ratio = "pad"
print("model.config",model.config)
model.eval()


with open('/share/liangzy/mmr/hf_mmr/MMR-benchmark/MMR-benchmark_modify.json', "r") as f:
        questions = json.load(f)

answer = []

for item in tqdm(questions):
    image_path = os.path.join('/share/liangzy/mmr/hf_mmr/MMR-benchmark', item['image'])
    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]


    qs = item['question']

    conv_template = "qwen_2"  # Make sure you use correct chat template for different models
    ins = "\nAnswer with the option's letter from the given choices directly.\n"
    question = DEFAULT_IMAGE_TOKEN + ins + qs
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            # output_attentions=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    outputs = tokenizer.batch_decode(output_ids[0], skip_special_tokens=True)[0].strip()
    print(outputs)

    logits = output_ids.scores

    final_logits = logits[0]
    
    for idx, choice_token_id in enumerate(answer_token_ids):
        token_id = choice_token_id[-1] if isinstance(choice_token_id, list) else choice_token_id
        token_logits = final_logits[0, token_id]
        print(f"Logits for choice {answer_choices[idx]} ({token_id}): {token_logits}")

    print([
        item['id'],
        {'text': outputs},
        item['question_type'],
        {'type': item['question_type']},
        {'logits': {
            'A': final_logits[0, answer_token_ids[0][-1]].item(),
            'B': final_logits[0, answer_token_ids[1][-1]].item(),
            'C': final_logits[0, answer_token_ids[2][-1]].item(),
            'D': final_logits[0, answer_token_ids[3][-1]].item(),
        }}
    ])
    answer.append([
        item['id'],
        {'text': outputs},
        item['question_type'],
        {'type': item['question_type']},
        {'logits': {
            'A': final_logits[0, answer_token_ids[0][-1]].item(),
            'B': final_logits[0, answer_token_ids[1][-1]].item(),
            'C': final_logits[0, answer_token_ids[2][-1]].item(),
            'D': final_logits[0, answer_token_ids[3][-1]].item(),
        }}
    ])

with open('/share/liangzy/mmr/Multimodal-Robustness-Benchmark/evaluation/evaluation/models/attention_score/statistic_results/llavaonevision-7b_results.json', "w") as f:
    json.dump(answer, f)