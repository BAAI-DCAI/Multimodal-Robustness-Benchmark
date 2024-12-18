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

warnings.filterwarnings("ignore")
pretrained = "/share/liangzy/LLaVA-NeXT/checkpoints/llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-si_stage_am9"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

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

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    ins = "\nAnswer with the option's letter from the given choices directly.\n"
    question = DEFAULT_IMAGE_TOKEN + ins + qs
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    with torch.inference_mode():
        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        print(outputs)

    answer.append([item['id'], {'text': outputs}, item['question_type']])
with open('/share/liangzy/mmr/hf_mmr/MMR-benchmark/results/llavaone7b_results.json', "w") as f:
    json.dump(answer, f)


