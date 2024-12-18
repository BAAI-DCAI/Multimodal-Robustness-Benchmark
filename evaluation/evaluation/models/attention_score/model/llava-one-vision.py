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
pretrained = "/share/liangzy/model_cache/llava-onevision-qwen2-0.5b-si"
# pretrained = "/share/liangzy/model_cache/llava-onevision-qwen2-0.5b-si"
model_name = "llava_qwen"
device = "cuda:0"
# device_map = "auto"
device_map = "cuda:0"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map,attn_implementation="eager")
# model.config.image_aspect_ratio = "pad"
print("model.config",model.config)
model.eval()

def compute_statistics(lower_tri_matrix, positions):
    x1, x2, x3 = sorted(positions)

    region1 = lower_tri_matrix[:, :x1 + 1]
    mean1, min1, max1 = np.mean(region1), np.min(region1), np.max(region1)

    region2 = lower_tri_matrix[:, x1:x2 + 1]
    mean2, min2, max2 = np.mean(region2), np.min(region2), np.max(region2)

    region3 = lower_tri_matrix[:, x2:x3 + 1]
    mean3, min3, max3 = np.mean(region3), np.min(region3), np.max(region3)

    region4 = lower_tri_matrix[:, x3:]
    mean4, min4, max4 = np.mean(region4), np.min(region4), np.max(region4)

    return {
        "System token": {"mean": mean1, "min": min1, "max": max1},
        "Visual token": {"mean": mean2, "min": min2, "max": max2},
        "Question token": {"mean": mean3, "min": min3, "max": max3},
        "Answer token": {"mean": mean4, "min": min4, "max": max4},
    }


def compute_row_col_statistics(lower_tri_matrix, positions):
    # Ensure positions are sorted, just in case they are not provided in order
    x1, x2, x3 = sorted(positions)

    # 1. 从行[x3,:] 列[0,x1] 获取下三角矩阵对应部分
    region1 = lower_tri_matrix[x3:, :x1 + 1]
    mean1, min1, max1 = np.mean(region1), np.min(region1), np.max(region1)

    # 2. 从行[x3,:] 列[x1,x2] 获取下三角矩阵对应部分
    region2 = lower_tri_matrix[x3:, x1:x2 + 1]
    mean2, min2, max2 = np.mean(region2), np.min(region2), np.max(region2)

    # 3. 从行[x3,:] 列[x2,x3] 获取下三角矩阵对应部分
    region3 = lower_tri_matrix[x3:, x2:x3 + 1]
    mean3, min3, max3 = np.mean(region3), np.min(region3), np.max(region3)

    # 4. 从行[x3,:] 列[x3,:] 获取下三角矩阵对应部分
    region4 = lower_tri_matrix[x3:, x3:]
    mean4, min4, max4 = np.mean(region4), np.min(region4), np.max(region4)

    return {
        "System token-only": {"mean": mean1, "min": min1, "max": max1},
        "Visual token-only": {"mean": mean2, "min": min2, "max": max2},
        "Question token-only": {"mean": mean3, "min": min3, "max": max3},
        "Answer token-only": {"mean": mean4, "min": min4, "max": max4},
    }


def combine_attention_maps(attention_map, last_attention_map):
    padding = (0, 1, 0, 1)
    new_attention_map = torch.nn.functional.pad(last_attention_map, padding)
    new_attention_map[:, :, -1, :] = attention_map[:, :, -1, :]
    return new_attention_map


def process_attention_maps(attention_maps_list, last_attention_maps_list):
    new_attention_maps_list = []

    for attention_map, last_attention_map in zip(attention_maps_list, last_attention_maps_list):
        if not isinstance(attention_map, torch.Tensor):
            attention_map = torch.tensor(attention_map)
        if not isinstance(last_attention_map, torch.Tensor):
            last_attention_map = torch.tensor(last_attention_map)
        new_attention_map = combine_attention_maps(attention_map, last_attention_map)
        new_attention_maps_list.append(new_attention_map)

    return new_attention_maps_list


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
            output_attentions=True,
            max_new_tokens=4096,
            return_dict_in_generate=True,
        )

    del image_tensor
    torch.cuda.empty_cache()
    
    print("model.config.image_feature_dim",model.config.image_feature_dim)
    print("answer",output_ids[0].size())

    #################### visualization ############################
    attns = [list(attn) for attn in output_ids.attentions]
    print("size",len(attns))
    p_before, p_after = prompt_question.split('<image>')
    print("p_before",p_before)
    print("p_after",p_after)
    p_before_tokens = tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(model.device).input_ids
    p_after_tokens = tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(model.device).input_ids
    p_before_tokens = tokenizer.convert_ids_to_tokens(p_before_tokens[0].tolist())
    p_after_tokens = tokenizer.convert_ids_to_tokens(p_after_tokens[0].tolist())

    NUM_IMAGE_TOKENS = model.config.image_feature_dim
    tokens = p_before_tokens + ['img_token'] * NUM_IMAGE_TOKENS + p_after_tokens

    len1 = len(p_before_tokens)
    len2 = len(p_before_tokens + ['img_token'] * NUM_IMAGE_TOKENS)
    len3 = len(p_before_tokens + ['img_token'] * NUM_IMAGE_TOKENS + p_after_tokens)
    position = [len1, len2, len3]
    print("position",position)

    for i in range(len(attns)):
        if i == 0:
            last_attention_map = attns[0]
            attn_map =  torch.stack(attns[0]).squeeze()
            attn_map = torch.mean(attn_map[-1], dim=0)
            print('current_attn_map',attn_map.size())
            attn_map = attn_map.cpu().numpy()
        else:
            attention_map = attns[i]
            current_attention_map = process_attention_maps(attention_map,last_attention_map)
            last_attention_map = current_attention_map

            current_attn_map = None
            for map_tensor in current_attention_map:
                map_tensor = map_tensor.cpu()  # 移到 CPU
                if current_attn_map is None:
                    current_attn_map = map_tensor.unsqueeze(0)  # 初始化
                else:
                    current_attn_map = torch.cat((current_attn_map, map_tensor.unsqueeze(0)), dim=0)  # 拼接
            current_attn_map = torch.mean(current_attn_map[-1], dim=0)
            print('current_attn_map',current_attn_map.size())
            attn_map = current_attn_map.cpu().numpy()

        if i == len(attns)-1:
            statistics_row_col = compute_row_col_statistics(attn_map, position)
            print("statistics_row_col",statistics_row_col)


    answer.append([item['id'],
     {'text': outputs}, 
     item['question_type'],
     {'type': item['question_type']},
     {'statistics_row_col': statistics_row_col}])

with open('/share/liangzy/mmr/Multimodal-Robustness-Benchmark/evaluation/evaluation/models/attention_score/statistic_results/llavaonevision-0.5b_results.json', "w") as f:
    json.dump(answer, f)