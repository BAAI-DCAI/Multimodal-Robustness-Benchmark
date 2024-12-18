import argparse
import torch
import torchvision.transforms as T
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from PIL import Image
from io import BytesIO
import re
import json
import os
import numpy as np
from tqdm import tqdm
import math
import cv2

##############################################Attention Visualization##############################################
def clean_sample_id(sample_id):
    return sample_id.replace("/", "_")

def process_tensor(tensor, positions):
    p1, p2, p3 = positions

    Attn_image = tensor[p3:, p1:p2]
    print("Attn_image.size",Attn_image.size())
    mask_tensor = Attn_image.mean(dim=0)

    side_length = int(math.sqrt(p2 - p1))
    mask_tensor_reshaped = mask_tensor.view(side_length, side_length)

    return mask_tensor_reshaped


def process_and_blend_map(attention_map, image_path, output_path, smooth_ksize=5, contrast_alpha=2.0, blend_alpha=0.2):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Check if the attention_map is valid and has content
    if attention_map.numel() == 0:
        raise ValueError("The attention map is empty.")

    # Resize attention_map to match the original image dimensions
    attention_map_np = attention_map.cpu().numpy()

    # Add a channel dimension
    attention_map_np = np.expand_dims(attention_map_np, axis=-1)
    min_val = np.min(attention_map_np)
    max_val = np.max(attention_map_np)
    normalized_map = (attention_map_np - min_val) / (max_val - min_val)

    attention_map_np = (normalized_map * 255).astype(np.uint8)

    target_width = int(original_image.shape[1])
    target_height = int(original_image.shape[0])

    resized_map = cv2.resize(attention_map_np, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    # Smooth, enhance, and blend the attention map with the original image
    smoothed_map = cv2.GaussianBlur(resized_map, (smooth_ksize, smooth_ksize), 0)
    enhanced_map = cv2.convertScaleAbs(smoothed_map, alpha=contrast_alpha, beta=0)
    enhanced_map_colored = cv2.applyColorMap(enhanced_map, cv2.COLORMAP_JET)
    blended_image = cv2.addWeighted(enhanced_map_colored, blend_alpha, original_image, 1 - blend_alpha, 0)

    cv2.imwrite(output_path, blended_image)

def process_and_blend_map_inverted(attention_map, image_path, output_path, smooth_ksize=5, contrast_alpha=2.0, blend_alpha=0.8):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not load image: {image_path}")

    if attention_map.numel() == 0:
        raise ValueError("The attention map is empty.")

    attention_map_np = attention_map.cpu().numpy()
    attention_map_np = np.expand_dims(attention_map_np, axis=-1)
    min_val = np.min(attention_map_np)
    max_val = np.max(attention_map_np)
    normalized_map = (attention_map_np - min_val) / (max_val - min_val)

    # 反转注意力图
    inverted_map = 1 - normalized_map
    attention_map_np = (inverted_map * 255).astype(np.uint8)

    target_width = int(original_image.shape[1])
    target_height = int(original_image.shape[0])
    resized_map = cv2.resize(attention_map_np, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    # 平滑、增强和混合图像
    smoothed_map = cv2.GaussianBlur(resized_map, (smooth_ksize, smooth_ksize), 0)
    enhanced_map = cv2.convertScaleAbs(smoothed_map, alpha=contrast_alpha, beta=0)
    enhanced_map_colored = cv2.applyColorMap(enhanced_map, cv2.COLORMAP_JET)

    # 使用更高的权重增强混合区域
    blended_image = cv2.addWeighted(original_image, blend_alpha, enhanced_map_colored, 1 - blend_alpha, 0)

    cv2.imwrite(output_path, blended_image)


##############################################Attention Score##############################################


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


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        with Image.open(BytesIO(response.content)) as img:
            return img.convert("RGB")
    else:
        # Ensure the file is closed properly
        with Image.open(image_file) as img:
            return img.convert("RGB")


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    prompt_info = """
Extract information from the image and structure the output as follows:

Text/Numbers: Identify any visible text or numbers, including their content, font, and precise positions in the image.
Objects: Identify each object, detailing:
Properties: name/category, color, texture, shape, and pose
Position: relative location within the image
Quantity: count of each type
People: Identify all visible individuals, with attributes such as: Position, Gender, Facial Expression
Identification: distinguish between real individuals and representations (e.g., in portraits, posters)
Relationships & Interactions:
Object Relationships: Describe spatial or contextual relationships between objects
People-Object Relationships: Describe interactions or spatial relationships between people and objects
Activities: Identify and describe any observed activities or actions involving people or objects
Events: Identify any significant events or scenes occurring in the image
"""

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name)


    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    
    with open(args.question_file, "r") as f:
        questions = json.load(f)

    answer = []
    for item in tqdm(questions):
        if args.prompt_type == "image":
            qs = item['question']
        else:
            qs = prompt_info

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if args.prompt_type == "image":
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_path = os.path.join(args.image_file, item['image'])
        image = [load_image(image_path)]
        image_sizes = [x.size for x in image]
        image_tensor = process_images(
            image,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                # output_attentions=True,
                # return_dict_in_generate=True,
            )

        if args.prompt_type == "info":
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            item["info"] = outputs
            print(outputs)
        
        if "image11111" in args.prompt_type:
            #################### visualization ############################
            attns = [list(attn) for attn in output_ids.attentions]
            p_before, p_after = prompt.split('<image>')
            p_before_tokens = tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(model.device).input_ids
            p_after_tokens = tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(model.device).input_ids
            p_before_tokens = tokenizer.convert_ids_to_tokens(p_before_tokens[0].tolist())
            p_after_tokens = tokenizer.convert_ids_to_tokens(p_after_tokens[0].tolist())

            bos = torch.ones([1, 1], dtype=torch.int64, device=model.device) * tokenizer.bos_token_id
            bos_tokens = tokenizer.convert_ids_to_tokens(bos[0].tolist())
            NUM_IMAGE_TOKENS = 576
            len1 = len(bos_tokens + p_before_tokens)
            len2 = len(bos_tokens + p_before_tokens + ['img_token'] * NUM_IMAGE_TOKENS)
            len3 = len(bos_tokens + p_before_tokens + ['img_token'] * NUM_IMAGE_TOKENS + p_after_tokens)-1
            position = [len1, len2, len3]
            print("position",position)

            for i in range(len(attns)):
                if i == 0:
                    last_attention_map = attns[0]
                    attn_map =  torch.stack(attns[0]).squeeze()
                    attn_map = torch.mean(attn_map[-1], dim=0)
                    attn_map = attn_map.cpu().numpy()
                else:
                    attention_map = attns[i]
                    current_attention_map = process_attention_maps(attention_map,last_attention_map)
                    last_attention_map = current_attention_map
                    current_attn_map = torch.stack(current_attention_map).squeeze()
                    current_attn_map = torch.mean(current_attn_map[-1], dim=0)
                    attn_map = current_attn_map.cpu().numpy()

                if i == len(attns)-1:
                    mask_tensor_reshaped=process_tensor(current_attn_map,position)
                    process_and_blend_map_inverted(mask_tensor_reshaped,image_path,os.path.join(args.output_image_folder, f"{clean_sample_id(item['id'])}_{item['question_type']}.png"))
                
            item["image_attn"]= os.path.join(args.output_image_folder, f"{clean_sample_id(item['id'])}_{item['question_type']}.png")
        answer.append(item)
        # assert False

    with open(args.output_file, "w") as f:
        json.dump(answer, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/share/liangzy/model_cache/llava-v1.6-vicuna-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default='/share/liangzy/mmr/hf_mmr/MMR-benchmark')
    parser.add_argument("--question-file", type=str, default="/share/liangzy/mmr/hf_mmr/MMR-benchmark/MMR-benchmark_modify.json")
    parser.add_argument("--output-file", type=str, default="/share/liangzy/mmr/Multimodal-Robustness-Benchmark/evaluation/evaluation/models/attention_score/statistic_results/MMR-benchmark_llava-v1.6-7b-text-info-new.json")
    parser.add_argument("--output-image-folder", type=str, default="/share/liangzy/mmr/Multimodal-Robustness-Benchmark/evaluation/evaluation/models/attention_score/mmr_attn_info")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prompt-type", type=str, default="info")
    args = parser.parse_args()

    eval_model(args)