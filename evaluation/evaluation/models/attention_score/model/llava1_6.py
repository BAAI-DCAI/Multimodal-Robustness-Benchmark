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

##############################################Attention Visualization##############################################
def process_tensor(tensor, positions):
    p1, p2, p3 = positions

    Attn_image = tensor[p3:, p1:p2]
    print("Attn_image.size",Attn_image.size())
    mask_tensor = Attn_image.mean(dim=0)

    side_length = int(math.sqrt(p2 - p1))
    mask_tensor_reshaped = mask_tensor.view(side_length, side_length)

    return mask_tensor_reshaped


def process_and_blend_map(attention_map, image_path, output_path, smooth_ksize=5, contrast_alpha=2.0,
                          blend_alpha=0.5):
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

##############################################Attention Score##############################################

import numpy as np

def compute_row_col_statistics(lower_tri_matrix, positions):
    # Ensure positions are sorted, just in case they are not provided in order
    x1, x2, x3 = sorted(positions)

    # Step 1: Compute the mean along columns for each region
    region1_mean_col = np.max(lower_tri_matrix[x2:x3 , :x1], axis=0)
    region2_mean_col = np.max(lower_tri_matrix[x2:x3, x1:x2], axis=0)
    region3_mean_col = np.max(lower_tri_matrix[x2:x3, x2:x3], axis=0)
    region4_mean_col = np.max(lower_tri_matrix[x2:x3, x3:], axis=0)

    # Step 2: Calculate statistics on the column-averaged arrays
    mean1, min1, max1 = np.mean(region1_mean_col), np.min(region1_mean_col), np.max(region1_mean_col)
    mean2, min2, max2 = np.mean(region2_mean_col), np.min(region2_mean_col), np.max(region2_mean_col)
    mean3, min3, max3 = np.mean(region3_mean_col), np.min(region3_mean_col), np.max(region3_mean_col)
    mean4, min4, max4 = np.mean(region4_mean_col), np.min(region4_mean_col), np.max(region4_mean_col)

    return {
        "System token-only": {"mean": mean1, "min": min1, "max": max1},
        "Visual token-only": {"mean": mean2, "min": min2, "max": max2},
        "Question token-only": {"mean": mean3, "min": min3, "max": max3},
        "Answer token-only": {"mean": mean4, "min": min4, "max": max4},
    }


# def compute_row_col_statistics(lower_tri_matrix, positions):
#     # Ensure positions are sorted, just in case they are not provided in order
#     x1, x2, x3 = sorted(positions)

#     region1 = lower_tri_matrix[x2:x3-1, :x1 - 1]
#     mean1, min1, max1 = np.mean(region1), np.min(region1), np.max(region1)

#     region2 = lower_tri_matrix[x2:x3-1, x1:x2 - 1]
#     mean2, min2, max2 = np.mean(region2), np.min(region2), np.max(region2)

#     region3 = lower_tri_matrix[x2:x3-1, x2:x3 - 1]
#     mean3, min3, max3 = np.mean(region3), np.min(region3), np.max(region3)

#     region4 = lower_tri_matrix[x2:x3-1, x3:]
#     mean4, min4, max4 = np.mean(region4), np.min(region4), np.max(region4)

#     return {
#         "System token-only": {"mean": mean1, "min": min1, "max": max1},
#         "Visual token-only": {"mean": mean2, "min": min2, "max": max2},
#         "Question token-only": {"mean": mean3, "min": min3, "max": max3},
#         "Answer token-only": {"mean": mean4, "min": min4, "max": max4},
#     }


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
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, attn_implementation="eager")


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
        qs = item['question']
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
                output_attentions=True,
                return_dict_in_generate=True,
            )

        outputs = tokenizer.batch_decode(output_ids[0], skip_special_tokens=True)[0].strip()
        print(outputs)

        #################### visualization ############################
        attns = [list(attn) for attn in output_ids.attentions]
        print("size",len(attns))
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
                attn_map = torch.mean(attn_map[0], dim=0)
                print('current_attn_map',attn_map.size())
                attn_map = attn_map.cpu().numpy()
            else:
                attention_map = attns[i]
                current_attention_map = process_attention_maps(attention_map,last_attention_map)
                last_attention_map = current_attention_map
                current_attn_map = torch.stack(current_attention_map).squeeze()
                current_attn_map = torch.mean(current_attn_map[0], dim=0)
                print('current_attn_map',current_attn_map.size())
                attn_map = current_attn_map.cpu().numpy()

            if i == len(attns)-1:
            # if i == 1:
                statistics_row_col = compute_row_col_statistics(attn_map, position)
                print("statistics_row_col",statistics_row_col)
                # mask_tensor_reshaped=process_tensor(current_attn_map,position)
                # process_and_blend_map(mask_tensor_reshaped,image_path,os.path.join(args.output_image_folder, f"Blended_{i}_{clean_sample_id(item['id'])}.png"))
                # assert False

        answer.append([item['id'],
        {'text': outputs}, 
        item['question_type'],
        {'type': item['question_type']},
        {'statistics_row_col': str(statistics_row_col)}])

    with open(args.output_file, "w") as f:
        json.dump(answer, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/share/liangzy/model_cache/llava-v1.6-34b")
    parser.add_argument("--model-path", type=str, default="/share/liangzy/model_cache/llava-v1.6-vicuna-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default='/share/liangzy/mmr/hf_mmr/MMR-benchmark')
    parser.add_argument("--question-file", type=str, default="/share/liangzy/mmr/hf_mmr/MMR-benchmark/MMR-benchmark_modify.json")
    parser.add_argument("--output-file", type=str, default="/share/liangzy/mmr/Multimodal-Robustness-Benchmark/evaluation/evaluation/models/attention_score/statistic_results/llava-v1.6_7b_att0_question2sv_ok_max.json")
    parser.add_argument("--output-image-folder", type=str, default="/share/liangzy/mmr/Multimodal-Robustness-Benchmark/evaluation/evaluation/models/attention_score/attention_image")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
