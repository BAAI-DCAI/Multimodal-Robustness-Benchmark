import argparse
import torch

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

import requests
from PIL import Image
from io import BytesIO
import re
import json
import os

from tqdm import tqdm

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    system_prompt = """
Please follow these instructions to accurately answer the questions based on the given image and question.

STEP-1 (DO NOT OUTPUT) Image Information Extraction
- Text/Numbers: Identify clear text or numbers in the image and their positions.
- Objects: Identify objects, their properties (name/category/color/texture/shape/pose), position, and quantity.
- People: Identify people, their attributes (name, position, gender, facial expression), and distinguish between real people and portraits in photos or posters.
- Events: Understand events in the image, and their cultural, emotional, or knowledge-related context.
- Object Relationships: Understand relationships between objects (relative positions or hierarchy).
- People-Object Relationships: Understand relationships between people and objects (e.g., someone riding a bike).
- Activities: If there are multiple people, identify their activities.

STEP-2 (DO NOT OUTPUT) Carefully review the image and the question to ensure accurate understanding of each question. """

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

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
        qs = "According to the information provided: " + item['info'] + "Answer the following question:" + '\n' + item['question']
        # qs = system_prompt + '\n' + item['question']
        # qs = item['question']
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
        # image_path = item['image_attn']
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
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)
        answer.append([item['id'], {'text': outputs}, item['question_type']])
    with open(args.output_file, "w") as f:
        json.dump(answer, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/share/liangzy/model_cache/llava-v1.6-vicuna-7b")
    parser.add_argument("--model-path", type=str, default="/share/liangzy/LLaVA/checkpoints/llava-v1.6-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default='/share/liangzy/mmr/hf_mmr/MMR-benchmark')
    parser.add_argument("--question-file", type=str, default="/share/liangzy/mmr/Multimodal-Robustness-Benchmark/evaluation/evaluation/models/attention_score/statistic_results/MMR-benchmark_llava-v1.6-7b-info.json")
    parser.add_argument("--output-file", type=str, default="/share/liangzy/mmr/Multimodal-Robustness-Benchmark/evaluation/evaluation/models/attention_score/statistic_results/llava-v1.6/results_llava-v1.6-7b-text-info.json")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
