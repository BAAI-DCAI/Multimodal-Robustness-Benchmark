import argparse
import torch
from PIL import Image
from transformers import TextStreamer

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="/model_cache/llava-v1.6-34b")
parser.add_argument("--image-file", type=str, default='MMR-benchmark-images')
parser.add_argument("--question-file", type=str, default="MMR-benchmark.json")
parser.add_argument("--output-file", type=str, default="result/llava_result.json")

args = parser.parse_args()

model_path = args.model_path
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")


with open(args.question_file, 'r') as f:
    questions = json.load(f)

answer = []
for item in tqdm(questions):
    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles

    image_file = os.path.join(args.image_file, item['image'])
    query = item['question'] + "\nAnswer with the option's letter from the given choices directly."

    image = Image.open(image_file).convert('RGB')
    max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = DEFAULT_IMAGE_TOKEN + query
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    temperature = 0.7
    max_new_tokens = 512

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            # streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)
    answer.append([item['id'], {"text": outputs}, item['question_type']])

with open(args.output_file, 'w') as f:
    json.dump(answer, f)
