from vllm import LLM
from vllm.sampling_params import SamplingParams
import json
import os
from tqdm import tqdm
import base64


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

model_name = "/share/liangzy/model_cache/Pixtral-12B-2409"

sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(model=model_name, tokenizer_mode="mistral")


with open('/share/liangzy/mmr/hf_mmr/MMR-benchmark/MMR-benchmark_modify.json', "r") as f:
        questions = json.load(f)

answer = []
for item in tqdm(questions):
    image_path = os.path.join('/share/liangzy/mmr/hf_mmr/MMR-benchmark', item['image'])
    image_url = encode_image_to_base64(image_path)

    qs = item['question']

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    ins = "\nAnswer with the option's letter from the given choices directly.\n"
    prompt = ins + qs

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]
        },
    ]

    outputs = llm.chat(messages, sampling_params=sampling_params)

    print(outputs[0].outputs[0].text)


    answer.append([item['id'], {'text': outputs[0].outputs[0].text}, item['question_type']])
with open('/share/liangzy/mmr/hf_mmr/MMR-benchmark/results/pixtral_results.json', "w") as f:
    json.dump(answer, f)