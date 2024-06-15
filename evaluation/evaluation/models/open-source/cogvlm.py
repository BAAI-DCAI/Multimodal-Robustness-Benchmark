"""
This is a demo for using CogVLM2 in CLI using multi-GPU with lower memory.
If your single GPU is not enough to drive this model, you can use this demo to run this model on multiple graphics cards with limited video memory.
Here, we default that your graphics card has 24GB of video memory, which is not enough to load the FP16 / BF16 model.
so , need to use two graphics cards to load. We set '23GiB' for each GPU to avoid out of memory.
GPUs less than 2 is recommended and need more than 16GB of video memory.

test success in 3 GPUs with 16GB video memory.
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    1   N/A  N/A   1890574      C   python                                    13066MiB |
|    2   N/A  N/A   1890574      C   python                                    14560MiB |
|    3   N/A  N/A   1890574      C   python                                    11164MiB |
+---------------------------------------------------------------------------------------+
"""
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

import json
import os
from tqdm import tqdm


MODEL_PATH = "/model_cache/cogvlm2-llama3-chat-19B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
    )

num_gpus = torch.cuda.device_count()
max_memory_per_gpu = "40GiB"
if num_gpus > 2:
    max_memory_per_gpu = f"{round(42 / num_gpus)}GiB"

device_map = infer_auto_device_map(
    model=model,
    max_memory={i: max_memory_per_gpu for i in range(num_gpus)},
    no_split_module_classes=["CogVLMDecoderLayer"]
)
model = load_checkpoint_and_dispatch(model, MODEL_PATH, device_map=device_map, dtype=TORCH_TYPE)
model = model.eval()

with open('MMR-benchmark.json', 'r') as f:
    questions = json.load(f)

answer = []
for item in tqdm(questions):
    qs = item['question'] + "\nAnswer with the option's letter from the given choices directly."
    url = os.path.join('MMR-benchmark-images', item['image'])
    image = Image.open(url).convert('RGB')

    # history = []

    input_by_model = model.build_conversation_input_ids(
        tokenizer,
        query=qs,
        # history=history,
        images=[image],
        template_version='chat'
    )
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("<|end_of_text|>")[0]
        print("\nCogVLM2:", response)
        answer.append([item['id'], {"text": response}, item['question_type']])

with open('result/cogvlm_result.json', 'w') as f:
    json.dump(answer, f)