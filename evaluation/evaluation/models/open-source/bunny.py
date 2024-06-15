import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from tqdm import tqdm
import json
import os

# disable some warnings
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

# set device
device = 'cuda'  # or cpu
torch.set_default_device(device)

# create model
model = AutoModelForCausalLM.from_pretrained(
    '/model_cache/Bunny-Llama-3-8B-V',
    torch_dtype=torch.float16, # float32 for cpu
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    '/model_cache/Bunny-Llama-3-8B-V',
    trust_remote_code=True)

with open('MMR-benchmark.json', 'r') as f:
    questions = json.load(f)

answer = []
for item in tqdm(questions):
    qs = item['question'] + "\nAnswer with the option's letter from the given choices directly."
    text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{qs} ASSISTANT:"
    # text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {qs} ASSISTANT:"
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(device)
    # input_ids = torch.tensor(tokenizer(text).input_ids, dtype=torch.long).unsqueeze(0).to(device)

    url = os.path.join('MMR-benchmark-images', item['image'])
    image = Image.open(url)
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype, device=device)
    # generate
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=100,
        use_cache=True)[0]

    response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(response)
    answer.append([item['id'], {"text": response}, item['question_type']])

with open('result/bunny_result.json', 'w') as f:
    json.dump(answer, f)

