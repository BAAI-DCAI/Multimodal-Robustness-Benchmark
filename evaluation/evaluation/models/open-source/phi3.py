from PIL import Image 
import os
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
import json
from tqdm import tqdm

image_token = "<|image_1|>\n"
user_prompt = '<|user|>\n'
assistant_prompt = '<|assistant|>\n'
prompt_suffix = "<|end|>\n"

model_id = "/model_cache/Phi-3-vision-128k-instruct" 

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto")

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

with open('MMR-benchmark.json', 'r') as f:
    questions = json.load(f)

answer = []
for item in tqdm(questions):
    qs = item['question'] + "\nAnswer with the option's letter from the given choices directly."
    prompt = f"{user_prompt}{image_token}{qs}{prompt_suffix}{assistant_prompt}"

    url = os.path.join('MMR-benchmark-images', item['image'])
    image = Image.open(url) 

    inputs = processor(prompt, image, return_tensors="pt").to("cuda")

    generation_args = { 
        "max_new_tokens": 500, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

    print(response)
    answer.append([item['id'], {"text": response}, item['question_type']])

with open('result/phi3_result.json', 'w') as f:
    json.dump(answer, f)
