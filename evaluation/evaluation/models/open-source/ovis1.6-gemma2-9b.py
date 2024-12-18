import torch
from PIL import Image
from transformers import AutoModelForCausalLM

import os
import json
from tqdm import tqdm

# load model
model = AutoModelForCausalLM.from_pretrained("/share/liangzy/model_cache/Ovis1.6-Gemma2-9B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=8192,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()


with open('/share/liangzy/mmr/hf_mmr/MMR-benchmark/MMR-benchmark_modify.json', "r") as f:
        questions = json.load(f)

answer = []
for item in tqdm(questions):
    image_path = os.path.join('/share/liangzy/mmr/hf_mmr/MMR-benchmark', item['image'])
    image = Image.open(image_path)


    qs = item['question']
    ins = "<image>\nAnswer with the option's letter from the given choices directly.\n"
    query = ins + qs

    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

    # generate output
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f'Output:\n{output}')

    answer.append([item['id'], {'text': output}, item['question_type']])
with open('/share/liangzy/mmr/hf_mmr/MMR-benchmark/results/ovis_results.json', "w") as f:
    json.dump(answer, f)