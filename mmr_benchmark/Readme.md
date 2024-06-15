# Evalutaion Guidelines

## Evaluation

### 1. Download the dataset
- `MMR-benchmark-images.zip` is the benchmark images.
- `MMR-benchmark.json` is the benchmark questions.

### 2. Run the inference code to get the answer results
- `evaluatoin` folder is the code to inference MMR-benchmark, including examples of 6 closed-source models and 12 open-source models.
- `evaluatoin/model_vqa_mmr.py` is the general inference code to get MMR-benchmark answers.
- `evaluatoin/calculation_mmr.py` is the general calculation code to calculate scores.
- `evaluatoin/mmr.sh` is a demo script to show how to use `model_vqa_mmr.py` and `calculation_mmr.py`.
- `evaluatoin/models/calculation.py` is the code to calculate scores of the answers file.
We provide 6 closed-source models and 12 open-source models inference code in `models` folder.

#### Closed-source models

For the closed-source models, please prepare your api keys, and run the inference code in `closed-source` folder.

#### Open-source models

For the open-source models, please download model cache first:

##### LLaVA

Follow the https://github.com/haotian-liu/LLaVA steps to install llava and download the model weights.

Put  `llava.py`  in the path `LLaVA/llava/eval/`.
```shell
python llava.py \
    --model-path /your_llava_model_path/ \
    --image-file /MMR-benchmark-images/ \
    --question-file /MMR-benchmark.json \
    --output-file /llava_result.json/
```

##### Deepseek-VL

Follow the https://github.com/deepseek-ai/DeepSeek-VL steps to install deepeek-vl and download the model weights.

Put `deepseek.py` in the `DeepSeek-VL/`.
```shell
python deepseek.py \
    --model-path /your_llava_model_path/ \
    --image-file /MMR-benchmark-images/ \
    --question-file /MMR-benchmark.json \
    --output-file /llava_result.json/
```

##### Mini-Gemini

Follow the https://github.com/dvlab-research/MGM steps to install deepeek-vl and download the model weights.

Put `mini-gemini.py` in the `MGM/mgm/eval/`.
```shell
python mini-gemini.py \
    --model-path /your_llava_model_path/ \
    --image-file /MMR-benchmark-images/ \
    --question-file /MMR-benchmark.json \
    --output-file /llava_result.json/
```

##### LLaVA-RLHF

Follow the https://github.com/llava-rlhf/LLaVA-RLHF steps to install deepeek-vl and download the model weights.

Put `llava-rlhf.py` in the `/LLaVA-RLHF/Eval/`.
```shell
python llava-rlhf.py \
    --model-path /your_llava_model_path/ \
    --image-file /MMR-benchmark-images/ \
    --question-file /MMR-benchmark.json \
    --output-file /llava_result.json/
```
##### Yi

Follow the https://github.com/01-ai/Yi/tree/main/VL steps to install deepeek-vl and download the model weights.

Put `yi.py` in the `/Yi/VL/`.
```shell
python yi.py \
    --model-path /your_llava_model_path/ \
    --image-file /MMR-benchmark-images/ \
    --question-file /MMR-benchmark.json \
    --output-file /llava_result.json/
```

##### Mplug-owl

Follow the https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2 steps to install deepeek-vl and download the model weights.

Put `mplug-owl.py` in the `/mPLUG-Owl/mPLUG-Owl2/`.
```shell
python mplug-owl.py \
    --model-path /your_llava_model_path/ \
    --image-file /MMR-benchmark-images/ \
    --question-file /MMR-benchmark.json \
    --output-file /llava_result.json/
```

##### Other open-source models

Install the following libraries.

```shell
numpy==1.24.4
Pillow==10.3.0
Requests==2.31.0
torch==2.3.0
torchvision==0.18.0
transformers==4.40.1
```

Then run the python file directly.

### 3. Get the score results.
Calculate scores of the answers file.
```shell
python evaluation/models/calculation.py \
    --result-file /llava_result.json/ \
    --groundtruth /MMR-benchmark.json/
```

