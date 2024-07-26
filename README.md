# Japanese & English Medical Language Model Evaluation Harness

Evaluations for medical LLMs in one line.   
ワンコマンドで実行可能な医療分野に特化したLLMの評価プログラム.

# ⚠️ Notes: codes and results are still under development

## Leaderboard

without shuffle of choices
| Model |[IgakuQA](https://github.com/jungokasai/IgakuQA) | lang |
|---|---|---|
|[MedSwallow-70B](https://huggingface.co/AIgroup-CVM-utokyohospital/MedSwallow-70b) | 46.1 | en |
|[MedSwallow-70B](https://huggingface.co/AIgroup-CVM-utokyohospital/MedSwallow-70b) | 46.5 | ja |
|[OpenBioLLM-70B](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B)| 58.5 | en |
|[OpenBioLLM-70B](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B)| 35.6 | ja |
|[Swallow-70B](https://huggingface.co/tokyotech-llm/Swallow-70b-instruct-hf) | 32.3 | ja |
|[Swallow-70B](https://huggingface.co/tokyotech-llm/Swallow-70b-instruct-hf) | - | en |
|[Meditron-70B(Llama2-based)](https://huggingface.co/epfl-llm/meditron-70b) | 29.9 | en |
|[Med42-70B](https://huggingface.co/m42-health/med42-70b) | 45.0 | en |
|[Llama2-70B](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)| 26.0 | en |
|[Llama3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)| 38.3 | en |
|[Llama3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)| 43.1 | ja |
|[Llama3-70B w/o quantize](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)| 37.6 | en |
|[Llama3-70B w/o quantize](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)|  35.5 | ja |
|[OpenBioLLM-8B](https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B)| 39.6 | en |
|[OpenBioLLM-8B](https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B)| 30.9 | ja |
|[Swallow-7B](tokyotech-llm/Swallow-7b-instruct-hf) | 18.6 | ja |
|[Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)| 29.0 | en |
|[Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)| 22.1 | ja |
|[Youko-8B](https://huggingface.co/rinna/llama-3-youko-8b)| 22.5 | en |
|[Youko-8B](https://huggingface.co/rinna/llama-3-youko-8b)| 24.2 | ja |
|[Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct)| 45.9 | en |
|[Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct)| 41.7 | ja |
|[JMedLLM-v1-7B(ja)]()| 52.5 | ja |
|[JMedLLM-v1-7B(ja)]()| 50.2 | en |
|[JMedLLM-v1-7B(en)]()| 44.2 | ja |
|[JMedLLM-v1-7B(en)]()| 46.2 | en |


with shuffle of choices
| Model |[IgakuQA](https://github.com/jungokasai/IgakuQA) | lang |
|---|---|---|
|[MedSwallow-70B](https://huggingface.co/AIgroup-CVM-utokyohospital/MedSwallow-70b) | 45.5 | ja |
|[Meditron-70B](https://huggingface.co/epfl-llm/meditron-70b) | 29.7 | en |
|[Med42-70B](https://huggingface.co/m42-health/med42-70b) | 45.5 | en |


(*) The training data of MedSwallow is the Japanese-translated MedQA data, which also includes test split.

<details>
<summary>Settings in Leaderboard</summary>
- If there exist the instruct-version LLMs, we use them for our experiments.
- prompt : medpalm_five_choice_cot (en) / medpalm_five_choice_cot_ja (ja)
- all zero-shot
- quantize : True for 70B models, False for 7B models  
- metric : Accuracy based on Gestalt distance (relatively robust)   
- use_vllm : off
- environment : NVIDIA A100  
</details>



## Metric
- For multiple-choices Q&A : accuracy (based on gestalt distance) , listed above
- For open-ended Q&A : gestalt distance, human evaluation ...

---

## Setup
```
nvidia-smi #check GPU availability
git clone https://github.com/stardust-coder/japanese-lm-med-harness
cd japanese-lm-med-harness
pip install -r requirements.txt #use virtual environment if needed
cd dataset
git clone https://github.com/jungokasai/IgakuQA.git # prepare datasets you want to use (※)

...

cd ..
```

(※) Set each dataset from previous studies as follows.
```
dataset/
- IgakuQA/
    - baseline_results
    - data
        - 2018
        - 2019
        - 2020
        - 2021
        - 2022
- MedQA
    - usmleqa_en.jsonl
    - usmleqa_ja.jsonl
- MedMCQA
    - medmcqa_en.jsonl
    - medmcqa_ja.jsonl
- JMMLU #jsonl will be automatically generated during inference
    - anatomy.csv
    - clinical_knowledge.csv
    - college_medicine.csv
    - medical_genetics.csv
    - professional_medicine.csv
- ClinicalQA25
    - clinicalqa_en.jsonl
    - clinicalqa_ja.jsonl
```


## Usage

Results will be saved in jsonl format at `result/`.

Ex 1. (load model_path first and then load peft model)
```
python eval_bench.py \
--model_path tokyotech-llm/Swallow-70b-instruct-hf \
--peft AIgroup-CVM-utokyohospital/MedSwallow-70b \
--data IgakuQA \
--prompt alpaca_ja \
--lang ja \
--quantize 
```

Ex2. 
```
python eval_bench.py \
--model_path tokyotech-llm/Swallow-7b-instruct-hf \
--data MedMCQA \
--prompt medpalm_five_choice_cot_ja \
--lang ja
--use_vllm
```

Ex3.
```
python eval_bench.py \
--model_path epfl-llm/meditron-7b \
--data IgakuQA2018 \
--prompt meditron_five_choice \
--lang en
--use_vllm
```

Ex4.
```
python eval_bench.py \
--model_path BioMistral/BioMistral-7B \
--data IgakuQA2018 \
--prompt medpalm_five_choice_cot \
--lang en
--use_vllm
```

Test code
```
python eval_bench.py \
--model_path tokyotech-llm/Swallow-7b-instruct-hf \
--data sample \
--prompt alpaca_med_five_choice_cot_jp \
--lang ja
```


### parameters
model_path : huggingface model id
lang : "ja" or "en"  
prompt : see [template.py](./template.py) for options. Add manually if you need your own prompt template.
use_vllm : True or False
num_gpus : specify when using vllm, defaults to 1.
quantize : True or False. Better to quantize when using 70B LLM.   
shuffle : Whether to shuffle the choices.  
data : 
* "sample" ・・・ for code test
* "IgakuQA" (default) ・・・ Removed non-5-choice Q&As due to its format. 
    * "IgakuQA20{18,19,20,21,22}"
* "MedQA"
* "MedMCQA
 

## Evaluation and Metrics

### Evaluation Datasets
- ClinicalQA25 from [Almanac]() : 25 Open-ended text generation tasks.
- [IgakuQA]() : Japanese National Medical License Exam. 
- [MedMCQA]() : Multi-Subject Multi-Choice Dataset for Medical domain, we only use evaluation split.
- [MedQA]() : Americal National Medical License Exam, we only use evaluation split.
- [JMMLU]() : Japanese Massive Multitask Language Understanding Benchmark
- [CardioExam]() : Sample questions extracted from [the Japanese Circulation Society](https://www.j-circ.or.jp/specialist/sen_training/).

### Default Metric for Multiple-choices question-answering
When the choices are  
a.) hoge  
b.) fuga  
...,    
the response of the LLM is meant to be "fuga" rather than "b". This can be controlled via prompting to a certain extent.
- accuracy based on exact match
- accuracy based on gestalt match (default, see [Sukeda et al.](https://arxiv.org/abs/2310.10083))


### Notes
- Swallow-7b-instruct-hf, NVIDIA A10G x 1 => 20GB VRAM on GPU, 10 seconds/inference.
- Meditron-7b, NVIDIA A10G x 1 => 20GB VRAM on GPU, 3 minutes/inference.
- greedy sampling (do_sample=False, num_beams=1, temperature=0)
- vllm == 0.3.0 does not support Gemma and Apollo. vllm==0.3.2 does.


### Environment
<details>
<summary>ABCI</summary>
- module load python/3.10/3.10.14 cuda/12.1/12.1.1 cudnn/8.9/8.9.7
</details>

<details>
<summary>pip list</summary>
accelerate==0.28.0
aiohttp==3.9.3
aiosignal==1.3.1
annotated-types==0.6.0
anyio==4.3.0
async-timeout==4.0.3
attrs==23.2.0
bitsandbytes==0.43.0
certifi==2024.2.2
charset-normalizer==3.3.2
click==8.1.7
cloudpickle==3.0.0
cupy-cuda12x==12.1.0
datasets==2.18.0
dill==0.3.8
diskcache==5.6.3
exceptiongroup==1.2.0
fastapi==0.110.0
fastrlock==0.8.2
filelock==3.13.3
frozenlist==1.4.1
fsspec==2024.2.0
h11==0.14.0
httptools==0.6.1
huggingface-hub==0.22.1
idna==3.6
importlib_resources==6.4.0
interegular==0.3.3
Jinja2==3.1.3
joblib==1.3.2
jsonschema==4.21.1
jsonschema-specifications==2023.12.1
lark==1.1.9
Levenshtein==0.25.0
llvmlite==0.42.0
loralib==0.1.2
MarkupSafe==2.1.5
mpmath==1.3.0
msgpack==1.0.8
multidict==6.0.5
multiprocess==0.70.16
nest-asyncio==1.6.0
networkx==3.2.1
ninja==1.11.1.1
numba==0.59.1
numpy==1.26.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.18.1
nvidia-nvjitlink-cu12==12.4.99
nvidia-nvtx-cu12==12.1.105
outlines==0.0.37
packaging==24.0
pandas==2.2.1
peft==0.10.0
prometheus_client==0.20.0
protobuf==5.26.1
psutil==5.9.8
pyarrow==15.0.2
pyarrow-hotfix==0.6
pydantic==2.6.4
pydantic_core==2.16.3
pynvml==11.5.0
python-dateutil==2.9.0.post0
python-dotenv==1.0.1
python-liquid==1.12.1
pytz==2024.1
PyYAML==6.0.1
rapidfuzz==3.7.0
ray==2.10.0
referencing==0.34.0
regex==2023.12.25
requests==2.31.0
rpds-py==0.18.0
safetensors==0.4.2
scipy==1.12.0
sentencepiece==0.2.0
six==1.16.0
sniffio==1.3.1
starlette==0.36.3
sympy==1.12
tokenizers==0.15.2
torch==2.1.2
tqdm==4.66.2
transformers==4.39.1
triton==2.1.0
typing_extensions==4.10.0
tzdata==2024.1
urllib3==2.2.1
uvicorn==0.29.0
uvloop==0.19.0
vllm==0.3.3
watchfiles==0.21.0
websockets==12.0
xformers==0.0.23.post1
xxhash==3.4.1
yarl==1.9.4
</details>



### Acknowledgement / 謝辞

We thank all the data providers used in our work.  
This work was supported by AIST KAKUSEI project (FY2023).  
本研究は、国立研究開発法人産業技術総合研究所事業の令和5年度覚醒プロジェクトの助成を受けたものです。 


