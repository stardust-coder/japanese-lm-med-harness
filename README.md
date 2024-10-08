# Japanese Medical Language Model Evaluation Harness
ワンコマンドで実行可能な医療分野に特化したLLMの日英能力評価プログラム.

## Leaderboard

w/o shuffle  
| Model |[IgakuQA](https://github.com/jungokasai/IgakuQA)| [MedQA](https://github.com/jind11/MedQA) | [MedMCQA](https://medmcqa.github.io) | lang |
|---|---|---|---|---|
|[Llama3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)| 38.3 | 57.7 | 38.8 | en |
|[Llama3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)| 43.1 | 40.9 | 37.2 | ja |
|[Llama3-70B w/o quantize](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)| 37.6 | 50.9 | 39.3 | en |
|[Llama3-70B w/o quantize](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)|  35.5 | 35.3 | 37.1 | ja |
|[MedSwallow-70B](https://huggingface.co/AIgroup-CVM-utokyohospital/MedSwallow-70b) | 46.1 | 71.5* | 45.8 | en |
|[MedSwallow-70B](https://huggingface.co/AIgroup-CVM-utokyohospital/MedSwallow-70b) | 46.5 | 79.3* | 39.2 | ja |
|[OpenBioLLM-70B](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B)| 58.5 | 70.2 | 65.0 | en |
|[OpenBioLLM-70B](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B)| 35.6 | 35.4 | 39.9 | ja |
|[Swallow-70B](https://huggingface.co/tokyotech-llm/Swallow-70b-instruct-hf) | 32.3 | 36.8 | 31.1 | ja |
|[Swallow-70B](https://huggingface.co/tokyotech-llm/Swallow-70b-instruct-hf) |  | 39.6 | 30.6 | en |
|[Meditron-70B](https://huggingface.co/epfl-llm/meditron-70b) | 29.9 | 44.7 | 32.8 | en |
|[Med42-70B](https://huggingface.co/m42-health/med42-70b) | 45.0 | 56.2 | 48.2 | en |
|[Llama2-70B](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)| 26.0 | 32.5 | 33.3 | en |
|---|---|---|---|---|
|[Swallow-7B](tokyotech-llm/Swallow-7b-instruct-hf) | 18.6 | 28.7 | 17.1 | ja |
|[Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)| 29.0 | 43.0 | 39.1 | en |
|[Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)| 22.1 | 30.4 | 31.2 | ja |
|[Youko-8B](https://huggingface.co/rinna/llama-3-youko-8b)| 22.5 | 34.1 | 29.4 | en |
|[Youko-8B](https://huggingface.co/rinna/llama-3-youko-8b)| 24.2 | 28.8 | 31.7 | ja |
|[Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct)| 46.4 | 36.9 | 34.7 | en |
|[Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct)| 44.6 | 30.8 | 31.5 | ja |



with shuffle  
| Model |[IgakuQA](https://github.com/jungokasai/IgakuQA)| [MedQA](https://github.com/jind11/MedQA) | [MedMCQA](https://medmcqa.github.io)| lang |
|---|---|---|---|---|
|[MedSwallow-70B](https://huggingface.co/AIgroup-CVM-utokyohospital/MedSwallow-70b) | 45.5 | 78.8* | 36.9 | ja |
|[Meditron-70B](https://huggingface.co/epfl-llm/meditron-70b) | 29.7 | 44.3 | 29.6 | en |
|[Med42-70B](https://huggingface.co/m42-health/med42-70b) | 45.5 | 54.6 | 47.4 | en |


(*) The training data of MedSwallow is the Japanese-translated MedQA data, which also includes test split.

<details>
<summary>Settings in Leaderboard</summary>

- prompt : medpalm_five_choice_cot / medpalm_five_choice_cot_ja, all zero-shot
- quantize : True for 70B models, False for 7B models  
- metric : Accuracy based on Gestalt distance (relatively robust)   
- use_vllm : off
- environment : NVIDIA A100  
</details>


---

## Setup
```
pip install -r requirements.txt
cd dataset
git clone https://github.com/jungokasai/IgakuQA.git
cd ..
```

Set each dataset as follows
```
dataset/
- IgakuQA/
    - baseline_results
    - data
        - 2018
        ...
        - 2022
- MedQA
    - usmleqa_en.jsonl
    - usmleqa_ja.jsonl
- MedMCQA
    - medmcqa_en.jsonl
    - medmcqa_ja.jsonl
- JMMLU
    - xxx.csv
- ClinicalQA25
    - clinicalqa_en.jsonl
    - clinicalqa_ja.jsonl
```


## Usage

Ex 1.
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

### Recommended models
- epfl-llm/meditron-7b
- BioMistral/BioMistral-7B
- FreedomIntelligence/Apollo-7B (not supported yet)
- tokyotech-llm/Swallow-7b-instruct-hf


### parameters
model_path : huggingface model id
lang : "ja" or "en"  
prompt : See [template.py](./template.py) for options. You can also add your own prompt template and use it.
use_vllm : True or False
num_gpus : Specify when using vllm, defaults to 1.
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
- ClinicalQA25 from [Almanac](https://ai.nejm.org/doi/full/10.1056/AIoa2300068) : 25 Open-ended text generation tasks.
- [IgakuQA](https://github.com/jungokasai/IgakuQA) : Japanese National Medical License Exam. 
- [MedMCQA](https://proceedings.mlr.press/v174/pal22a.html) : Multi-Subject Multi-Choice Dataset for Medical domain, we only use evaluation split.
- [MedQA](https://github.com/jind11/MedQA) : Americal National Medical License Exam, we only use evaluation split.

Japanese version of MedMCQA and MedQA were provided at [JMedBench](https://huggingface.co/datasets/Coldog2333/JMedBench) by Junfeng Jiang.


### For Multiple-choices question-answering
When the choices are  
a.) hoge  
b.) fuga  
...,    
the response of the LLM is meant to be "fuga" rather than "b". This can be controlled via prompting to a certain extent.
- accuracy based on exact match
- accuracy based on gestalt match


### Notes
- Swallow-7b-instruct-hf, NVIDIA A10G x 1 => 20GB VRAM on GPU, 10 seconds/inference.
- Meditron-7b, NVIDIA A10G x 1 => 20GB VRAM on GPU, 3 minutes/inference.
- greedy sampling (do_sample=False, num_beams=1, temperature=0)
- vllm == 0.3.0 does not support Gemma and Apollo. vllm==0.3.2 does.
- Under multi-gpu setting, when you run `eval_bench.py` with `--use_vllm`, you might face the error `RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method` (for example when using vllm==0.6.1.post2.) If so, please add the environmental variable with a line of code `os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"`.

### Environment
<details>
<summary>ABCI</summary>
`module load python/3.10/3.10.14 cuda/12.1/12.1.1 cudnn/8.9/8.9.7`
</details>

<details>
<summary>Library Environment (the result by `pip list`)</summary>
```
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
```
</details>



### Acknowledgement / 謝辞
This work was supported by AIST KAKUSEI project (FY2023).  
本研究は、国立研究開発法人産業技術総合研究所事業の令和5年度覚醒プロジェクトの助成を受けたものです。 

MedMCQA and MedQA were provided at [JMedBench](https://huggingface.co/datasets/Coldog2333/JMedBench) by Junfeng Jiang.

### How to cite

Please cite [our paper](https://arxiv.org/pdf/2409.11783) if you use this code!

```
@article{sukeda2024development,
  title={{Development and bilingual evaluation of Japanese medical large language model within reasonably low computational resources}},
  author={Sukeda, Issey},
  journal={arXiv preprint arXiv:2409.11783},
  year={2024},
}
```
