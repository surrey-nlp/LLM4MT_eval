# What do large language models need for machine translaiton evaluation?

This repository contains the code and data for our paper "What do large language models need for machine translation evaluation?", which will be presented at the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP2024). The paper investigates the effectiveness of large language models (LLMs) in machine translation evaluation without training. It explores various prompting strategies and the types of information provided to LLMs across different language pairs and model architectures. For more details, please find our paper at arXiv. 

## Installation

```
git clone https://github.com/shenbinqian/LLM4MT_eval.git
cd LLM4MT_eval
conda create -n vllm python=3.9
conda activate vllm
pip install -r requirements.txt
```

## Inference LLMs with our pre-built prompts using vLLM

```
python run.py
```

Check and change the model_type, template, model_name, lang_pair, and quantization varibles in run.py, to select different templates, models and language pairs.

```
# select model type under the folder prompts
model_type = "./prompts/llama/"
template = "04"
# model name from HuggingFace
model_name = "meta-llama/Llama-2-7b-chat-hf"
lang_pair = "en-de"
# 'awq' for Mixtral and None for the rest in our paper
quantization = None
```

## Parse LLM outputs for evaluation

```
python output_parser.py
```

Check and change the template_version and subfolder variables in output_parser.py, to get the right path to your saved LLM output files.

```
# template + model name
template_version = "03-mixtral"
# folder name where outputs stored
subfolder = "./llm_output_samples/"
```

## Build prompts with other data and formats

```
python prompt_building.py --prompt_format gemma_format --main_file raw_data/en-de/en-de_overlaps_dev.tsv
```

You can also add or change the format for different LLMs in prompt_building.py.

## Reference(s)

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP '23)*. Association for Computing Machinery, New York, NY, USA, 611–626. https://doi.org/10.1145/3600006.3613165

## Maintainer(s)

[Shenbin Qian](https://github.com/shenbinqian) \
[Archchana Sindhujan](https://www.surrey.ac.uk/people/archchana-sindhujan)