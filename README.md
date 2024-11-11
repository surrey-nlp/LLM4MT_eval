# What do large language models need for machine translaiton evaluation?

This repository contains the code and data for our paper "What do large language models need for machine translation evaluation?", which will be presented at the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP2024). The paper investigates the effectiveness of large language models (LLMs) in machine translation evaluation without training. It explores various prompting strategies and the types of information provided to LLMs across different language pairs and model architectures. For more details, please find our paper at [arXiv](https://arxiv.org/abs/2410.03278) or [here](https://aclanthology.org/2024.emnlp-main.214/). 

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

Check and change the model_type, template, model_name, lang_pair, and quantization variables in run.py, to select different templates, models and language pairs.

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

## Citation

Shenbin Qian, Archchana Sindhujan, Minnie Kabra, Diptesh Kanojia, Constantin Orasan, Tharindu Ranasinghe, and Fred Blain. 2024. What do Large Language Models Need for Machine Translation Evaluation?. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 3660–3674, Miami, Florida, USA. Association for Computational Linguistics.

## BibTex Citation

```
@inproceedings{qian-etal-2024-large,
    title = "What do Large Language Models Need for Machine Translation Evaluation?",
    author = "Qian, Shenbin  and
      Sindhujan, Archchana  and
      Kabra, Minnie  and
      Kanojia, Diptesh  and
      Orasan, Constantin  and
      Ranasinghe, Tharindu  and
      Blain, Fred",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.214",
    pages = "3660--3674",
    abstract = "Leveraging large language models (LLMs) for various natural language processing tasks has led to superlative claims about their performance. For the evaluation of machine translation (MT), existing research shows that LLMs are able to achieve results comparable to fine-tuned multilingual pre-trained language models. In this paper, we explore what translation information, such as the source, reference, translation errors and annotation guidelines, is needed for LLMs to evaluate MT quality. In addition, we investigate prompting techniques such as zero-shot, Chain of Thought (CoT) and few-shot prompting for eight language pairs covering high-, medium- and low-resource languages, leveraging varying LLM variants. Our findings indicate the importance of reference translations for an LLM-based evaluation. While larger models do not necessarily fare better, they tend to benefit more from CoT prompting, than smaller models. We also observe that LLMs do not always provide a numerical score when generating evaluations, which poses a question on their reliability for the task. Our work presents a comprehensive analysis for resource-constrained and training-less LLM-based evaluation of machine translation. We release the accrued prompt templates, code and data publicly for reproducibility.",
}
```


## Reference(s)

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP '23)*. Association for Computing Machinery, New York, NY, USA, 611–626. https://doi.org/10.1145/3600006.3613165

## Maintainer(s)

[Shenbin Qian](https://github.com/shenbinqian) \
[Archchana Sindhujan](https://www.surrey.ac.uk/people/archchana-sindhujan)
