# Golden-Touchstone


Golden Touchstone is a simple, effective, and systematic benchmark for bilingual (Chinese-English) financial large language models, driving the research and implementation of financial large language models, akin to a touchstone. We also have trained and open-sourced Touchstone-GPT as a baseline for subsequent community research.



[Golden Touchstone Benchmark](https://huggingface.co/datasets/IDEA-FinAI/Golden-Touchstone)



[![Model Weight](https://github.com/IDEA-FinAI/Golden-Touchstone/blob/main/Touchstone-GPT-logo.png)](https://huggingface.co/IDEA-FinAI/TouchstoneGPT-7B-Instruct/)

## Introduction
Below is the information of the open source benchmarks cited in this work
|  Date  | Name |      Author       |    Institute    | Links  | Paper
| :-----: | :-----: | :------------------: | :--------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | 
| 2023-05~2024-03 | Xuanyuan 1&2 <br/>(轩辕) | Zhang X, Yang Q.  | Duxiaoman Co., China<br/>(度小满) |[![arXiv](https://img.shields.io/badge/Arxiv-2305.12002-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2305.12002)[![github](https://img.shields.io/github/stars/Duxiaoman-DI/XuanYuan.svg?style=social)](https://github.com/Duxiaoman-DI/XuanYuan)[![huggingface](https://img.shields.io/badge/🤗-Model%206B~70B-yellow.svg)](https://huggingface.co/Duxiaoman-DI/XuanYuan2-70B-Chat) |XuanYuan 2.0: A Large Chinese Financial Chat Model with Hundreds of Billions Parameters |
| 2023-11 | CFBenchmark | Yang Lei, Li J, et al. | TongjiFinLab & Shanghai AI Lab, China<br/>(同济大学网络金融安全协同创新中心与上海人工智能实验室联合团队) | [![arXiv](https://img.shields.io/badge/Arxiv-2311.05812-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2311.05812)[![github](https://img.shields.io/github/stars/TongjiFinLab/CFBenchmark.svg?style=social)](https://github.com/TongjiFinLab/CFBenchmark)[![huggingface](https://img.shields.io/badge/🤗-Model%207B-yellow.svg)](https://huggingface.co/TongjiFinLab/CFGPT1-sft-7B-Full) | CFBenchmark: Chinese Financial Assistant Benchmark for Large Language Model |
| 2023-08 | FinEval | Liwen Zhang, et al. | Shanghai University of Finance and Economics | [![arXiv](https://img.shields.io/badge/Arxiv-2308.09975-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2308.09975)[![github](https://img.shields.io/github/stars/SUFE-AIFLM-Lab/FinEval.svg?style=social)](https://github.com/SUFE-AIFLM-Lab/FinEval)![huggingface](https://img.shields.io/badge/🤗-None-yellow.svg) | FinEval: A Chinese Financial Domain Knowledge Evaluation Benchmark for Large Language Models |
| 2023-06 | Pixiu Finma | Xie Q, Han W, Zhang X, et al.  | The FinAI & ChanceFocus Co. & Wuhan University, China<br/>(武汉大学等) | [![arXiv](https://img.shields.io/badge/Arxiv-2306.05443-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2306.05443)[![github](https://img.shields.io/github/stars/The-FinAI/PIXIU.svg?style=social)](https://github.com/The-FinAI/PIXIU)[![huggingface](https://img.shields.io/badge/🤗-Model%207B-yellow.svg)](https://huggingface.co/ChanceFocus/finma-7b-full) | Pixiu: A large language model, instruction data and evaluation benchmark for finance | 
| 2023-06 | FinGPT | Yang H, Liu X Y, Wang C D. | AI4Finance-Foundation | [![arXiv](https://img.shields.io/badge/Arxiv-2306.06031-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2306.06031)[![github](https://img.shields.io/github/stars/AI4Finance-Foundation/FinGPT.svg?style=social)](https://github.com/AI4Finance-Foundation/FinGPT)[![huggingface](https://img.shields.io/badge/🤗-Model%207B%20lora-yellow.svg)](https://huggingface.co/FinGPT/fingpt-mt_llama2-7b_lora) | Fingpt: Open-source financial large language models |
| 2023-02 | BBT-Benchmark | Dakuan Lu, Hengkui Wu, et al. | Shanghai Key Laboratory of Data Science | [![arXiv](https://img.shields.io/badge/Arxiv-2302.09432-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2302.09432)[![github](https://img.shields.io/github/stars/ssymmetry/BBT-FinCUGE-Applications.svg?style=social)](https://github.com/ssymmetry/BBT-FinCUGE-Applications)![huggingface](https://img.shields.io/badge/🤗-None-yellow.svg) | BBT-Fin: Comprehensive Construction of Chinese Financial Domain Pre-trained Language Model, Corpus and Benchmark |
| 2024 | IDEA-FinLLM | coming soon | coming soon | coming soon | coming soon |

The paper shows the evaluation of the diversity, systematicness and LLM adaptability of each open source benchmark.

![benchmark_info](https://github.com/IDEA-FinAI/Golden-Touchstone/blob/main/benchmark_info.png)

By collecting and selecting representative task datasets, we built our own Chinese-English bilingual Touchstone Benchmark, which includes 22 datasets

![golden_touchstone_info](https://github.com/IDEA-FinAI/Golden-Touchstone/blob/main/golden_touchstone_info.png)

We extensively evaluated GPT-4o, llama3, qwen2, fingpt and our own trained Touchstone-GPT, analyzed the advantages and disadvantages of these models, and provided direction for subsequent research on financial large language models

![evaluation](https://github.com/IDEA-FinAI/Golden-Touchstone/blob/main/evaluation.png)

## Evaluation
### Quick Inference Use
Our inference is based on the llama-factory framework, and eval_benchmark.sh is our reasoning script. Register the template and dataset in llama-factory, and download the specified open source model before you can use it.

```code
bash eval_benchmark.sh
```

All files of our llama-factory framework will be uploaded later
### Quick Eval Use
evaluate_all.py is an evaluation program based on the file generated by llama-factory reasoning, which contains three main parameters: 

__Model__, __eval_dataset_path__, __output_dir__

__Model__ specifies the model you use, which is embedded in the two file paths of eval_dataset_path and output_dir.

__eval_dataset_path__ indicates the file path generated after the llama-factory framework completes reasoning, which should contain the output folder of each data set

__output_dir__ indicates the path of all the data set task results you want to output, and the output result is in json format

After specifying these three address variables, use
```python
python evaluate_all.py
```
to find all the evaluation results in the output_dir