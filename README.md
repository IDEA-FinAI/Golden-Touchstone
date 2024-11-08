# Golden-Touchstone



Golden Touchstone is a simple, effective, and systematic benchmark for bilingual (Chinese-English) financial large language models, driving the research and implementation of financial large language models, akin to a touchstone. We also have trained and open-sourced Touchstone-GPT as a baseline for subsequent community research.

[Golden Touchstone Benchmark](https://huggingface.co/datasets/IDEA-FinAI/Golden-Touchstone)

![TouchStone-GPT-logo](https://github.com/IDEA-FinAI/Golden-Touchstone/blob/main/Touchstone-GPT-logo.png)
[Model Weight](https://huggingface.co/IDEA-FinAI/TouchstoneGPT-7B-Instruct/)


## Evalation
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