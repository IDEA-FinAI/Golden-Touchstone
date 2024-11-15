# 使用LLaMA-Factory微调Touchstone-GPT模型


## 安装LLaMA-Factory
下载并安装LLaMA-Factory：
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

安装完成后，执行`llamafactory-cli version`，若出现以下提示，则表明安装成功：
```
----------------------------------------------------------
| Welcome to LLaMA Factory, version 0.8.4.dev0           |
|                                                        |
| Project page: https://github.com/hiyouga/LLaMA-Factory |
----------------------------------------------------------
```

## 准备训练数据
自定义的训练数据应保存为jsonl文件，每一行的格式如下：
```json
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Tell me something about large language models."
        },
        {
            "role": "assistant",
            "content": "Large language models are a type of language model that is trained on a large corpus of text data. They are capable of generating human-like text and are used in a variety of natural language processing tasks..."
        }
      
    ]
}
```

在LLaMA-Factory文件夹下的`data/dataset_info.json`文件中注册自定义的训练数据，在文件尾部添加如下配置信息：
```
"alpaca_en_demo": {
    "file_name": "alpaca_en_demo.json"
  },
  "alpaca_zh_demo": {
    "file_name": "alpaca_zh_demo.json"
  },
```

## 配置训练参数
设置训练参数的配置文件，我们提供了全量参数、LoRA、QLoRA训练所对应的示例文件，你可以根据自身需求自行修改，配置详情见本目录下对应的文件:
- `Touchstone-GPT_sft_full.yaml`: 全量参数训练
- `Touchstone-GPT_sft_lora.yaml`: LoRA训练


## 开始训练

全量参数训练：
```bash
FORCE_TORCHRUN=1 llamafactory-cli train Touchstone-GPT_sft_full.yaml 
```

LoRA训练：
```bash
llamafactory-cli train Touchstone-GPT_sft_lora.yaml 
```


使用上述训练配置，各个方法实测的显存占用如下。训练中的显存占用与训练参数配置息息相关，可根据自身实际需求进行设置。
- 全量参数训练：42.18GB
- LoRA训练：20.17GB


## 合并模型权重
如果采用LoRA进行训练，脚本只保存对应的LoRA权重，需要合并权重才能进行推理。**全量参数训练无需执行此步骤**


```bash
llamafactory-cli export Touchstone-GPT_merge_lora.yaml
```



## 模型推理
训练完成，合并模型权重之后，即可加载完整的模型权重进行推理， 推理的示例脚本如下：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto
model_name_or_path = YOUR-MODEL-PATH

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```