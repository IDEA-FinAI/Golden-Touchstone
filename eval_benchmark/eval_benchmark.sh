#!/bin/bash

# 模型列表
models=(
    /finance_ML/ljx_data/Qwen2-7B-Instruct
    /finance_ML/ljx_data/Meta-Llama-3-8B-Instruct
    /finance_ML/ljx_data/Fingpt-8B-lora
    /finance_ML/wuxiaojun/pretrained/FinLLM/CFGPT1-sft-7B-Full
    /finance_ML/ljx_data/finma-7b-full
    /finance_ML/ljx_data/finma-7b-nlp
    /finance_ML/ljx_data/finma-7b-trade
)
# 声明一个关联数组
declare -A my_dict
# 向字典中添加键值对
my_dict[Qwen2-7B-Instruct]="qwen"
my_dict[Meta-Llama-3-8B-Instruct]="llama3"
my_dict[Fingpt-8B-lora]="fingpt"
my_dict[CFGPT1-sft-7B-Full]="cfgpt"
my_dict[finma-7b-full]="finma"
my_dict[finma-7b-nlp]="finma"
my_dict[finma-7b-trade]="finma"

# 数据集列表
datasets=(
    #Chinese benchmark
    stockA_prediction
    finfe
    Fineval-multiple_choice
    finna
    finre
    finqa
    finnl
    finese
    fincqa
    CPA
    ################
    English benchmark
    flare-edtsum
    flare-fpb
    fiqasa
    flare-headlines
    flare-fomc
    CFA-multiple_choice
    DJIA_stock_prediction
    cra-lendingclub
    flare-ner
    flare-convfinqa
    flare-finqa
    fingpt-finred
)

# 输出目录基础路径
output_base=/finance_ML/ljx_data/saves/sh_test

# 循环遍历模型和数据集
for model in ${models[@]}; do
    for dataset in ${datasets[@]}; do
        # 构建输出路径
        output_dir=${output_base}/$(basename $model)/lora/predict/eval_dataset/${dataset}

        # 检查是否生成了有效的 output_dir
        echo "Output directory: $output_dir"
        echo "model directory: $model"
        echo "dataset: $dataset"
        echo "template: ${my_dict[$(basename $model)]}"

        # 执行命令
        CUDA_VISIBLE_DEVICES=1,4 llamafactory-cli train \
            --stage sft \
            --model_name_or_path $model \
            --preprocessing_num_workers 16 \
            --finetuning_type lora \
            --template ${my_dict[$(basename $model)]} \
            --eval_dataset $dataset \
            --cutoff_len 1024 \
            --overwrite_cache True \
            --max_samples 100000 \
            --per_device_eval_batch_size 2 \
            --predict_with_generate True \
            --output_dir $output_dir \
            --do_predict True \
            --overwrite_output_dir True

        echo "Completed training for model $model with dataset $dataset, output saved to $output_dir"
    done
done
