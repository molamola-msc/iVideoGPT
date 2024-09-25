#!/bin/sh
# sbatch -N 1 --gres=gpu:4 --qos=gpugpu --nodelist=paraai-n32-h-01-agent-9 -p vip_gpu_scx6378_02 run.sh
# sbatch -N 1 --gres=gpu:2 --qos=gpugpu --nodelist=paraai-n32-h-01-agent-47 -p vip_gpu_scx6378_02 run.sh

module purge
module load anaconda/2021.11 compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
source activate ivideogpt # your environment name
export PYTHONUNBUFFERED=1

# accelerate launch train_tokenizer.py \
#     --exp_name bair_tokenizer_ft --output_dir log_vqgan --seed 0 --mixed_precision bf16 \
#     --model_type ctx_vqgan \
#     --train_batch_size 16 --gradient_accumulation_steps 1 --disc_start 1000005 \
#     --oxe_data_mixes_type open-television --resolution 64 --dataloader_num_workers 16 \
#     --rand_select --video_stepsize 3 --segment_horizon 16 --segment_length 8 --context_length 1 \
#     --pretrained_model_name_or_path pretrained_models
#     >> trm03.log 2>&1 &
# wait

accelerate launch --config_file accelerate_config.yaml train_gpt.py\
    --exp_name bair_llama_ft --output_dir log_trm --seed 0 --mixed_precision bf16 \
    --vqgan_type ctx_vqgan \
    --pretrained_model_name_or_path /home/bingxing2/home/scx6378/miaoshangchen/iVideoGPT-main/log_vqgan/2024-09-21-01:44:12-bair_tokenizer_ft/checkpoint-580000/unwrapped_model \
    --config_name configs/llama/config.json --action_dim 3390 \
    --pretrained_transformer_path pretrained_transformer/pretrained_transformer \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --lr_scheduler_type cosine \
    --oxe_data_mixes_type open-television --resolution 64 --dataloader_num_workers 16 \
    --video_stepsize 3 --segment_length 16 --context_length 1 \
    --use_fvd --use_frame_metrics \
    --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd --action_conditioned --load_internal_llm --action_cat \
    --frame_per_action 15 --action_predict 100
    # >> trm03.log 2>&1 &

wait