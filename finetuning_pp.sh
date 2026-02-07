

export CUDA_VISIBLE_DEVICES=0.1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0   

for BATCH in 4
do
    export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"  #"meta-llama/Meta-Llama-3.1-8B-Instruct" 
    export TRAIN_FILE="data/train_pp_ver1.jsonl"
    export VAL_FILE="data/valid_pp_ver1.jsonl"
    export OUTPUT_DIR="outputs/pp_ver1_bs${BATCH}_qwen"
    export BATCH=$BATCH

    accelerate launch \
        --multi_gpu \
        --num_processes 2 \
        --main_process_port 29509 \
        finetuning_qwen.py > logs/pp_ver1_bs${BATCH}_qwen_$(date +%Y%m%d_%H%M%S).log 2>&1
done