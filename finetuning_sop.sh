export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0  


for BATCH in 2
do

    export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
    export TRAIN_FILE="data/train_sop_ver1.jsonl"
    export VAL_FILE="data/valid_sop_ver1.jsonl"
    export OUTPUT_DIR="outputs/sop_ver1_bs${BATCH}_llama"
    export BATCH=$BATCH

    accelerate launch \
        --multi_gpu \
        --num_processes 2 \
        --main_process_port 29504 \
        train_qlora_op_moe_output.py > logs/sop_ver1_bs${BATCH}_llama_$(date +%Y%m%d_%H%M%S).log 2>&1
done