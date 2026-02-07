

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0  

for BATCH in 4
do  
    export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    export TEST_FILE="data/test_pp_ver1.jsonl"
    export ADAPTER_DIR="outputs/pp_ver1_bs${BATCH}_qwen"
    export OUT_FILE="preds_qlora/pp_ver1_bs${BATCH}_test_qwen.jsonl"
    python inference_qwen.py    
done   

