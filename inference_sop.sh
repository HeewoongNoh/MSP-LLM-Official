


export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0  

for BATCH in 2
do  
    export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
    export TEST_FILE="data/test_sop_ver1.jsonl"
    export ADAPTER_DIR="outputs/sop_ver1_bs${BATCH}_llama"
    export OUT_FILE="preds/sop_ver1_bs${BATCH}_test_llama.jsonl"
    python inference_llama.py    
done

