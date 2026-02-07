# train_qlora_qwen25.py
import os
import torch
from typing import Dict
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
TRAIN_FILE = os.environ.get("TRAIN_FILE", "data/train_sop_ver1.jsonl")
VAL_FILE = os.environ.get("VAL_FILE", "data/valid_sop_ver1.jsonl") 
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/qwen")


os.makedirs(OUTPUT_DIR, exist_ok=True)

"""
LoRA was applied to the attention blocks (q, k, v, o). While extending tuning to the MLP layers (gate, up, down) is feasible, preliminary experiments showed no significant improvement in model performance.
"""


lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    bias="none", task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=(
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    ),
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=(
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    ),
    device_map={"": torch.cuda.current_device()}, 
    low_cpu_mem_usage=True,
)



model.config.use_cache = False
model.gradient_checkpointing_enable()
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model)

model.lm_head = model.lm_head.to(torch.float32)
model.get_output_embeddings().weight.data = (
    model.get_output_embeddings().weight.data.to(torch.float32)
)

train_data = load_dataset(
    "json", data_files=TRAIN_FILE, split="train"
)
val_data = (
    load_dataset("json", data_files=VAL_FILE, split="train")
    if VAL_FILE else None
)


def format_example(example: Dict) -> Dict:
    conv = example["messages"]
    if not conv or conv[-1]["role"] != "assistant":
        raise ValueError("Conversation must end with assistant message")
    return {
        "text": tokenizer.apply_chat_template(
            conv, tokenize=False
        )
    }


train_data = train_data.map(
    format_example,
    remove_columns=train_data.column_names
)
if val_data is not None:
    val_data = val_data.map(
        format_example,
        remove_columns=val_data.column_names
    )

response_template = "<|im_start|>assistant\n"

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
    ignore_index=-100,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=int(os.environ.get("BATCH", 16)),
    per_device_eval_batch_size=int(os.environ.get("EVAL_BATCH", 1)),
    gradient_accumulation_steps=int(os.environ.get("GRAD_ACC", 2)),
    num_train_epochs=float(os.environ.get("EPOCHS", 3)),
    learning_rate=float(os.environ.get("LR", 1e-4)),
    lr_scheduler_type=os.environ.get("SCHED", "cosine"),
    warmup_ratio=float(os.environ.get("WARMUP", 0.03)),
    logging_steps=int(os.environ.get("LOG_STEPS", 1)),
    save_strategy="epoch",
    save_total_limit=3,
    eval_strategy="steps" if val_data is not None else "no",
    eval_steps=int(os.environ.get("EVAL_STEPS", 200)) if val_data is not None else None,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    optim="paged_adamw_8bit",
    weight_decay=0.0,
    max_grad_norm=1.0,
    logging_first_step=True,
    report_to=["none"],
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    peft_config=lora_config,
    train_dataset=train_data,
    eval_dataset=val_data,
    dataset_text_field="text",
    data_collator=collator,
    args=training_args,
)

trainer.train()
trainer.save_model()
