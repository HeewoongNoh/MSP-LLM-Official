import os
import json, ast, re, torch
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


MODEL_NAME  = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "outputs_qwen/ckpt_folder")
TEST_FILE   = os.environ.get("TEST_FILE",   "data/test_sop_ver1.jsonl")
OUT_FILE    = os.environ.get("OUT_FILE",    "preds/inferenced_file.jsonl")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

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

tok = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,  
    use_fast=True,
)
tok.padding_side = "right"

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=(
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    ),
    device_map={"": 0},     
    low_cpu_mem_usage=True,
)

base.resize_token_embeddings(len(tok))
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

def parse_list(text: str) -> List[str]:
    t = text if isinstance(text, str) else str(text)
    t = t.strip()

    s = t.find('[')
    if s == -1:
        raise ValueError('no list-like output')

    depth = 0
    e = -1
    for i in range(s, len(t)):
        ch = t[i]
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                e = i
                break

    if e == -1:
        last_r = t.rfind(']')
        if last_r != -1 and last_r > s:
            snippet = t[s:last_r+1]
        else:
            snippet = t[s:]
            if not snippet.endswith(']'):
                snippet += ']'
    else:
        snippet = t[s:e+1]

    try:
        lst = ast.literal_eval(snippet)
        if isinstance(lst, list):
            return [str(x) for x in lst]
        raise ValueError("not a list")
    except Exception:
        inner = snippet[1:-1]
        token_re = re.compile(
            r'''\s*(?:"((?:\\.|[^"\\])*)"|'((?:\\.|[^'\\])*)'|([^,\[\]\n]+))\s*(?:,|$)'''
        )
        tokens, pos = [], 0
        while pos < len(inner):
            m = token_re.match(inner, pos)
            if not m:
                break
            dq, sq, bare = m.group(1), m.group(2), m.group(3)
            if dq is not None:
                tokens.append(dq)
            elif sq is not None:
                tokens.append(sq)
            elif bare is not None:
                b = bare.strip()
                if b:
                    tokens.append(b)
            pos = m.end()
        if not tokens:
            raise ValueError("parse failed")
        return [str(x) for x in tokens]


GEN_KW = dict(
    max_new_tokens=128,
    do_sample=False,    
    num_beams=10,         
    num_return_sequences=10,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.pad_token_id,
)
if GEN_KW["num_return_sequences"] > 1 and not GEN_KW["do_sample"]:
    GEN_KW["num_beams"] = max(GEN_KW["num_return_sequences"], 1)

ds = load_dataset("json", data_files=TEST_FILE, split="train")

def build_messages_for_infer(messages: List[Dict]) -> List[Dict]:
    if not messages:
        raise ValueError("messages empty")
    if messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def make_prompt(msgs: List[Dict]) -> str:
    return tok.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=False,
    )

pred_count, err_count = 0, 0

with open(OUT_FILE, "w", encoding="utf-8") as w:
    for ex in ds:
        try:
            msgs = build_messages_for_infer(ex["messages"])
            prompt = make_prompt(msgs)

            enc = tok(prompt, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            prefix_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                gen = model.generate(**enc, **GEN_KW)

            if GEN_KW["num_return_sequences"] == 1:
                out_text = tok.decode(
                    gen[0][prefix_len:], skip_special_tokens=True
                )
                try:
                    pred_list = parse_list(out_text)
                except Exception:
                    out_full = tok.decode(gen[0], skip_special_tokens=True)
                    pred_list = parse_list(out_full)

                rec = {
                    "prediction": pred_list,
                }
                if "messages" in ex:
                    rec["messages"] = ex["messages"]
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pred_count += 1
            else:
                seqs = []
                for i in range(gen.shape[0]):
                    out_text = tok.decode(
                        gen[i][prefix_len:], skip_special_tokens=True
                    )
                    try:
                        seqs.append(parse_list(out_text))
                    except Exception:
                        out_full = tok.decode(gen[i], skip_special_tokens=True)
                        seqs.append(parse_list(out_full))

                rec = {
                    "predictions": seqs,
                    "n": GEN_KW["num_return_sequences"],
                }
                if "messages" in ex:
                    rec["messages"] = ex["messages"]
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pred_count += 1

        except Exception as e:
            err_count += 1
            w.write(
                json.dumps({"error": repr(e), "raw": ex}, ensure_ascii=False) + "\n"
            )

print(f"[DONE] wrote {pred_count} preds to {OUT_FILE} (errors: {err_count})")
