import argparse
import json
from pathlib import Path
from typing import Any, List, Union
from ast import literal_eval
from pymatgen.core import Composition


import random
import re

def extract_elements(formula: str) -> set:

    return set(re.findall(r"[A-Z][a-z]?", formula))

elem_library            = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al',
                           'Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe',
                           'Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr',
                           'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
                           'Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm',
                           'Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W',
                           'Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
                           'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf',
                           'Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds',
                           'Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

alkali_metal            = ['Li','Na','K','Rb','Cs']
alkaline_earth_metal    = ['Be','Mg','Ca','Sr','Ba']
transition_metal        = ['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
                           'Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd','Hf',
                           'Ta','W','Re','Os','Ir','Pt','Au','Hg']
lanthanide_elem         = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu']
actinide_elem           = ['Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']
post_transition_metal   = ['Al','Ga','In','Sn','Tl','Pb','Bi']
metalloid               = ['B','Si','Ge','As','Sb','Te']
non_metal               = ['H','C','N','O','F','P','S','Cl','Se','Br','I']
noble_gas               = ['He','Ne','Ar','Kr','Xe']
artificial_elem         = ['Tc','Pm','Po','At','Rn','Fr','Ra','Rf','Db','Sg','Bh',
                           'Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

essen_elem = alkali_metal + alkaline_earth_metal + transition_metal \
             + lanthanide_elem + actinide_elem + post_transition_metal + metalloid + ['P','Se','S']

inorg_elem = alkali_metal + alkaline_earth_metal + transition_metal \
             + lanthanide_elem + actinide_elem + post_transition_metal + metalloid


### Classifying material group based on anion type (code adapted from https://github.com/kaist-amsg/ElemwiseRetro/blob/main/Data.py)

def get_SourceElem(comp_list, comp_type='Target'):
    source_elem = []
    env_elem = []
    for comp in comp_list:
        non_source_elem = []
        comp_dict = Composition(comp).get_el_amt_dict()
        elements_seq = list(comp_dict.keys())

        for ee in elements_seq:
            if ee in essen_elem:
                source_elem.append(ee)
            else:
                non_source_elem.append(ee)
        for ee in non_source_elem:
            env_elem.append(ee)

    source_elem = list(set(source_elem))
    env_elem = list(set(env_elem))

    return source_elem, env_elem

def get_AnionPart(composition, source_elem, ExceptionMode=True, TargetTypeMode=True):
    comp_dict = Composition(composition).get_el_amt_dict()
    ca_count = 0
    an_count = 0
    anion = ""
    for elem, stoi in comp_dict.items():
        if TargetTypeMode:
            if str(elem) in inorg_elem:
            #if str(elem) in source_elem:
                ca_count += 1
            else:
                an_count += 1
                anion += str(elem)+str(stoi)
        else:
            if str(elem) in source_elem:
                ca_count += 1
            else:
                an_count += 1
                anion += str(elem)+str(stoi)
    if ca_count == 0:
        if ExceptionMode:
            pass
        else:
            raise NotImplementedError('No source elem', composition)

    if anion != "":
        anion = str(Composition(anion).get_integer_formula_and_factor()[0])
    return anion

def map_group_with_rules(anion: str) -> str:

    if anion == 'O2':
        return 'oxide'
    elif anion == 'Composite':
        return 'composite'
    elif anion == '':
        return 'alloy'
    elif anion == 'PO4':
        return 'phosphate'
    elif anion == 'P2O7':
        return 'pyrophosphate'
    elif anion in ['F2', 'Cl2', 'Br', 'I']:
        return 'halide'
    elif set(list(Composition(anion).get_el_amt_dict().keys())) in [set(['F','O']), set(['Cl','O']), set(['Br','O']), set(['I','O'])]:
        return 'oxyhalide'
    elif anion == 'C':
        return 'carbide'
    elif anion == 'N2':
        return 'nitride'
    elif anion == 'H2':
        return 'hydride'
    elif anion == 'Se':
        return 'selenide'
    elif anion == 'S':
        return 'sulfide'
    else:
        return 'other'

DATA_DIR = Path("./data")
SPLITS = ["train", "valid", "test"]


with open(DATA_DIR / "filtered_precursor_counts_ver1.json", "r", encoding="utf-8") as prec:
    precursor_list_dict = json.load(prec)
GLOBAL_ALLOWED_PRECURSORS: List[str] = list(precursor_list_dict.keys())

# === Prompt ===================================================================
synthesis_operation_prompt = "You are an expert solid-state chemist planning a material synthesis."


# === Utilities ================================================================
ALLOWED_CANON = {"heating","sintering","annealing","mixing","shaping","quenching","drying"}

def to_single_sequence(x: Any) -> List[str]:

    if x is None:
        raise ValueError("synthesis_operation is None")

    if isinstance(x, str):
        try:
            x = literal_eval(x)
        except Exception:
            raise ValueError("synthesis_operation string is not a valid Python literal")

    if isinstance(x, list):
        if all(isinstance(t, str) for t in x):
            return x
        if all(isinstance(t, list) for t in x):
            for seq in x:
                if seq:
                    if not all(isinstance(t, str) for t in seq):
                        raise ValueError("nested synthesis_operation contains non-string items")
                    return seq
            raise ValueError("all sequences are empty")
    raise ValueError(f"Unsupported synthesis_operation format: {type(x)}")

def canon_ops(seq: List[str], lowercase_ops: bool = True) -> List[str]:
    out = []
    for op in seq:
        op = str(op).strip()
        if lowercase_ops:
            op = op.lower()
        out.append(op)
    return out

def validate_ops(seq: List[str]):
    if not seq:
        raise ValueError("empty sequence")
    for op in seq:
        if op not in ALLOWED_CANON:
            raise ValueError(f"invalid op: {op}")
def to_single_precursor_set(x: Any) -> List[str]:

    if x is None:
        raise ValueError("ground-truth precursors is None")

    if isinstance(x, str):
        try:
            x = literal_eval(x)
        except Exception:
            raise ValueError("precursors string is not a valid Python literal")

    if isinstance(x, list):
        if all(isinstance(t, str) for t in x):
            return [t for t in x if t]
        if all(isinstance(t, list) for t in x):
            for seq in x:
                if seq and all(isinstance(t, str) for t in seq):
                    return [t for t in seq if t]
            raise ValueError("no non-empty precursor set found")
    raise ValueError(f"Unsupported precursors format: {type(x)}")
def pylist_with_single_quotes(obj: Union[List[str], str]) -> str:
    """Serialize list[str] to a Python-like list string with single quotes."""
    if isinstance(obj, str):
        return "'" + obj.replace("'", "\\'") + "'"
    if isinstance(obj, list):
        inner = ", ".join(pylist_with_single_quotes(v) for v in obj)
        return "[" + inner + "]"
    return repr(obj)

def extract_field(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default
def keyword_to_text(k):
    return (
        f"Host material: {k['host_material']}. "
        f"Dopant or substitution: {k['dopant_or_substitution']}. "
        f"Material class: {k['material_class']}. "
        f"Functional property: {k['functional_property']}. "
        f"Composition control: {k['composition_control']}."
    )
from collections import Counter
from typing import List, Dict, Tuple
def pylist_with_single_quotes_pc(xs):
    return "[" + ", ".join(f"'{x}'" for x in xs) + "]"

OPS = ['heating','sintering','annealing','mixing','shaping','quenching','drying']

TYPE_VOCAB = ["carbonate","nitrate","ammonium","phosphate","oxide","other"]


def precursor_type(p: str) -> str:

    p_clean = p.replace(" ", "")
    p_lower = p_clean.lower()

    if "no3" in p_lower: return "nitrate"
    if "nh4" in p_lower: return "ammonium"
    if "po4" in p_lower: return "phosphate"
    if "co3" in p_lower:
        if "CO3" in p_clean or "Co" not in p_clean:
            return "carbonate"

    if "O" in p_clean: 
        return "oxide"
        
    return "other"

def build_type_stats(precs: List[str]) -> Dict[str, float]:
    cnt = Counter(precursor_type(p) for p in precs)
    total = sum(cnt.values()) if cnt else 1
    return {t: cnt.get(t, 0) / total for t in TYPE_VOCAB}

def list_to_text(lst, wrap_brackets=False, quote_items=False):

    if isinstance(lst, list):
        if quote_items:
            text = ", ".join(f"'{str(x).strip()}'" for x in lst)
        else:
            text = ", ".join(map(str, lst))
        return f"[{text}]" if wrap_brackets else text
    return str(lst)



def create_prompt_pp(target_formula: str) -> str:
    return (
        f"Task:\n"
        f"Given the target compound {target_formula}, first classify its material group and must select the precursor needed to synthesize the target. "
        f"Typically, two or more precursors are required.\n"
        f"The possible material groups are: "
        f"['oxide', 'alloy', 'phosphate', 'pyrophosphate', 'halide', 'oxyhalide', "
        f"'carbide', 'nitride', 'selenide', 'sulfide', 'other'].\n"
        f"Output format:\n"
        f"[['material group'], ['precursor1', 'precursor2', ..., 'precursorN']]\n"
        f"Return only the Python list of single-quoted strings with no extra text or explanation."
    )


def create_prompt_sop(
    target_formula,
    precursors=None,
    precursor_types_block=None,
    keyword="",
    mode="sop", 
):
    parts = []

    if isinstance(precursors, (list, tuple)):
        prec_text = "[" + ", ".join(map(repr, precursors)) + "]"
    else:
        prec_text = str(precursors)

    if precursor_types_block:
        parts.append(precursor_types_block)

    parts.append(f"""
Task:
Given the target compound {target_formula}, first classify its material group,
then based on the selected precursors {prec_text},
propose one plausible synthesis operation sequence using only the following operations:
['heating', 'sintering', 'annealing', 'mixing', 'shaping', 'quenching', 'drying'].

Synthesis context:{keyword}

Typically, two or more precursors are required.

The possible material groups are:
['oxide', 'alloy', 'phosphate', 'pyrophosphate', 'halide', 'oxyhalide',
 'carbide', 'nitride', 'selenide', 'sulfide', 'other'].

Output format:
[['material group'], ['precursor_type1', 'precursor_type2', ..., 'precursor_typeK'], ['precursor1', 'precursor2', ..., 'precursorN'], ['op1', 'op2', ..., 'opM']]

Return only the Python list of single-quoted strings with no extra text or explanation.

""")
    
    return "\n".join(parts)



def build_precursor_types_block(type_stats, include_ratio: bool = True):
    types_present = [t for t, v in type_stats.items() if v > 0]

    lines = [
        "[PRECURSOR_TYPES]",
        f"types_present: {', '.join(types_present)}",
    ]

    if include_ratio:
        ratio_line = "type_ratio: " + ", ".join(
            f"{t}={type_stats[t]:.2f}" for t in TYPE_VOCAB
        )
        lines.append(ratio_line)

    lines.append("[/PRECURSOR_TYPES]")

    return "\n".join(lines) + "\n"



# === Main data builder ====
def mk_sft_format(split: str,
                  out_prefix: str = "precursor_synthops",
                  use_short_prompt: bool = True,
                  lowercase_ops: bool = True):
    in_path = DATA_DIR / f"{split}_dataset_ver1.json"
    out_path = DATA_DIR / f"{split}_{out_prefix}_ver1.jsonl"

    with open(in_path, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    keyword_path = DATA_DIR / f"title_keyword/{split}_keywords.json"
    with open(keyword_path, "r", encoding="utf-8") as f:
        keywords = json.load(f)

    n_ok = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for ex in source_data:
            retrieved_materials = []
            target_formula = extract_field(ex, "target_formula", "target", "Target")
            precursors = extract_field(ex, "precursors", "Precursors", "precursor_list", default=[])
            gt = extract_field(ex, "synthesis_operation", "synthesis_operations")
            source_elem, _ = get_SourceElem(target_formula, comp_type='Precursor')
            anion = get_AnionPart(target_formula[0], source_elem, ExceptionMode=True, TargetTypeMode=True)

            material_group = map_group_with_rules(anion)

            keyword = keyword_to_text(keywords[n_ok]['keywords'])

            if target_formula is None or gt is None:
                continue

            try:
                seq = to_single_sequence(gt)
                seq = canon_ops(seq, lowercase_ops=lowercase_ops)
                precursors = to_single_precursor_set(precursors)
                validate_ops(seq)
            except Exception:
                continue

            type_stats = build_type_stats(precursors)
            precursor_types = [t for t, v in type_stats.items() if v > 0]
            blocks = build_precursor_types_block(type_stats, include_ratio=args.ratio)
            
            if args.mode == "pp":
                user_prompt  = create_prompt_pp(target_formula)

            if args.mode == "sop":
                user_prompt  = create_prompt_sop(target_formula, precursors, blocks, keyword, mode=args.mode)

            op_out = pylist_with_single_quotes(seq)
            prec_out = pylist_with_single_quotes(precursors)
            material_group_out = pylist_with_single_quotes([material_group])
            precursor_type_out = pylist_with_single_quotes_pc(precursor_types)

            if args.mode == "sop" :
                assistant_out = f"[{material_group_out}, {precursor_type_out}, {prec_out}, {op_out}]"

            elif args.mode == 'pp':
                assistant_out = f"[{material_group_out}, {prec_out}]"
            else:
                raise ValueError(f"Unknown mode: {args.mode}")
            sample = {
                "messages": [
                    {"role": "system", "content": synthesis_operation_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_out}
                ]
            }
            w.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_ok += 1


    print(f"[{split}] {n_ok} samples -> {out_path}")

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="sop")
    args = ap.parse_args()

    for split in SPLITS:

        out_prefix = args.mode
        mk_sft_format(split, out_prefix=out_prefix, use_short_prompt=True)
        print(f"Processed {split} split (single).")
