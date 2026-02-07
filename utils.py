from pymatgen.core import Composition
import ast
from typing import List, Dict, Any, Tuple
import re

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

_ALLOWED_GROUPS = {
    'oxide','alloy','phosphate','pyrophosphate','halide','oxyhalide',
    'carbide','nitride','selenide','sulfide','composite','hydride','other'
}
def extract_field(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _normalize_group(g: str) -> str:
    if not isinstance(g, str):
        return ""
    g = g.strip().lower()
    alias = {
        'oxides': 'oxide',
        'alloys': 'alloy',
        'phosphates': 'phosphate',
        'pyro-phosphate': 'pyrophosphate',
        'oxy-halide': 'oxyhalide',
        'carbides': 'carbide',
        'nitrides': 'nitride',
        'selenides': 'selenide',
        'sulfides': 'sulfide',
        'composites': 'composite',
        'hydrides': 'hydride',
    }
    g = alias.get(g, g)
    return g if g in _ALLOWED_GROUPS else g  

def _first_or_empty(xs: list) -> str:
    return xs[0] if isinstance(xs, list) and xs else ""

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
            e = last_r
        else:
            snippet = t[s:]
            if not snippet.endswith(']'):
                snippet = snippet + ']'
    else:
        snippet = t[s:e+1]

    try:
        lst = ast.literal_eval(snippet)
        if isinstance(lst, list):
            return [str(x) for x in lst]
        else:
            raise ValueError('parsed object is not a list')
    except Exception:
        inner = snippet[1:-1]
        token_re = re.compile(r'''\s*(?:"((?:\\.|[^"\\])*)"|'((?:\\.|[^'\\])*)'|([^,\[\]\n]+))\s*(?:,|$)''')

        tokens = []
        pos = 0
        while pos < len(inner):
            m = token_re.match(inner, pos)
            if not m:
                break
            dq, sq, bare = m.group(1), m.group(2), m.group(3)
            if dq is not None:
                val = dq.encode('utf-8').decode('unicode_escape')
                tokens.append(val)
            elif sq is not None:
                val = sq.encode('utf-8').decode('unicode_escape')
                tokens.append(val)
            elif bare is not None:
                b = bare.strip()
                if b != '':
                    tokens.append(b)
            pos = m.end()

        if not tokens:
            raise ValueError('could not parse list output')

        # final coercion: all to str
        return [str(x) for x in tokens]

def _parse_list_of_str(obj: Any) -> List[str]:

    if isinstance(obj, list):
        return [str(x) for x in obj if x is not None]
    if isinstance(obj, str):
        try:
            parsed = ast.literal_eval(obj)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x is not None]
        except Exception:
            return []
    return []

def parse_pred_sets(s) -> List[List[str]]:

    if isinstance(s, list):
        return [list(x) for x in s]

    if not isinstance(s, str):
        raise TypeError("Input must be a str or list.")

    obj = ast.literal_eval(s)       
    if isinstance(obj, str):
        obj = ast.literal_eval(obj)

    obj = [list(inner) for inner in obj]
    if not all(isinstance(y, str) for inner in obj for y in inner):
        raise ValueError("Parsed object is not List[List[str]].")
    return obj

def standardize_formula(formula):
    if not isinstance(formula, str) or not formula or formula == "None":
        return formula
    formula = formula.strip()
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return formula


from typing import List, Dict

def _normalize_precursor_list(lst, use_reduced: bool) -> set:
    if not lst:
        return set()
    out = []
    for p in lst:
        if not p:
            continue
        x = standardize_formula(p) if use_reduced else p
        if x:
            out.append(x)
    return set(out)

OPERATION_CANONICAL_MAP = {
    "heating": "heating",
    "heated": "heating",
    "heat": "heating",
    "firing": "heating",
    "calcination": "heating",

    "sintering": "sintering",
    "sinter": "sintering",

    "annealing": "annealing",
    "anneal": "annealing",

    "mixing": "mixing",
    "mix": "mixing",

    "shaping": "shaping",
    "shape": "shaping",
    "pressing": "shaping",

    "quenching": "quenching",
    "quench": "quenching",

    "drying": "drying",
    "dry": "drying",
}

def _normalize_operation(op: str) -> str:
    if not op:
        return ""
    key = op.strip().lower()
    return OPERATION_CANONICAL_MAP.get(key, key)  

def _normalize_operations(seq: List[str]) -> List[str]:
    return [_normalize_operation(op) for op in seq if op]
from collections import Counter

def _levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[n][m]


def _ned_similarity(pred, gt):
    if not pred and not gt:
        return 1.0
    d = _levenshtein(pred, gt)
    return 1.0 - d / max(len(pred), len(gt), 1)


def _lcs_len(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[n][m]


def _lcs_f1(pred, gt):
    if not pred or not gt:
        return 0.0
    l = _lcs_len(pred, gt)
    p = l / len(pred)
    r = l / len(gt)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _multiset_f1(pred, gt):
    cp = Counter(pred)
    cg = Counter(gt)
    match = sum(min(cp[k], cg[k]) for k in cp.keys() | cg.keys())

    if match == 0:
        return 0.0

    p = match / sum(cp.values())
    r = match / sum(cg.values())
    return 2 * p * r / (p + r)

def evaluate_pp(
    answer_json: List[Dict],
    pred_pairs_list: List[List],  
    reduced_formula: bool = True,
    Ks: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[str, Dict[str, float]]:

    prec_hits  = {k: 0 for k in Ks}
    ops_hits   = {k: 0 for k in Ks}
    joint_hits = {k: 0 for k in Ks}
    group_hits = {k: 0 for k in Ks}

    n_prec = n_ops = n_joint = 0
    n_group = 0

    for answer, ranked_items in zip(answer_json, pred_pairs_list):
        target_formula = extract_field(answer, "target_formula", "target", "Target")
        source_elem, _ = get_SourceElem(target_formula, comp_type='Precursor')
        anion = get_AnionPart(target_formula[0], source_elem, ExceptionMode=True, TargetTypeMode=True)
        gt_group = map_group_with_rules(anion) 
        gt_group_norm = _normalize_group(gt_group)

        gt_prec_raw = answer.get("precursors", []) or []
        gt_ops_raw  = answer.get("synthesis_operation", []) or answer.get("operations", []) or []

        gt_prec_set = _normalize_precursor_list(gt_prec_raw, reduced_formula) if gt_prec_raw else set()
        gt_ops_norm = _normalize_operations(gt_ops_raw) if gt_ops_raw else []

        cand_tuples: List[Tuple[set, List[str], str]] = []

        for item in (ranked_items or []):
            if not item:
                continue

            if isinstance(item, (list, tuple)) and len(item) >= 3:
                raw_group, raw_pc_types, raw_prec = item[0], item[1], item[2]
                cand_group_list = _parse_list_of_str(raw_group)
                cand_group_norm = _normalize_group(_first_or_empty(cand_group_list))

                cand_prec = _parse_list_of_str(raw_prec)

            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                raw_group, raw_prec = item[0], item[1]
                cand_group_list = _parse_list_of_str(raw_group)
                cand_group_norm = _normalize_group(_first_or_empty(cand_group_list))
                cand_prec = _parse_list_of_str(raw_prec)


            else:
                continue

            cand_prec_set = _normalize_precursor_list(cand_prec, reduced_formula)
            cand_ops_norm = _normalize_operations([])
            cand_tuples.append((cand_prec_set, cand_ops_norm, cand_group_norm))

        if gt_prec_set:
            n_prec += 1
            for k in Ks:
                topk = cand_tuples[:k]
                ok = any(gt_prec_set == cprec for (cprec, _, _) in topk)
                if ok:
                    prec_hits[k] += 1

        if gt_ops_norm:
            n_ops += 1
            for k in Ks:
                topk = cand_tuples[:k]
                ok = any(gt_ops_norm == cops for (_, cops, _) in topk)
                if ok:
                    ops_hits[k] += 1

        if gt_prec_set and gt_ops_norm:
            n_joint += 1
            for k in Ks:
                topk = cand_tuples[:k]
                ok = any((gt_prec_set == cprec) and (gt_ops_norm == cops) for (cprec, cops, _) in topk)
                if ok:
                    joint_hits[k] += 1

        if gt_group_norm:  
            n_group += 1
            for k in Ks:
                topk = cand_tuples[:k]
                ok = any((cgroup != "") and (gt_group_norm == cgroup) for (_, _, cgroup) in topk)
                if ok:
                    group_hits[k] += 1

    def _pack(hits: Dict[int, int], n: int) -> Dict[str, float]:
        return {
            "top1":  hits.get(1, 0)  / n if n else 0.0,
            "top3":  hits.get(3, 0)  / n if n else 0.0,
            "top5":  hits.get(5, 0)  / n if n else 0.0,
            "top10": hits.get(10, 0) / n if n else 0.0,
            "n": float(n),
        }

    return {
        "precursors": _pack(prec_hits, n_prec),
    }


def evaluate_sop(
    answer_json: List[Dict],
    preds_all: List[List],
    reduced_formula: bool = True,
    Ks: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[str, Dict[str, Dict[str, float]]]:

    def _eval_core(
        answer_json_subset: List[Dict],
        pred_pairs_list_subset: List[List],
        Ks: Tuple[int, ...],
        reduced_formula: bool,
    ):
        prec_hits  = {k: 0 for k in Ks}
        ops_hits   = {k: 0 for k in Ks}
        joint_hits = {k: 0 for k in Ks}
        group_hits = {k: 0 for k in Ks}

        ops_scores = {
            "ned":   {k: 0.0 for k in Ks},
            "lcs":   {k: 0.0 for k in Ks},
            "msf1":  {k: 0.0 for k in Ks},
        }

        joint_scores = {
            "ned":   {k: 0.0 for k in Ks},
            "lcs":   {k: 0.0 for k in Ks},
            "msf1":  {k: 0.0 for k in Ks},
        }

        n_prec = n_ops = n_joint = n_group = 0

        for answer, ranked_items in zip(answer_json_subset, pred_pairs_list_subset):
            target_formula = extract_field(answer, "target_formula", "target", "Target")
            source_elem, _ = get_SourceElem(target_formula, comp_type='Precursor')
            anion = get_AnionPart(target_formula[0], source_elem, ExceptionMode=True, TargetTypeMode=True)
            gt_group = map_group_with_rules(anion)
            gt_group_norm = _normalize_group(gt_group)

            gt_prec_raw = answer.get("precursors", []) or []
            gt_ops_raw  = answer.get("synthesis_operation", []) or answer.get("operations", []) or []

            gt_prec_set = _normalize_precursor_list(gt_prec_raw, reduced_formula) if gt_prec_raw else set()
            gt_ops_norm = _normalize_operations(gt_ops_raw) if gt_ops_raw else []

            cand_tuples = []

            for item in (ranked_items or []):
                if not item:
                    continue

                if isinstance(item, (list, tuple)) and len(item) == 3:
                    raw_group, raw_prec, raw_ops = item
                    cand_group_list = _parse_list_of_str(raw_group)
                    cand_group_norm = _normalize_group(_first_or_empty(cand_group_list))
                    cand_prec = _parse_list_of_str(raw_prec)
                    cand_ops  = _parse_list_of_str(raw_ops)

                elif isinstance(item, (list, tuple)) and len(item) >= 4:
                    raw_group, raw_prec, raw_ops = item[0], item[2], item[3]
                    cand_group_list = _parse_list_of_str(raw_group)
                    cand_group_norm = _normalize_group(_first_or_empty(cand_group_list))
                    cand_prec = _parse_list_of_str(raw_prec)
                    cand_ops  = _parse_list_of_str(raw_ops)

                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    raw_prec, raw_ops = item[0], item[1]
                    cand_group_norm = ""
                    cand_prec = _parse_list_of_str(raw_prec)
                    cand_ops  = _parse_list_of_str(raw_ops)

                else:
                    continue

                cand_prec_set = _normalize_precursor_list(cand_prec, reduced_formula)
                cand_ops_norm = _normalize_operations(cand_ops)

                cand_tuples.append((cand_prec_set, cand_ops_norm, cand_group_norm))

            if gt_prec_set:
                n_prec += 1
                for k in Ks:
                    if any(gt_prec_set == cprec for (cprec, _, _) in cand_tuples[:k]):
                        prec_hits[k] += 1

            if gt_ops_norm:
                n_ops += 1
                for k in Ks:
                    if any(gt_ops_norm == cops for (_, cops, _) in cand_tuples[:k]):
                        ops_hits[k] += 1

            if gt_ops_norm:
                for k in Ks:
                    topk = cand_tuples[:k]
                    best_ned = best_lcs = best_msf1 = 0.0

                    for (_, cops, _) in topk:
                        best_ned = max(best_ned, _ned_similarity(cops, gt_ops_norm))
                        best_lcs = max(best_lcs, _lcs_f1(cops, gt_ops_norm))
                        best_msf1 = max(best_msf1, _multiset_f1(cops, gt_ops_norm))

                    ops_scores["ned"][k] += best_ned
                    ops_scores["lcs"][k] += best_lcs
                    ops_scores["msf1"][k] += best_msf1

            if gt_prec_set and gt_ops_norm:
                n_joint += 1
                for k in Ks:
                    if any((gt_prec_set == cprec) and (gt_ops_norm == cops)
                           for (cprec, cops, _) in cand_tuples[:k]):
                        joint_hits[k] += 1

            if gt_prec_set and gt_ops_norm:
                for k in Ks:
                    topk = cand_tuples[:k]
                    best_ned = best_lcs = best_msf1 = 0.0

                    for (cprec, cops, _) in topk:
                        if cprec == gt_prec_set:
                            best_ned = max(best_ned, _ned_similarity(cops, gt_ops_norm))
                            best_lcs = max(best_lcs, _lcs_f1(cops, gt_ops_norm))
                            best_msf1 = max(best_msf1, _multiset_f1(cops, gt_ops_norm))

                    joint_scores["ned"][k] += best_ned
                    joint_scores["lcs"][k] += best_lcs
                    joint_scores["msf1"][k] += best_msf1

            if gt_group_norm:
                n_group += 1
                for k in Ks:
                    if any((cgroup != "") and (gt_group_norm == cgroup)
                           for (_, _, cgroup) in cand_tuples[:k]):
                        group_hits[k] += 1

        return {
            "hits": {
                "precursors": prec_hits,
                "operations": ops_hits,
                "joint": joint_hits,
                "group": group_hits,
            },
            "n": {
                "precursors": n_prec,
                "operations": n_ops,
                "joint": n_joint,
                "group": n_group,
            },
            "ops_scores": ops_scores,
            "joint_scores": joint_scores,
        }

    def _pack(h, n):
        return {
            "top1": h.get(1, 0) / n if n else 0.0,
            "top3": h.get(3, 0) / n if n else 0.0,
            "top5": h.get(5, 0) / n if n else 0.0,
            "top10": h.get(10, 0) / n if n else 0.0,
            "n": float(n),
        }

    def _pack_scores(score_dict, n):
        return {
            "top1": score_dict.get(1, 0.0) / n if n else 0.0,
            "top3": score_dict.get(3, 0.0) / n if n else 0.0,
            "top5": score_dict.get(5, 0.0) / n if n else 0.0,
            "top10": score_dict.get(10, 0.0) / n if n else 0.0,
        }

    core = _eval_core(answer_json, preds_all, Ks, reduced_formula)

    overall = {
        "operations": _pack(core["hits"]["operations"], core["n"]["operations"]),
        "operations_soft": {
            "ned": _pack_scores(core["ops_scores"]["ned"], core["n"]["operations"]),
            "lcs": _pack_scores(core["ops_scores"]["lcs"], core["n"]["operations"]),
            "msf1": _pack_scores(core["ops_scores"]["msf1"], core["n"]["operations"]),
        },

    }

    return {"overall": overall}



def evaluate_msp(
    answer_json: List[Dict],
    preds_all: List[List],         
    filtered_precs: List[List],  
    reduced_formula: bool = True,
    Ks: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[str, Dict[str, Dict[str, float]]]:

    def _eval_core(
        answer_json_subset: List[Dict],
        pred_pairs_list_subset: List[List],
        filtered_precs_subset: List[List],
        Ks: Tuple[int, ...],
        reduced_formula: bool,
    ):
        prec_hits  = {k: 0 for k in Ks}
        ops_hits   = {k: 0 for k in Ks}
        joint_hits = {k: 0 for k in Ks}
        group_hits = {k: 0 for k in Ks}

        ops_scores = {
            "ned":   {k: 0.0 for k in Ks},
            "lcs":   {k: 0.0 for k in Ks},
            "msf1":  {k: 0.0 for k in Ks},
        }
        joint_scores = {
            "ned":   {k: 0.0 for k in Ks},
            "lcs":   {k: 0.0 for k in Ks},
            "msf1":  {k: 0.0 for k in Ks},
        }

        n_prec = n_ops = n_joint = 0
        n_group = 0

        for answer, ranked_items, prec_top1 in zip(
            answer_json_subset,
            pred_pairs_list_subset,
            filtered_precs_subset
        ):
            target_formula = extract_field(answer, "target_formula", "target", "Target")
            source_elem, _ = get_SourceElem(target_formula, comp_type='Precursor')
            anion = get_AnionPart(
                target_formula[0],
                source_elem,
                ExceptionMode=True,
                TargetTypeMode=True
            )
            gt_group_norm = _normalize_group(map_group_with_rules(anion))

            gt_prec_raw = answer.get("precursors", []) or []
            gt_ops_raw = (
                answer.get("synthesis_operation", [])
                or answer.get("operations", [])
                or []
            )

            gt_prec_set = (
                _normalize_precursor_list(gt_prec_raw, reduced_formula)
                if gt_prec_raw else set()
            )
            gt_ops_norm = _normalize_operations(gt_ops_raw) if gt_ops_raw else []

            cand_prec_set = (
                _normalize_precursor_list(prec_top1, reduced_formula)
                if prec_top1 else set()
            )

            if gt_prec_set:
                n_prec += 1
            if gt_ops_norm:
                n_ops += 1
            if gt_group_norm:
                n_group += 1
            if gt_prec_set and gt_ops_norm:
                n_joint += 1

            if gt_prec_set and cand_prec_set == gt_prec_set:
                for k in Ks:
                    prec_hits[k] += 1

            for k in Ks:
                topk_items = (ranked_items or [])[:k]

                op_ok = False
                group_ok = False
                joint_ok = False  

                best_ned = best_lcs = best_msf1 = 0.0
                best_joint_ned = best_joint_lcs = best_joint_msf1 = 0.0

                for item in topk_items:
                    if not item:
                        continue

                    if not (isinstance(item, (list, tuple)) and len(item) >= 4):
                        continue

                    raw_group = item[0]
                    raw_ops   = item[3]

                    cand_group_norm = _normalize_group(
                        _first_or_empty(_parse_list_of_str(raw_group))
                    )
                    cand_ops = _parse_list_of_str(raw_ops)
                    cand_ops_norm = _normalize_operations(cand_ops)

                    if gt_ops_norm and cand_ops_norm == gt_ops_norm:
                        op_ok = True

                    if gt_ops_norm:
                        best_ned = max(best_ned, _ned_similarity(cand_ops_norm, gt_ops_norm))
                        best_lcs = max(best_lcs, _lcs_f1(cand_ops_norm, gt_ops_norm))
                        best_msf1 = max(best_msf1, _multiset_f1(cand_ops_norm, gt_ops_norm))

                    if gt_group_norm and cand_group_norm == gt_group_norm:
                        group_ok = True

                    if (
                        gt_prec_set
                        and cand_prec_set == gt_prec_set
                        and gt_ops_norm
                        and cand_ops_norm == gt_ops_norm
                    ):
                        joint_ok = True  

                    if (
                        gt_prec_set
                        and cand_prec_set == gt_prec_set
                        and gt_ops_norm
                    ):
                        best_joint_ned = max(best_joint_ned, _ned_similarity(cand_ops_norm, gt_ops_norm))
                        best_joint_lcs = max(best_joint_lcs, _lcs_f1(cand_ops_norm, gt_ops_norm))
                        best_joint_msf1 = max(best_joint_msf1, _multiset_f1(cand_ops_norm, gt_ops_norm))

                if gt_ops_norm and op_ok:
                    ops_hits[k] += 1
                if gt_group_norm and group_ok:
                    group_hits[k] += 1
                if gt_prec_set and gt_ops_norm and joint_ok:
                    joint_hits[k] += 1  

                if gt_ops_norm:
                    ops_scores["ned"][k]  += best_ned
                    ops_scores["lcs"][k]  += best_lcs
                    ops_scores["msf1"][k] += best_msf1

                if gt_prec_set and gt_ops_norm:
                    joint_scores["ned"][k]  += best_joint_ned
                    joint_scores["lcs"][k]  += best_joint_lcs
                    joint_scores["msf1"][k] += best_joint_msf1

        return {
            "hits": {
                "precursors": prec_hits,
                "operations": ops_hits,
                "joint":      joint_hits,
                "group":      group_hits,
            },
            "n": {
                "precursors": n_prec,
                "operations": n_ops,
                "joint":      n_joint,
                "group":      n_group,
            },
            "ops_scores": ops_scores,
            "joint_scores": joint_scores,
        }

    def _pack(h: Dict[int, int], n: int) -> Dict[str, float]:
        return {
            "top1":  h.get(1, 0)  / n if n else 0.0,
            "top3":  h.get(3, 0)  / n if n else 0.0,
            "top5":  h.get(5, 0)  / n if n else 0.0,
            "top10": h.get(10, 0) / n if n else 0.0,
            "n": float(n),
        }

    def _pack_scores(score_dict, n):
        return {
            "top1":  score_dict.get(1, 0.0)  / n if n else 0.0,
            "top3":  score_dict.get(3, 0.0)  / n if n else 0.0,
            "top5":  score_dict.get(5, 0.0)  / n if n else 0.0,
            "top10": score_dict.get(10, 0.0) / n if n else 0.0,
        }

    def _pack_all(core) -> Dict[str, Dict[str, float]]:
        return {
            "joint":      _pack(core["hits"]["joint"],      core["n"]["joint"]),

        }

    core = _eval_core(
        answer_json,
        preds_all,
        filtered_precs,
        Ks,
        reduced_formula
    )

    return {"overall": _pack_all(core)}
