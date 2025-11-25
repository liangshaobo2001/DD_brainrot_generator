#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build a phrase lexicon for 谢彬烂梗生成器.

Sources (all optional, controlled by CLI flags):

1. THUOCL (local, already downloaded)
2. Wikipedia-zh titles via HuggingFace `wikimedia/wikipedia` with abbreviation variants:
     - 黑神话：悟空 -> 黑神话：悟空, 黑神话, 悟空, 黑神话悟空
     - 寂静岭系列 -> 寂静岭系列, 寂静岭
3. (Best-effort) THUCNews titles via HF (may fail due to HF script changes)
4. (Best-effort) Weibo short phrases via HF (may fail similarly)
5. Optional allowlist file to force-include specific phrases (e.g. 舒肤佳)

Output:
  TSV: phrase<TAB>score
"""

import argparse
import collections
import math
import os
import re
import sys
from typing import Dict, Iterable, List, Tuple, Optional

# -----------------------------
# Optional datasets + OpenCC
# -----------------------------
try:
    from datasets import load_dataset, Value
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

try:
    from opencc import OpenCC  # type: ignore
    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None

# -----------------------------
# Basic CJK / text utils
# -----------------------------

CJK_SPAN_RE = re.compile(r'[\u4e00-\u9fff]+')

def log(msg: str) -> None:
    print(msg, file=sys.stderr)

def is_cjk_char(ch: str) -> bool:
    return '\u4e00' <= ch <= '\u9fff'

def is_all_cjk(s: str) -> bool:
    s = s.strip()
    return bool(s) and all(is_cjk_char(ch) for ch in s)

def extract_cjk_spans(text: str) -> Iterable[str]:
    """Yield contiguous CJK spans from text."""
    for m in CJK_SPAN_RE.finditer(text):
        span = m.group(0)
        if span:
            yield span

def normalize_phrase(s: str) -> str:
    return s.strip()

# -----------------------------
# THUOCL loader
# -----------------------------

def load_thuocl(thuocl_root: str,
                min_len: int = 2,
                max_len: int = 8,
                weight: float = 1.0) -> Dict[str, float]:
    """
    Load THUOCL files from a directory:
      corpus/THUOCL/THUOCL_*.txt

    Each line: "phrase freq"
    """
    data_dir = thuocl_root
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"THUOCL directory not found: {data_dir}")

    counter: Dict[str, float] = collections.Counter()

    files = [f for f in os.listdir(data_dir)
             if f.startswith("THUOCL_") and f.endswith(".txt")]
    files.sort()
    for fname in files:
        path = os.path.join(data_dir, fname)
        log(f"[THUOCL] Loading {path} ...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                phrase, freq_str = parts[0], parts[1]
                phrase = normalize_phrase(phrase)
                if not phrase:
                    continue
                if not (min_len <= len(phrase) <= max_len):
                    continue
                if not is_all_cjk(phrase):
                    continue
                try:
                    freq = float(freq_str)
                except Exception:
                    freq = 1.0
                # log-frequency scaled by weight
                score = weight * math.log1p(freq)
                counter[phrase] += score

    log(f"[THUOCL] {len(counter)} unique phrases after filter")
    return counter

# -----------------------------
# Wikipedia title -> variants
# -----------------------------

COMMON_SURNAMES = set(list(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水"
    "窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗"
    "毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧"
    "计伏成戴谈宋熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱"
    "骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫房裘缪解应宗丁宣贲邓郁单杭洪包诸左石崔吉"
    "龚程邢裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯"
    "宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹"
))

def _t2s(text: str) -> str:
    if _OPENCC is not None:
        try:
            return _OPENCC.convert(text)
        except Exception:
            return text
    return text

def _strip_trailing_parens(text: str) -> str:
    # remove trailing （...） or (...)
    return re.sub(r"[（(][^（）()]*[）)]$", "", text)

def _normalize_cjk_only(text: str) -> Optional[str]:
    t = text.strip()
    if not t:
        return None
    # remove punctuation / spaces, keep CJK
    t = re.sub(r"[·•・∙：:，,。、《》〈〉「」『』（）()［］\[\]【】—\-．.！!？?\s]+", "", t)
    if not t:
        return None
    if not all(is_cjk_char(ch) for ch in t):
        return None
    return t

def looks_like_person_name(t: str) -> bool:
    L = len(t)
    if L not in (2, 3):
        return False
    if t[0] not in COMMON_SURNAMES:
        return False
    non_person_suffixes = (
        "公司", "集团", "大学", "学院", "中学", "小学",
        "银行", "影业", "电影", "电视", "出版社", "医院",
        "乐队", "市", "区", "县",
    )
    for suf in non_person_suffixes:
        if t.endswith(suf):
            return False
    return True

def generate_wiki_variants(raw_title: str) -> List[Tuple[str, float]]:
    """
    Given a raw zh wiki title, generate (variant_text, variant_weight_factor).

    Heuristics:
      - T->S
      - strip trailing parens: 黑神话：悟空（电子游戏） -> 黑神话：悟空
      - split on colon-like & middle-dot-like separators:
          黑神话：悟空 / 黑神话·悟空 -> 黑神话, 悟空, 黑神话悟空
      - strip suffixes like 系列/公司/集团/大学/... to get root:
          寂静岭系列 -> 寂静岭
    """
    variants: List[Tuple[str, float]] = []

    t = raw_title.strip()
    if not t:
        return variants

    t = _t2s(t)
    t = _strip_trailing_parens(t)

    # base
    variants.append((t, 1.0))

    # split on colon-like / dot-like separators
    if any(ch in t for ch in ["：", ":", "·", "•", "∙"]):
        parts = re.split(r"[：:·•∙]", t, maxsplit=1)
        if len(parts) == 2:
            left, right = parts
            left = left.strip()
            right = right.strip()
            if left:
                variants.append((left, 0.8))
            if right:
                variants.append((right, 0.8))
            concat = left + right
            if concat:
                variants.append((concat, 0.9))

    # strip known suffixes on base t
    suffixes = [
        "系列", "公司", "集团", "大学", "学院", "中学", "小学",
        "游戏", "手游", "动画", "漫画", "电视剧", "电影",
        "专辑", "歌曲", "乐队", "手机", "科技集团", "科技公司",
    ]
    for suf in suffixes:
        if t.endswith(suf) and len(t) > len(suf):
            root = t[:-len(suf)]
            variants.append((root, 0.9))

    # deduplicate by text, keep max factor
    agg: Dict[str, float] = {}
    for txt, factor in variants:
        txt = txt.strip()
        if not txt:
            continue
        prev = agg.get(txt, 0.0)
        if factor > prev:
            agg[txt] = factor

    return list(agg.items())

WIKI_KEYWORDS_DEFAULT = [
    "电影", "影片", "电视剧", "动画", "漫画", "综艺",
    "游戏", "手游", "公司", "品牌", "科技", "手机",
    "专辑", "歌曲", "乐队", "高校", "大学", "学院",
    "组织", "俱乐部", "工作室",
]

def load_wiki_titles(min_len: int = 2,
                     max_len: int = 8,
                     weight: float = 1.0,
                     keyword_bonus: float = 2.0,
                     max_titles: Optional[int] = None,
                     wiki_config: str = "20231101.zh",
                     keywords: Optional[List[str]] = None,
                     drop_person_names: bool = True) -> Dict[str, float]:
    """
    Use `wikimedia/wikipedia` zh config and:
      - generate variants per title (base + abbrevs),
      - normalize to CJK-only tokens,
      - filter length & person-name,
      - weight = weight * variant_factor * (keyword_bonus if keyword matches else 1).
    """
    if not HAS_DATASETS:
        log("[WIKI] datasets package not available; skipping.")
        return {}

    if keywords is None:
        keywords = WIKI_KEYWORDS_DEFAULT

    log(f"[WIKI] Loading wikimedia/wikipedia config={wiki_config} split=train ...")
    ds = load_dataset("wikimedia/wikipedia", wiki_config, split="train")

    counter: Dict[str, float] = collections.Counter()
    n_raw = 0
    n_variants = 0
    n_kept = 0

    for ex in ds:
        title = ex.get("title", "")
        if not title:
            if max_titles and n_raw >= max_titles:
                break
            continue

        n_raw += 1
        variants = generate_wiki_variants(title)
        n_variants += len(variants)

        for raw_v, factor in variants:
            norm = _normalize_cjk_only(raw_v)
            if not norm:
                continue
            L = len(norm)
            if L < min_len or L > max_len:
                continue
            if drop_person_names and looks_like_person_name(norm):
                continue

            bonus_mult = 1.0
            for kw in keywords:
                if kw in raw_v or kw in norm:
                    bonus_mult = keyword_bonus
                    break

            score = weight * factor * bonus_mult
            counter[norm] += score
            n_kept += 1

        if max_titles is not None and n_raw >= max_titles:
            break

    log(
        f"[WIKI] Titles scanned={n_raw}, variants={n_variants}, "
        f"kept_tokens={n_kept} in [{min_len},{max_len}]"
    )
    return counter

# -----------------------------
# HF helpers (THUCNews / Weibo)
# -----------------------------

def _pick_first_string_field(ds) -> Optional[str]:
    if not hasattr(ds, "features"):
        return None
    for name, feat in ds.features.items():
        if isinstance(feat, Value) and feat.dtype == "string":
            return name
    return None

def load_thucnews_titles(min_len: int = 2,
                         max_len: int = 12,
                         weight: float = 0.7,
                         max_docs: int = 300000) -> Dict[str, float]:
    if not HAS_DATASETS:
        log("[THUC] datasets package not available; skipping.")
        return {}
    log("[THUC] Loading seamew/THUCNewsTitle ...")
    try:
        ds = load_dataset("seamew/THUCNewsTitle", split="train")
    except Exception as e:
        log(f"[THUC] Failed to load seamew/THUCNewsTitle: {e}")
        return {}

    field = _pick_first_string_field(ds)
    if not field:
        log("[THUC] No string field found; skipping seamew/THUCNewsTitle.")
        return {}

    counter: Dict[str, float] = collections.Counter()
    total = 0
    kept = 0
    for i, ex in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        text = ex.get(field, "")
        if not isinstance(text, str):
            continue
        total += 1
        phrase = normalize_phrase(text)
        if not (min_len <= len(phrase) <= max_len):
            continue
        if not is_all_cjk(phrase):
            continue
        counter[phrase] += weight
        kept += 1
    log(f"[THUC] Scanned {total} docs, kept {kept} phrases from seamew/THUCNewsTitle")
    return counter

def load_weibo_phrases(min_len: int = 2,
                       max_len: int = 8,
                       weight: float = 0.5,
                       max_docs: int = 200000) -> Dict[str, float]:
    if not HAS_DATASETS:
        log("[WEIBO] datasets package not available; skipping.")
        return {}
    log("[WEIBO] Loading seamew/Weibo ...")
    try:
        ds = load_dataset("seamew/Weibo", split="train")
    except Exception as e:
        log(f"[WEIBO] Failed to load seamew/Weibo: {e}")
        return {}

    field = _pick_first_string_field(ds)
    if not field:
        log("[WEIBO] No string field found; skipping seamew/Weibo.")
        return {}

    counter: Dict[str, float] = collections.Counter()
    total = 0
    span_kept = 0
    for i, ex in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            break
        text = ex.get(field, "")
        if not isinstance(text, str):
            continue
        total += 1
        for span in extract_cjk_spans(text):
            span = normalize_phrase(span)
            if not (min_len <= len(span) <= max_len):
                continue
            if not is_all_cjk(span):
                continue
            counter[span] += weight
            span_kept += 1
    log(f"[WEIBO] Scanned {total} posts, kept {span_kept} spans from seamew/Weibo")
    return counter

# -----------------------------
# Allowlist
# -----------------------------

def load_allowlist(path: Optional[str],
                   default_weight: float = 1e6) -> Dict[str, float]:
    """
    allowlist file:
      - one word per line, or
      - word<TAB>freq
    If freq missing, use default_weight.
    """
    if path is None:
        return {}
    if not os.path.isfile(path):
        log(f"[ALLOW] Warning: allowlist {path} not found, ignoring")
        return {}

    counter: Dict[str, float] = collections.Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            word = parts[0].strip()
            if not word:
                continue
            if len(parts) >= 2:
                try:
                    freq = float(parts[1])
                except ValueError:
                    freq = default_weight
            else:
                freq = default_weight
            counter[word] += freq
    log(f"[ALLOW] {len(counter)} words in allowlist (default weight={default_weight})")
    return counter

# -----------------------------
# Merge & write
# -----------------------------

def merge_sources(sources: List[Dict[str, float]]) -> Dict[str, float]:
    merged: Dict[str, float] = collections.Counter()
    for d in sources:
        for k, v in d.items():
            merged[k] += v
    return merged

def write_lexicon(lexicon: Dict[str, float],
                  output_path: str,
                  max_lexicon_size: Optional[int] = None) -> None:
    items = sorted(lexicon.items(), key=lambda kv: (-kv[1], kv[0]))
    if max_lexicon_size is not None:
        items = items[:max_lexicon_size]
    log(f"[WRITE] {len(items)} phrases → {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for phrase, score in items:
            f.write(f"{phrase}\t{score:.6f}\n")

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lexicon TSV for 谢彬烂梗生成器.")

    # IO + global len filter
    parser.add_argument("--output", type=str, required=True,
                        help="Output TSV path, e.g. lexicon_real.tsv")
    parser.add_argument("--min_len", type=int, default=2,
                        help="Minimum phrase length in characters.")
    parser.add_argument("--max_len", type=int, default=8,
                        help="Maximum phrase length in characters.")
    parser.add_argument("--max_lexicon_size", type=int, default=250000,
                        help="Keep at most this many phrases in final lexicon.")

    # THUOCL
    parser.add_argument("--thuocl_dir", type=str,
                        default=os.path.join("corpus", "THUOCL"),
                        help="Directory containing THUOCL_*.txt files.")
    parser.add_argument("--thuocl_weight", type=float, default=1.0,
                        help="Weight for THUOCL frequencies.")

    # Wikipedia-zh titles
    parser.add_argument("--use_wiki", action="store_true", default=True,
                        help="Use wikimedia/wikipedia zh titles.")
    parser.add_argument("--wiki_weight", type=float, default=1.0,
                        help="Base weight for each Wikipedia title variant.")
    parser.add_argument("--wiki_keyword_bonus", type=float, default=2.5,
                        help="Multiplier when a pop-culture keyword matches.")
    parser.add_argument("--wiki_max_titles", type=int, default=None,
                        help="Optional cap on number of wiki titles to scan.")
    parser.add_argument("--wiki_config", type=str, default="20231101.zh",
                        help="Config name for wikimedia/wikipedia dataset.")
    parser.add_argument("--wiki_drop_person_names", action="store_true", default=True,
                        help="Drop likely Chinese person names from wiki variants.")
    parser.add_argument("--wiki_no_drop_person_names",
                        dest="wiki_drop_person_names", action="store_false")

    # THUCNews
    parser.add_argument("--use_thucnews", action="store_true", default=False,
                        help="Use seamew/THUCNewsTitle (if loadable).")
    parser.add_argument("--thucnews_weight", type=float, default=0.7,
                        help="Weight for each THUCNews title.")
    parser.add_argument("--thucnews_max_docs", type=int, default=300000,
                        help="Max docs to scan from THUCNewsTitle.")
    parser.add_argument("--thucnews_max_len", type=int, default=12,
                        help="Max length for THUCNews titles.")

    # Weibo
    parser.add_argument("--use_weibo", action="store_true", default=False,
                        help="Use seamew/Weibo and mine short CJK spans, if loadable.")
    parser.add_argument("--weibo_weight", type=float, default=0.5,
                        help="Weight for each Weibo phrase occurrence.")
    parser.add_argument("--weibo_max_docs", type=int, default=200000,
                        help="Max posts to scan from Weibo.")
    parser.add_argument("--weibo_min_len", type=int, default=2,
                        help="Min length for Weibo spans.")
    parser.add_argument("--weibo_max_len", type=int, default=8,
                        help="Max length for Weibo spans.")

    # Allowlist
    parser.add_argument("--allowlist", type=str, default=None,
                        help="Allowlist file (word[\\t freq]); force-include with high weight.")
    parser.add_argument("--allow_weight", type=float, default=1e6,
                        help="Default weight for allowlist entries without freq.")

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    sources: List[Dict[str, float]] = []

    # THUOCL
    if args.thuocl_dir and os.path.isdir(args.thuocl_dir):
        thuocl = load_thuocl(
            thuocl_root=args.thuocl_dir,
            min_len=args.min_len,
            max_len=args.max_len,
            weight=args.thuocl_weight,
        )
        sources.append(thuocl)
    else:
        log(f"[THUOCL] Directory not found, skipping: {args.thuocl_dir}")

    # WIKI
    if args.use_wiki:
        wiki = load_wiki_titles(
            min_len=args.min_len,
            max_len=args.max_len,
            weight=args.wiki_weight,
            keyword_bonus=args.wiki_keyword_bonus,
            max_titles=args.wiki_max_titles,
            wiki_config=args.wiki_config,
            drop_person_names=args.wiki_drop_person_names,
        )
        sources.append(wiki)

    # THUCNews
    if args.use_thucnews:
        thuc = load_thucnews_titles(
            min_len=args.min_len,
            max_len=args.thucnews_max_len,
            weight=args.thucnews_weight,
            max_docs=args.thucnews_max_docs,
        )
        sources.append(thuc)

    # Weibo
    if args.use_weibo:
        weibo = load_weibo_phrases(
            min_len=args.weibo_min_len,
            max_len=args.weibo_max_len,
            weight=args.weibo_weight,
            max_docs=args.weibo_max_docs,
        )
        sources.append(weibo)

    # Allowlist
    allow = load_allowlist(args.allowlist, default_weight=args.allow_weight)
    if allow:
        sources.append(allow)

    if not sources:
        log("[WARN] No sources enabled; nothing to write.")
        return

    merged = merge_sources(sources)
    write_lexicon(
        merged,
        output_path=args.output,
        max_lexicon_size=args.max_lexicon_size,
    )

if __name__ == "__main__":
    main()
