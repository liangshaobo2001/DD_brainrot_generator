#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a unified lexicon TSV from:
  1) THUOCL files under ./corpus/THUOCL/*.txt or ./corpus/THUOCL/data/*.txt (phrase \t freq)
  2) Chinese Wikipedia titles from Hugging Face (wikimedia/wikipedia), with:
       - abbreviation / substring mining (e.g. 寂静岭系列 -> 寂静岭;
         黑神话：悟空 -> 黑神话悟空 / 黑神话 / 悟空)
  3) Optional extra TSV lexicons (dictionaries / Baike / brand lists)
  4) Optional allowlist (manual words you *must* include, e.g. 舒肤佳)

Output:
    lexicon_real.tsv     # phrase<TAB>freq (float), aggregated & sorted by freq desc

Key features:
- Keep only fully-CJK phrases, length in [min_len, max_len].
- Drop likely Chinese person names (2/3-char 姓名) unless disabled.
- Popularity proxy for wiki titles:
    base_weight * (keyword_bonus if keyword matches else 1)
  plus smaller weights for abbreviation variants.
- Size control via --max_lexicon_size.
"""

import argparse
import collections
import pathlib
import re
from typing import Dict, Iterable, Tuple, Optional, List

# Optional OpenCC (Traditional->Simplified)
try:
    from opencc import OpenCC  # type: ignore
    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_THUOCL_DIR = ROOT / "corpus" / "THUOCL"
DEFAULT_OUTPUT = ROOT / "lexicon_real.tsv"

CJK_ONLY = re.compile(r"^[\u4e00-\u9fff]+$")

# A compact surname set for person-name filtering
COMMON_SURNAMES = set(list(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水"
    "窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗"
    "毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧"
    "计伏成戴谈宋熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱"
    "骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫房裘缪解应宗丁宣贲邓郁单杭洪包诸左石崔吉"
    "龚程邢裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯"
    "宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹"
))

# ----------------------------
# THUOCL parsing
# ----------------------------

def parse_thuocl_file(path: pathlib.Path) -> Iterable[Tuple[str, int]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            freq_str = parts[-1]
            phrase = "".join(parts[:-1])
            try:
                freq = int(freq_str)
            except ValueError:
                continue
            if not phrase:
                continue
            yield phrase, freq


def _detect_thuocl_txt_files(thuocl_dir: pathlib.Path) -> List[pathlib.Path]:
    if not thuocl_dir.exists():
        raise FileNotFoundError(f"THUOCL directory not found: {thuocl_dir}")
    direct = sorted(thuocl_dir.glob("THUOCL_*.txt"))
    data_sub = sorted((thuocl_dir / "data").glob("THUOCL_*.txt"))
    if direct:
        return direct
    if data_sub:
        return data_sub
    any_direct = sorted(thuocl_dir.glob("*.txt"))
    any_data = sorted((thuocl_dir / "data").glob("*.txt"))
    files = any_direct or any_data
    if files:
        return files
    raise FileNotFoundError(f"No THUOCL .txt files in {thuocl_dir} or {thuocl_dir/'data'}")


def load_thuocl(thuocl_dir: pathlib.Path, weight: float = 1.0) -> Dict[str, float]:
    files = _detect_thuocl_txt_files(thuocl_dir)
    counter: Dict[str, float] = collections.Counter()
    for path in files:
        print(f"[THUOCL] Loading {path} ...")
        for phrase, freq in parse_thuocl_file(path):
            if not phrase:
                continue
            counter[phrase] += float(freq) * weight
    print(f"[THUOCL] {len(counter)} unique phrases (raw)")
    return counter

# ----------------------------
# Extra TSV lexicons (dictionary / Baike / brand lists)
# ----------------------------

def load_extra_tsv(paths: List[pathlib.Path], default_freq: float = 1.0) -> Dict[str, float]:
    """
    Generic TSV: word[\\t freq]
    If freq missing, use default_freq.
    """
    counter: Dict[str, float] = collections.Counter()
    for p in paths:
        if not p.exists():
            print(f"[EXTRA] Warning: {p} does not exist, skipping")
            continue
        print(f"[EXTRA] Loading {p} ...")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                word = parts[0]
                if len(parts) >= 2:
                    try:
                        freq = float(parts[1])
                    except ValueError:
                        freq = default_freq
                else:
                    freq = default_freq
                counter[word] += freq
    print(f"[EXTRA] {len(counter)} unique phrases from extra TSVs")
    return counter

# ----------------------------
# Allowlist
# ----------------------------

def load_allowlist(path: Optional[pathlib.Path], allow_weight: float = 1e6) -> Dict[str, float]:
    """
    allowlist file:
      - one word per line, or
      - word<TAB>freq
    If freq missing, use allow_weight.
    """
    if path is None:
        return {}
    if not path.exists():
        print(f"[ALLOW] Warning: allowlist {path} not found, ignoring")
        return {}

    counter: Dict[str, float] = collections.Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            word = parts[0]
            if len(parts) >= 2:
                try:
                    freq = float(parts[1])
                except ValueError:
                    freq = allow_weight
            else:
                freq = allow_weight
            counter[word] += freq

    print(f"[ALLOW] {len(counter)} words in allowlist (default weight={allow_weight})")
    return counter

# ----------------------------
# Wikipedia titles (Hugging Face)
# ----------------------------

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
    if not CJK_ONLY.match(t):
        return None
    return t

def looks_like_person_name(t: str) -> bool:
    L = len(t)
    if L not in (2, 3):
        return False
    if t[0] not in COMMON_SURNAMES:
        return False
    non_person_suffixes = (
        "公司", "集团", "大学", "学院", "中学", "小学", "银行",
        "影业", "电影", "电视", "出版社", "医院", "乐队", "市", "区", "县",
    )
    for suf in non_person_suffixes:
        if t.endswith(suf):
            return False
    return True

def generate_wiki_variants(raw_title: str) -> List[Tuple[str, float]]:
    """
    Generate (variant_text, variant_weight_factor) pairs from a single raw wiki title.

    Heuristics:
      - Convert T->S.
      - Strip trailing parens (years, qualifiers).
      - Split on colon-like / separator chars (： : · • ∙) to get left/right + concatenation.
      - Strip suffixes like 系列/公司/集团/大学/学院/游戏/动画/漫画/专辑/歌曲/手机/...
    """
    variants: List[Tuple[str, float]] = []

    t = raw_title.strip()
    if not t:
        return variants

    # 1) normalize
    t = _t2s(t)
    t = _strip_trailing_parens(t)

    # base variant: full title (before punctuation normalization)
    variants.append((t, 1.0))

    # 2) split on colon-like / middle-dot-like separators
    # e.g. "黑神话·悟空" or "黑神话：悟空"
    if any(ch in t for ch in ["：", ":", "·", "•", "∙"]):
        parts = re.split(r"[：:·•∙]", t, maxsplit=1)
        if len(parts) == 2:
            left, right = parts
            left = left.strip()
            right = right.strip()
            if left:
                # 比完整标题略低一点
                variants.append((left, 0.8))
            if right:
                variants.append((right, 0.8))
            # 拼接左+右（去掉符号）
            concat = left + right
            if concat:
                variants.append((concat, 0.9))

    # 3) known suffix-stripping on the base normalized title
    #    e.g. "寂静岭系列" -> "寂静岭"
    suffixes = [
        "系列", "公司", "集团", "大学", "学院", "中学", "小学",
        "游戏", "手游", "动画", "漫画", "电视剧", "电影",
        "专辑", "歌曲", "乐队", "手机", "科技集团", "科技公司",
    ]
    for suf in suffixes:
        if t.endswith(suf) and len(t) > len(suf):
            root = t[:-len(suf)]
            variants.append((root, 0.9))

    # de-duplicate by keeping the max factor per text
    agg: Dict[str, float] = {}
    for txt, factor in variants:
        txt = txt.strip()
        if not txt:
            continue
        prev = agg.get(txt, 0.0)
        if factor > prev:
            agg[txt] = factor

    return list(agg.items())

def load_wiki_titles(
    wiki_config: str = "20231101.zh",
    split: str = "train",
    min_len: int = 2,
    max_len: int = 8,
    base_weight: float = 1.0,
    keyword_bonus: float = 2.0,
    keywords: Optional[List[str]] = None,
    drop_person_names: bool = True,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    """
    Load Chinese Wikipedia titles (wikimedia/wikipedia on HF), with abbreviation variants.

    For each raw title:
      - generate variants (base + abbrevs)
      - normalize to CJK-only tokens
      - filter by length and person-name heuristic
      - weight = base_weight * variant_factor * (keyword_bonus if keyword matches else 1)
    """
    if keywords is None:
        keywords = [
            "电影", "影片", "电视剧", "动画", "漫画", "综艺",
            "游戏", "手游", "公司", "品牌", "科技", "手机",
            "专辑", "歌曲", "乐队", "高校", "大学", "学院",
            "组织", "俱乐部", "工作室",
        ]

    print(f"[WIKI] Loading wikimedia/wikipedia config={wiki_config} split={split} ...")
    from datasets import load_dataset  # local import
    ds = load_dataset("wikimedia/wikipedia", wiki_config, split=split)

    counter: Dict[str, float] = collections.Counter()
    n_raw = 0
    n_variants = 0
    n_kept = 0

    for ex in ds:
        n_raw += 1
        raw_title = ex.get("title", "")
        if not raw_title:
            if limit and n_raw >= limit:
                break
            continue

        variants = generate_wiki_variants(raw_title)
        n_variants += len(variants)

        for raw_v, factor in variants:
            # normalize to CJK-only
            norm = _normalize_cjk_only(raw_v)
            if not norm:
                continue
            L = len(norm)
            if L < min_len or L > max_len:
                continue
            if drop_person_names and looks_like_person_name(norm):
                continue

            bonus = 1.0
            for kw in keywords:
                if kw in raw_v or kw in norm:
                    bonus = keyword_bonus
                    break

            weight = base_weight * factor * bonus
            counter[norm] += weight
            n_kept += 1

        if limit and n_raw >= limit:
            break

    print(
        f"[WIKI] Titles scanned={n_raw}, variants={n_variants}, "
        f"kept_tokens={n_kept} in [{min_len},{max_len}]"
    )
    return counter

# ----------------------------
# Merge & write
# ----------------------------

def merge_and_write(
    thuocl: Dict[str, float],
    wiki: Dict[str, float],
    extra: Dict[str, float],
    allow: Dict[str, float],
    out_path: pathlib.Path,
    min_freq: float = 1.0,
    top_k: Optional[int] = None,
    max_lexicon_size: Optional[int] = None,
):
    agg: Dict[str, float] = collections.Counter()
    agg.update(thuocl)
    for k, v in wiki.items():
        agg[k] += v
    for k, v in extra.items():
        agg[k] += v
    for k, v in allow.items():
        agg[k] += v

    items = [(w, f) for w, f in agg.items() if f >= min_freq]
    items.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        items = items[:top_k]
    if max_lexicon_size is not None:
        items = items[:max_lexicon_size]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for w, f in items:
            out.write(f"{w}\t{f:.6f}\n")

    print(f"[WRITE] {len(items)} phrases → {out_path}")

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build lexicon_real.tsv from THUOCL + Wikipedia titles + optional extra corpora & allowlist."
    )
    ap.add_argument("--thuocl_dir", type=pathlib.Path, default=DEFAULT_THUOCL_DIR,
                    help=f"THUOCL dir (default: {DEFAULT_THUOCL_DIR})")
    ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT,
                    help=f"Output TSV path (default: {DEFAULT_OUTPUT})")

    # phrase length guardrails
    ap.add_argument("--min_len", type=int, default=2, help="Min phrase length (default 2)")
    ap.add_argument("--max_len", type=int, default=8, help="Max phrase length (default 8)")

    # weights
    ap.add_argument("--thuocl_weight", type=float, default=1.0, help="Weight for THUOCL freqs")
    ap.add_argument("--wiki_weight", type=float, default=1.0, help="Base weight per wiki title")
    ap.add_argument("--wiki_keyword_bonus", type=float, default=2.5,
                    help="Multiplier when a keyword matches (default 2.5)")

    # HF wikipedia dataset options
    ap.add_argument("--wiki_config", type=str, default="20231101.zh",
                    help='wikimedia/wikipedia config (e.g. "20231101.zh")')
    ap.add_argument("--wiki_split", type=str, default="train", help='split (default "train")')
    ap.add_argument("--wiki_limit", type=int, default=None, help="Process first N titles only (debug fast-run)")

    # extra TSVs & allowlist
    ap.add_argument("--extra_tsv", type=pathlib.Path, nargs="*", default=[],
                    help="Additional TSV lexicons (word[\\t freq])")
    ap.add_argument("--extra_default_freq", type=float, default=1.0,
                    help="Default freq if extra TSV rows lack freq")
    ap.add_argument("--allowlist", type=pathlib.Path, default=None,
                    help="Allowlist file (word[\\t freq]); will be force-included with high weight")
    ap.add_argument("--allow_weight", type=float, default=1e6,
                    help="Default weight for allowlist entries lacking freq")

    # filters
    ap.add_argument("--drop_person_names", action="store_true", default=True,
                    help="Drop likely Chinese personal names (2/3-char 姓名). Default True.")
    ap.add_argument("--no_drop_person_names", dest="drop_person_names", action="store_false")

    # final pruning
    ap.add_argument("--min_freq", type=float, default=1.0, help="Min frequency in merged output")
    ap.add_argument("--top_k", type=int, default=None, help="Keep only top-K items")
    ap.add_argument("--max_lexicon_size", type=int, default=250000,
                    help="Hard cap on final lexicon size (default 250k)")

    args = ap.parse_args()

    # THUOCL
    thuocl_all = load_thuocl(args.thuocl_dir, weight=args.thuocl_weight)
    thuocl_len_filtered: Dict[str, float] = collections.Counter()
    for w, f in thuocl_all.items():
        if args.min_len <= len(w) <= args.max_len and CJK_ONLY.match(w):
            thuocl_len_filtered[w] += f
    print(f"[THUOCL] After length/CJK filter: {len(thuocl_len_filtered)}")

    # Extra TSV corpora (dictionary / Baike / brands)
    extra_lex = {}
    if args.extra_tsv:
        extra_lex = load_extra_tsv(args.extra_tsv, default_freq=args.extra_default_freq)
    else:
        print("[EXTRA] No extra TSVs specified")

    # Allowlist
    allow = load_allowlist(args.allowlist, allow_weight=args.allow_weight)

    # WIKI titles + abbreviation variants
    wiki_titles = load_wiki_titles(
        wiki_config=args.wiki_config,
        split=args.wiki_split,
        min_len=args.min_len,
        max_len=args.max_len,
        base_weight=args.wiki_weight,
        keyword_bonus=args.wiki_keyword_bonus,
        drop_person_names=args.drop_person_names,
        limit=args.wiki_limit,
    )

    # Merge and write
    merge_and_write(
        thuocl=thuocl_len_filtered,
        wiki=wiki_titles,
        extra=extra_lex,
        allow=allow,
        out_path=args.output,
        min_freq=args.min_freq,
        top_k=args.top_k,
        max_lexicon_size=args.max_lexicon_size,
    )


if __name__ == "__main__":
    main()
