#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a unified lexicon TSV from:
  1) THUOCL files under ./corpus/THUOCL/*.txt or ./corpus/THUOCL/data/*.txt (phrase \t freq)
  2) Chinese Wikipedia titles from Hugging Face (wikimedia/wikipedia)

Output:
    lexicon_real.tsv     # phrase<TAB>freq (float), aggregated & sorted by freq desc

Improvements vs previous version:
- Popularity proxy for Wikipedia titles: base weight * keyword bonus
- Aggressive filtering:
  * Keep only fully-CJK, length in [min_len, max_len]
  * Drop titles that look like Chinese personal names (2/3-char 姓名) unless whitelisted
  * Keep pop-culture by keyword bonus (电影/电视剧/游戏/公司/品牌/手机/漫画/动画/综艺/歌曲/专辑/高校/大学 等)
- Size control: --max_lexicon_size to cap final TSV size
- THUOCL kept with their numeric frequencies (strong prior)

Tip:
- You can crank up --wiki_weight and keyword bonuses if you want even more pop-culture.
"""

import argparse
import collections
import pathlib
import re
from typing import Dict, Iterable, Tuple, Optional, List

# Optional OpenCC (Traditional->Simplified). Script still runs without it.
try:
    from opencc import OpenCC  # type: ignore
    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_THUOCL_DIR = ROOT / "corpus" / "THUOCL"
DEFAULT_OUTPUT = ROOT / "lexicon_real.tsv"

CJK_ONLY = re.compile(r"^[\u4e00-\u9fff]+$")

# A compact common-surname set (covers the vast majority; we don't need 500+).
COMMON_SURNAMES = set(list("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫柯房裘缪解应宗丁宣贲邓郁单杭洪包诸左石崔吉龚程邢裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹车侯邱") )

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
    print(f"[THUOCL] {len(counter)} unique phrases")
    return counter

# ----------------------------
# Wikipedia titles (Hugging Face)
# ----------------------------

def _normalize_title(title: str) -> Optional[str]:
    if not title:
        return None
    t = title.strip()
    if not t:
        return None
    if _OPENCC is not None:
        try:
            t = _OPENCC.convert(t)
        except Exception:
            pass
    # strip punctuation/symbols/spaces
    t = re.sub(r"[·•・∙：:，,。、《》〈〉「」『』（）()［］\[\]【】—\-．.！!？?\s]+", "", t)
    if not CJK_ONLY.match(t):
        return None
    return t


def looks_like_person_name(t: str) -> bool:
    """
    Simple heuristic:
      - 2-char or 3-char fully-CJK
      - starts with a common surname
      - not obviously an org/work suffix
    """
    L = len(t)
    if L not in (2, 3):
        return False
    if t[0] not in COMMON_SURNAMES:
        return False
    # whitelist some typical non-person endings (e.g., 电影, 公司, 大学, 市, 区)
    non_person_suffixes = ("公司", "大学", "学院", "医院", "影业", "电影", "电视", "出版社", "银行", "乐队", "集团", "中学", "小学", "市", "区", "县")
    for suf in non_person_suffixes:
        if t.endswith(suf):
            return False
    return True


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
    Load Chinese Wikipedia titles (wikimedia/wikipedia on HF) and return a popularity-weighted dict.

    Popularity proxy:
      score = base_weight * (keyword_bonus if any keyword matches else 1.0)

    Filtering:
      - fully CJK, length [min_len, max_len]
      - drop person-name-like titles (2/3-char 姓名) if drop_person_names=True
    """
    if keywords is None:
        keywords = ["电影", "影片", "电视剧", "动画", "漫画", "综艺", "游戏", "手游", "公司", "品牌", "科技", "手机", "专辑", "歌曲", "乐队", "高校", "大学", "学院", "组织", "俱乐部"]

    print(f"[WIKI] Loading wikimedia/wikipedia config={wiki_config} split={split} ...")
    from datasets import load_dataset  # local import
    ds = load_dataset("wikimedia/wikipedia", wiki_config, split=split)

    counter: Dict[str, float] = collections.Counter()
    n_total = 0
    n_kept = 0

    for ex in ds:
        n_total += 1
        title = ex.get("title", "")
        t = _normalize_title(title)
        if not t:
            if limit and n_total >= limit: break
            continue
        L = len(t)
        if L < min_len or L > max_len:
            if limit and n_total >= limit: break
            continue

        if drop_person_names and looks_like_person_name(t):
            if limit and n_total >= limit: break
            continue

        # popularity proxy by keyword bonus
        bonus = 1.0
        for kw in keywords:
            if kw in t:
                bonus = keyword_bonus
                break

        counter[t] += base_weight * bonus
        n_kept += 1

        if limit and n_total >= limit:
            break

    print(f"[WIKI] Scanned {n_total} titles, kept {n_kept} in [{min_len},{max_len}], bonus keywords={keywords}")
    return counter

# ----------------------------
# Merge & write
# ----------------------------

def merge_and_write(
    thuocl: Dict[str, float],
    wiki: Dict[str, float],
    out_path: pathlib.Path,
    min_freq: float = 1.0,
    top_k: Optional[int] = None,
    max_lexicon_size: Optional[int] = None,
):
    agg: Dict[str, float] = collections.Counter()
    agg.update(thuocl)
    # Counter.update adds values
    for k, v in wiki.items():
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
        description="Build lexicon_real.tsv from THUOCL + Chinese Wikipedia titles with filtering & popularity proxy."
    )
    ap.add_argument("--thuocl_dir", type=pathlib.Path, default=DEFAULT_THUOCL_DIR,
                    help=f"THUOCL dir (default: {DEFAULT_THUOCL_DIR})")
    ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT,
                    help=f"Output TSV path (default: {DEFAULT_OUTPUT})")

    # phrase length guardrails
    ap.add_argument("--min_len", type=int, default=2, help="Min phrase length (default 2)")
    ap.add_argument("--max_len", type=int, default=8, help="Max phrase length (default 8)")

    # weights
    ap.add_argument("--thuocl_weight", type=float, default=1.0, help="Weight for THUOCL freqs (default 1.0)")
    ap.add_argument("--wiki_weight", type=float, default=1.0, help="Base weight per Wikipedia title (default 1.0)")
    ap.add_argument("--wiki_keyword_bonus", type=float, default=2.0, help="Multiplier when a keyword matches (default 2.0)")

    # HF wikipedia dataset options
    ap.add_argument("--wiki_config", type=str, default="20231101.zh",
                    help='wikimedia/wikipedia config (e.g. "20231101.zh")')
    ap.add_argument("--wiki_split", type=str, default="train", help='split (default "train")')
    ap.add_argument("--wiki_limit", type=int, default=None, help="Process first N titles only (debug fast-run)")

    # filters
    ap.add_argument("--drop_person_names", action="store_true", default=True,
                    help="Drop likely Chinese personal names (2/3-char 姓名). Default True.")
    ap.add_argument("--no_drop_person_names", dest="drop_person_names", action="store_false")

    # final pruning
    ap.add_argument("--min_freq", type=float, default=1.0, help="Min frequency in merged output")
    ap.add_argument("--top_k", type=int, default=None, help="Keep only top-K items")
    ap.add_argument("--max_lexicon_size", type=int, default=300000,
                    help="Hard cap on final lexicon size (default 300k)")

    args = ap.parse_args()

    # Load THUOCL
    thuocl_all = load_thuocl(args.thuocl_dir, weight=args.thuocl_weight)

    # Filter THUOCL by length/CJK
    thuocl_len_filtered: Dict[str, float] = collections.Counter()
    for w, f in thuocl_all.items():
        if args.min_len <= len(w) <= args.max_len and CJK_ONLY.match(w):
            thuocl_len_filtered[w] += f
    print(f"[THUOCL] After length/CJK filter: {len(thuocl_len_filtered)}")

    # Load Wikipedia titles (Hugging Face) with popularity proxy
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
        out_path=args.output,
        min_freq=args.min_freq,
        top_k=args.top_k,
        max_lexicon_size=args.max_lexicon_size,
    )


if __name__ == "__main__":
    main()
