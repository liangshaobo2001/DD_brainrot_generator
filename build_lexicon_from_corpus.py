#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build a phrase lexicon for 谢彬烂梗生成器.

Sources (all optional, controlled by CLI flags):

1. THUOCL (local, already downloaded)
2. Wikipedia-zh titles via HuggingFace `wikimedia/wikipedia`
3. THUCNews titles via HuggingFace `seamew/THUCNewsTitle`
4. Weibo short phrases via HuggingFace `seamew/Weibo`

Each source produces a dict: phrase -> score.
Scores are summed, then we keep top-N phrases and write TSV: `phrase\tfreq`.

You can control:
- weights per source
- max lexicon size
- min/max phrase length (in characters)
"""

import argparse
import collections
import math
import os
import re
import sys
from typing import Dict, Iterable, List, Tuple

try:
    from datasets import load_dataset, Value
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False


# -----------------------------
# Utils
# -----------------------------

CJK_RE = re.compile(r'[\u4e00-\u9fff]+')

def is_cjk_char(ch: str) -> bool:
    return '\u4e00' <= ch <= '\u9fff'

def is_all_cjk(s: str) -> bool:
    s = s.strip()
    return bool(s) and all(is_cjk_char(ch) for ch in s)

def normalize_phrase(s: str) -> str:
    # Very light normalization for now.
    return s.strip()

def extract_cjk_spans(text: str) -> Iterable[str]:
    """Yield contiguous CJK spans from text."""
    for m in CJK_RE.finditer(text):
        span = m.group(0)
        if span:
            yield span

def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# -----------------------------
# THUOCL loader
# -----------------------------

def load_thuocl(thuocl_root: str,
                min_len: int = 2,
                max_len: int = 8,
                weight: float = 1.0) -> Dict[str, float]:
    """
    Load THUOCL files from a directory.

    Expected structure (what you currently have):

        corpus/THUOCL/
            THUOCL_IT.txt
            THUOCL_animal.txt
            ...

    Each file: "phrase<TAB>freq"
    """
    data_dir = thuocl_root
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"THUOCL directory not found: {data_dir}")

    counter: Dict[str, float] = collections.Counter()

    files = [f for f in os.listdir(data_dir) if f.startswith("THUOCL_") and f.endswith(".txt")]
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
                # Use log frequency as base, scaled by weight
                score = weight * math.log1p(freq)
                counter[phrase] += score

    log(f"[THUOCL] {len(counter)} unique phrases after filter")
    return counter


# -----------------------------
# Wikipedia-zh titles
# -----------------------------

WIKI_KEYWORDS_DEFAULT = [
    "电影", "影片", "电视剧", "动画", "漫画", "综艺",
    "游戏", "手游", "公司", "品牌", "科技", "手机",
    "专辑", "歌曲", "乐队", "高校", "大学", "学院",
    "组织", "俱乐部"
]

def load_wiki_titles(min_len: int = 2,
                     max_len: int = 8,
                     weight: float = 1.0,
                     keyword_bonus: float = 2.0,
                     max_titles: int = None,
                     wiki_config: str = "20231101.zh",
                     keywords: List[str] = None) -> Dict[str, float]:
    """
    Use HuggingFace `wikimedia/wikipedia` dataset, zh config, and
    take the `title` field as phrase.

        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")

    We keep titles:
      - length in [min_len, max_len]
      - composed of CJK chars (loose heuristic: at least half CJK)
    """
    if not HAS_DATASETS:
        log("[WIKI] datasets package not available; skipping.")
        return {}

    if keywords is None:
        keywords = WIKI_KEYWORDS_DEFAULT

    log(f"[WIKI] Loading wikimedia/wikipedia config={wiki_config} split=train ...")
    ds = load_dataset("wikimedia/wikipedia", wiki_config, split="train")

    counter: Dict[str, float] = collections.Counter()
    total = 0
    kept = 0
    for i, ex in enumerate(ds):
        title = ex.get("title", "")
        if not title:
            continue
        total += 1

        title = normalize_phrase(title)
        if not (min_len <= len(title) <= max_len):
            continue

        # require majority of characters to be CJK
        cjk_count = sum(1 for ch in title if is_cjk_char(ch))
        if cjk_count < len(title) * 0.5:
            continue

        score = weight
        if any(k in title for k in keywords):
            score += keyword_bonus

        counter[title] += score
        kept += 1

        if max_titles is not None and kept >= max_titles:
            break

    log(f"[WIKI] Scanned {total} titles, kept {kept} in [{min_len},{max_len}]")
    return counter


# -----------------------------
# Generic HF text-field loader
# -----------------------------

def _pick_first_string_field(ds) -> str:
    """
    Inspect HuggingFace dataset features and pick the first string field.
    This keeps us robust to unknown schemas (THUCNewsTitle / Weibo, etc.).
    """
    if not hasattr(ds, "features"):
        return None
    for name, feat in ds.features.items():
        # Value for primitive types
        if isinstance(feat, Value) and feat.dtype == "string":
            return name
    return None


# -----------------------------
# THUCNews titles (HF: seamew/THUCNewsTitle)
# -----------------------------

def load_thucnews_titles(min_len: int = 2,
                         max_len: int = 12,
                         weight: float = 0.7,
                         max_docs: int = 300000) -> Dict[str, float]:
    """
    Load short Chinese news titles from HuggingFace `seamew/THUCNewsTitle`.

    We treat each title as a phrase candidate if:
      - length in [min_len, max_len]
      - all CJK
    """
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


# -----------------------------
# Weibo short phrases (HF: seamew/Weibo)
# -----------------------------

def load_weibo_phrases(min_len: int = 2,
                       max_len: int = 8,
                       weight: float = 0.5,
                       max_docs: int = 200000) -> Dict[str, float]:
    """
    Load Weibo data from HuggingFace `seamew/Weibo` and mine short CJK phrases.

    Strategy:
      - Load the dataset (train split by default).
      - Identify a string field via dataset.features.
      - For each post:
          * Extract contiguous CJK spans using regex.
          * For each span with length in [min_len, max_len], add weight.
      - This gives us some internet slang / short proper nouns without heavy NLP.

    NOTE: This is intentionally conservative to keep runtime manageable.
    """
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

        # Extract short CJK spans as candidate phrases
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
                  max_lexicon_size: int = None) -> None:
    # Sort by score desc, then lexicographically
    items = sorted(lexicon.items(), key=lambda kv: (-kv[1], kv[0]))
    if max_lexicon_size is not None:
        items = items[:max_lexicon_size]

    log(f"[WRITE] {len(items)} phrases → {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for phrase, score in items:
            # Downstream uses this as a "freq-like" value; keeping float is fine.
            f.write(f"{phrase}\t{score:.6f}\n")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lexicon TSV for 谢彬烂梗生成器.")

    # Core IO
    parser.add_argument("--output", type=str, required=True,
                        help="Output TSV path, e.g. lexicon_real.tsv")

    # Phrase length filter (global)
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
                        help="Base weight for each Wikipedia title.")
    parser.add_argument("--wiki_keyword_bonus", type=float, default=2.5,
                        help="Bonus weight if title contains certain pop-culture keywords.")
    parser.add_argument("--wiki_max_titles", type=int, default=None,
                        help="Optional cap on number of titles to scan.")
    parser.add_argument("--wiki_config", type=str, default="20231101.zh",
                        help="Config name for wikimedia/wikipedia dataset.")

    # THUCNews titles
    parser.add_argument("--use_thucnews", action="store_true", default=False,
                        help="Use seamew/THUCNewsTitle (news titles).")
    parser.add_argument("--thucnews_weight", type=float, default=0.7,
                        help="Weight for each THUCNews title.")
    parser.add_argument("--thucnews_max_docs", type=int, default=300000,
                        help="Max docs to scan from THUCNewsTitle.")
    parser.add_argument("--thucnews_max_len", type=int, default=12,
                        help="Max length for THUCNews titles (some are longer than 8).")

    # Weibo
    parser.add_argument("--use_weibo", action="store_true", default=False,
                        help="Use seamew/Weibo and mine short CJK spans as phrases.")
    parser.add_argument("--weibo_weight", type=float, default=0.5,
                        help="Weight for each Weibo phrase occurrence.")
    parser.add_argument("--weibo_max_docs", type=int, default=200000,
                        help="Max posts to scan from Weibo.")
    parser.add_argument("--weibo_min_len", type=int, default=2,
                        help="Min length for Weibo spans.")
    parser.add_argument("--weibo_max_len", type=int, default=8,
                        help="Max length for Weibo spans.")

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

    # Wikipedia titles
    if args.use_wiki:
        wiki = load_wiki_titles(
            min_len=args.min_len,
            max_len=args.max_len,
            weight=args.wiki_weight,
            keyword_bonus=args.wiki_keyword_bonus,
            max_titles=args.wiki_max_titles,
            wiki_config=args.wiki_config,
        )
        sources.append(wiki)

    # THUCNews titles
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
