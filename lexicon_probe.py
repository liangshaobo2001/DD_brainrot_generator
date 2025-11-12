#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lexicon probe utilities for 谢彬烂梗生成器.

Features:
  1) Exact lookup of words in a TSV lexicon (word<TAB>freq).
  2) Optional pinyin print (uses pypinyin if available; else a deterministic fallback).
  3) Super-substring check: does TARGET contain QUERY's full pinyin sequence (superset)?
     - shows align position (prefix/suffix/middle), len_diff, and a quick pass/fail.

Usage examples:
  # A) Exact lookup
  python lexicon_probe.py --lex lexicon_real.tsv --lookup 舒肤佳 寂静岭 黑神话悟空

  # B) Lookup with pinyin
  python lexicon_probe.py --lex lexicon_real.tsv --lookup 舒肤佳 --show-pinyin

  # C) Super check (would TARGET be a 'super' match for QUERY under Model 1 rules?)
  python lexicon_probe.py --lex lexicon_real.tsv --check-super --query 舒服 --target 舒肤佳

  # D) Batch check multiple targets against one query
  python lexicon_probe.py --lex lexicon_real.tsv --check-super --query 舒服 --target 舒肤佳 描述符 特殊符号

Returns non-zero exit when any requested lookup is missing (handy for CI).
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple

# --------- Pinyin helpers (match xiebin_model1_pilot) ---------
USE_DUMMY_PINYIN = False
try:
    from pypinyin import pinyin as _py, Style  # type: ignore
except Exception:
    USE_DUMMY_PINYIN = True
    _py = None
    Style = None

def _norm_syllable(s: str) -> str:
    s = s.lower()
    return "".join(ch for ch in s if not ch.isdigit())

def to_pinyin_list(text: str) -> List[str]:
    if not text:
        return []
    if not USE_DUMMY_PINYIN and _py is not None:
        syls = _py(text, style=Style.TONE3, strict=False)
        return [_norm_syllable(x[0]) for x in syls]
    # deterministic fallback
    return [f"p{ord(ch)%97:02d}" for ch in text]

def find_subseq_start(seq: List[str], sub: List[str]) -> Optional[int]:
    n, m = len(seq), len(sub)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if seq[i:i+m] == sub:
            return i
    return None

def pos_type(start: int, end: int, total_len: int) -> str:
    if start == 0 and end == total_len:
        return "prefix"  # equal-length case; here we still call it prefix
    if start == 0:
        return "prefix"
    if end == total_len:
        return "suffix"
    return "middle"

# --------- Lexicon I/O ---------
def load_lexicon(path: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) < 2: continue
            w = parts[0]
            try:
                freq = float(parts[1])
            except:
                continue
            out[w] = freq
    return out

# --------- CLI ops ---------
def do_lookup(lex: Dict[str, float], words: List[str], show_pinyin: bool) -> int:
    rc = 0
    for w in words:
        if w in lex:
            msg = f"[FOUND] {w}\tfreq={lex[w]:.6f}"
            if show_pinyin:
                msg += f"\tpinyin={to_pinyin_list(w)}"
            print(msg)
        else:
            print(f"[MISS ] {w}")
            rc = 2
    return rc

def do_check_super(lex: Dict[str, float], query: str, targets: List[str], max_len_delta: Optional[int] = None) -> int:
    rc = 0
    q_py = to_pinyin_list(query)
    print(f"[QUERY] {query}\tpy={q_py}")

    for tgt in targets:
        exists = tgt in lex
        t_py = to_pinyin_list(tgt)
        start = find_subseq_start(t_py, q_py)
        pass_super = start is not None and (len(t_py) - len(q_py) >= 1)
        if max_len_delta is not None and pass_super:
            pass_super = (len(t_py) - len(q_py)) <= max_len_delta

        align = None
        ptype = None
        if start is not None:
            end = start + len(q_py)
            align = (start, end)
            ptype = pos_type(start, end, len(t_py))

        print(
            f"[TARGET] {tgt}\t"
            f"exists={exists}\t"
            f"py={t_py}\t"
            f"align={align}\t"
            f"pos_type={ptype}\t"
            f"len_diff={len(t_py)-len(q_py)}\t"
            f"super_ok={pass_super}"
        )
        if not exists:
            rc = 3
    return rc

def main():
    ap = argparse.ArgumentParser(description="Probe lexicon and super-match conditions.")
    ap.add_argument("--lex", required=True, help="lexicon TSV (word<TAB>freq)")
    ap.add_argument("--lookup", nargs="*", help="exact words to check in lexicon")
    ap.add_argument("--show-pinyin", action="store_true", help="print pinyin for lookup words")

    ap.add_argument("--check-super", action="store_true", help="check super (TARGET contains QUERY's pinyin)")
    ap.add_argument("--query", type=str, help="query word for super check")
    ap.add_argument("--target", nargs="*", help="one or more TARGET words to test against QUERY")
    ap.add_argument("--max-len-delta", type=int, default=None, help="optional: cap len_diff for super check")

    args = ap.parse_args()

    lex = load_lexicon(args.lex)

    rc = 0
    if args.lookup:
        rc = max(rc, do_lookup(lex, args.lookup, args.show_pinyin))

    if args.check_super:
        if not args.query or not args.target:
            print("ERROR: --check-super requires --query and --target ...", file=sys.stderr)
            return 4
        rc = max(rc, do_check_super(lex, args.query, args.target, max_len_delta=args.max_len_delta))

    if not args.lookup and not args.check_super:
        print("Nothing to do. Use --lookup or --check-super.", file=sys.stderr)
        return 5

    return rc

if __name__ == "__main__":
    sys.exit(main())
