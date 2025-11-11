#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从原始中文语料构建 lexicon.tsv 的小工具。

输入：
    一个目录，里面是若干 .txt 文件（UTF-8），内容是中文文本（未分词也没关系）。

处理：
    - 用 jieba 分词
    - 只保留“全部是中文”的 token
    - 只保留长度在 [min_len, max_len] 的词（一般 2~6 比较适合做梗）
    - 统计频次，按频次降序输出为 TSV：word<TAB>freq

示例：
    python build_lexicon_from_corpus.py \
        --input_dir ./corpus_txt \
        --out_lex lexicon.tsv \
        --min_len 2 \
        --max_len 6 \
        --min_freq 5
"""

import os
import re
import argparse
from collections import Counter
from typing import Iterator

import jieba
from tqdm import tqdm

CHINESE_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def iter_text_files(input_dir: str) -> Iterator[str]:
    for root, _, files in os.walk(input_dir):
        for name in files:
            if not name.lower().endswith(".txt"):
                continue
            path = os.path.join(root, name)
            yield path


def build_lexicon(
    input_dir: str,
    min_len: int = 2,
    max_len: int = 6,
    min_freq: int = 5,
) -> Counter:
    counter = Counter()
    files = list(iter_text_files(input_dir))
    if not files:
        print(f"[WARN] No .txt files found under {input_dir}")
        return counter

    for path in tqdm(files, desc="Scanning corpus"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            # 简单跳过非 UTF-8 文件
            continue

        # jieba 分词
        for token in jieba.cut(text):
            token = token.strip()
            if not token:
                continue
            # 只保留“全部是中文”的 token
            if not CHINESE_RE.match(token):
                continue
            # 长度限制
            if len(token) < min_len or len(token) > max_len:
                continue
            counter[token] += 1

    # 过滤低频词
    if min_freq > 1:
        for w in list(counter.keys()):
            if counter[w] < min_freq:
                del counter[w]

    return counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="包含若干 .txt 中文语料文件的目录")
    ap.add_argument("--out_lex", required=True, help="输出 lexicon.tsv 路径")
    ap.add_argument("--min_len", type=int, default=2, help="最小词长")
    ap.add_argument("--max_len", type=int, default=6, help="最大词长")
    ap.add_argument("--min_freq", type=int, default=5, help="最小频次阈值")
    args = ap.parse_args()

    counter = build_lexicon(
        input_dir=args.input_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        min_freq=args.min_freq,
    )

    print(f"[INFO] Collected {len(counter)} words after filtering.")

    # 按频次降序写出 TSV
    with open(args.out_lex, "w", encoding="utf-8") as out:
        for word, freq in counter.most_common():
            out.write(f"{word}\t{freq}\n")

    print(f"[INFO] Wrote lexicon to {args.out_lex}")


if __name__ == "__main__":
    main()
