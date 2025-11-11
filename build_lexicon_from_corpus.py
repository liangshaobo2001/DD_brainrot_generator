#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a lexicon.tsv from the THUOCL corpus.

Expected THUOCL layout (relative to project root):

    corpus/
        THUOCL/
            data/
                THUOCL_chengyu.txt
                THUOCL_it.txt
                ...

Each data file has lines like:

    坚定不移\t54113

We treat the first field as the phrase and the last field as the integer frequency.
All THUOCL files are merged into a single lexicon, summing frequencies for
duplicate phrases across domains.
"""

import argparse
import collections
import pathlib
from typing import Dict, Iterable, Tuple


ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_THUOCL_DIR = ROOT / "corpus" / "THUOCL" / "data"
DEFAULT_OUTPUT = ROOT / "lexicon.tsv"


def parse_thuocl_file(path: pathlib.Path) -> Iterable[Tuple[str, int]]:
    """
    Yield (phrase, freq) pairs from a THUOCL .txt file.

    Lines look like:
        坚定不移\t54113
    or occasionally have extra whitespace.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # split on whitespace (tab or spaces)
            parts = line.split()
            if len(parts) < 2:
                continue

            # last token is frequency, everything before is the phrase
            freq_str = parts[-1]
            phrase = "".join(parts[:-1])

            try:
                freq = int(freq_str)
            except ValueError:
                # if we can't parse frequency, skip this line
                continue

            if not phrase:
                continue

            yield phrase, freq


def load_thuocl(thuocl_dir: pathlib.Path) -> Dict[str, int]:
    """
    Load all THUOCL .txt files under `thuocl_dir` and return
    a dict: phrase -> total frequency.
    """
    counter: Dict[str, int] = collections.Counter()

    if not thuocl_dir.exists():
        raise FileNotFoundError(f"THUOCL directory not found: {thuocl_dir}")

    for path in sorted(thuocl_dir.glob("THUOCL_*.txt")):
        print(f"[INFO] Loading {path.name}...")
        for phrase, freq in parse_thuocl_file(path):
            counter[phrase] += freq

    print(f"[INFO] Loaded {len(counter)} unique phrases from THUOCL.")
    return counter


def filter_and_sort(
    freq_dict: Dict[str, int],
    min_freq: int = 1,
    top_k: int | None = None,
) -> Iterable[Tuple[str, int]]:
    """
    Apply a min frequency filter and return phrases sorted by frequency desc.
    Optionally keep only top_k entries.
    """
    items = [(p, f) for p, f in freq_dict.items() if f >= min_freq]
    items.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        items = items[:top_k]

    return items


def write_lexicon(
    items: Iterable[Tuple[str, int]],
    output_path: pathlib.Path,
) -> None:
    """
    Write lexicon.tsv with columns:
        phrase \t freq

    If your downstream code expects different columns (e.g. phrase \t weight),
    you can adjust this function accordingly.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for phrase, freq in items:
            out.write(f"{phrase}\t{freq}\n")
    print(f"[INFO] Wrote lexicon to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build lexicon.tsv from THUOCL corpus."
    )
    parser.add_argument(
        "--thuocl-dir",
        type=pathlib.Path,
        default=DEFAULT_THUOCL_DIR,
        help=f"Path to THUOCL data directory (default: {DEFAULT_THUOCL_DIR})",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help=f"Output lexicon file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum frequency threshold for phrases (default: 1)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep only top-K phrases by frequency (default: keep all)",
    )

    args = parser.parse_args()

    freq_dict = load_thuocl(args.thuocl_dir)
    items = filter_and_sort(freq_dict, min_freq=args.min_freq, top_k=args.top_k)
    write_lexicon(items, args.output)


if __name__ == "__main__":
    main()
