#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model 1 pilot for 谢彬烂梗生成器

功能：
- 从词频词表构建：
    * word list
    * 完整拼音序列 -> 词 的映射（用于等长子串 equal_sub）
    * 拼音 bigram 倒排表（用于超串 super）
- 给定 query：
    * 计算 query 的拼音序列
    * 召回：
        - equal_sub: 候选拼音 == query 拼音的某个连续子串（>= min_sub_len）
        - super: 候选拼音中包含 query 全段拼音（side > middle 可加分）
    * 为每个候选计算一个轻量得分（位置 + 长度差 + 编辑距离 + 词频）
    * 输出 topk 候选，方便你后续接语义模型做 Model 2

用法示例：
    python xiebin_model1_pilot.py \
        --lex lexicon.tsv \
        --query 舒服 \
        --topk 20

词表格式（UTF-8 TSV）：
    词语<TAB>频次
    舒服\t8000
    舒肤佳\t7000
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Set

# =========================
# 拼音工具（带 fallback）
# =========================

USE_DUMMY_PINYIN = False
try:
    from pypinyin import pinyin, Style  # type: ignore
except Exception:
    USE_DUMMY_PINYIN = True
    pinyin = None
    Style = None


def normalize_pinyin_syllable(py: str) -> str:
    """小写 + 去掉数字声调"""
    py = py.lower()
    # 非严格模式下可能自带数字声调，统统去掉
    out = []
    for ch in py:
        if not ch.isdigit():
            out.append(ch)
    return "".join(out)


def text_to_pinyin_list(text: str) -> List[str]:
    """
    将中文字符串转换为“无声调拼音”的列表。
    - 优先使用 pypinyin
    - 如果不可用，则使用确定性的伪拼音（仍可测试流程）
    """
    if not text:
        return []

    if not USE_DUMMY_PINYIN and pinyin is not None:
        # strict=False 更宽容；TONE3 保留声调数字，后面再剥离
        syls = pinyin(text, style=Style.TONE3, strict=False)
        return [normalize_pinyin_syllable(s[0]) for s in syls]

    # fallback: 伪拼音，只保证稳定可复现
    return [f"p{ord(ch) % 97:02d}" for ch in text]


# =========================
# 声母/韵母拆分 & 加权编辑距
# =========================

SM_SIM = [
    {"z", "zh", "j"}, {"c", "ch", "q"}, {"s", "sh", "x"},
    {"l", "n"}, {"r", "l"}, {"g", "k", "h"}
]
YM_SIM = [
    {"an", "ang"}, {"en", "eng"}, {"in", "ing"},
    {"ian", "iang"}, {"uan", "uang"},
    {"ou", "uo"}, {"i", "yi"}, {"u", "wu"}, {"v", "yu"}
]


def split_sm_ym(py: str) -> Tuple[str, str]:
    """粗略将拼音拆成 声母 + 韵母"""
    sms = [
        "zh", "ch", "sh",  # 先长的
        "b", "p", "m", "f", "d", "t", "n", "l",
        "g", "k", "h", "j", "q", "x", "r", "z", "c", "s",
        "y", "w",
    ]
    for sm in sms:
        if py.startswith(sm):
            return sm, py[len(sm):]
    return "", py


def _is_similar(a: str, b: str, groups: List[Set[str]]) -> bool:
    for g in groups:
        if a in g and b in g:
            return True
    return False


def weighted_edit_distance(a: List[str], b: List[str]) -> float:
    """
    加权编辑距离：
      - 完全相同: 0
      - 声母或韵母相似: 0.5
      - 完全不同: 1
      - 插入/删除: 1
    """
    la, lb = len(a), len(b)
    dp = [[0.0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        dp[i][0] = float(i)
    for j in range(1, lb + 1):
        dp[0][j] = float(j)

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            asm, aym = split_sm_ym(a[i - 1])
            bsm, bym = split_sm_ym(b[j - 1])

            sm_cost = 0.0 if asm == bsm else (0.5 if _is_similar(asm, bsm, SM_SIM) else 1.0)
            ym_cost = 0.0 if aym == bym else (0.5 if _is_similar(aym, bym, YM_SIM) else 1.0)
            sub_cost = (sm_cost + ym_cost) / 2.0

            dp[i][j] = min(
                dp[i - 1][j] + 1.0,        # 删除
                dp[i][j - 1] + 1.0,        # 插入
                dp[i - 1][j - 1] + sub_cost
            )
    return dp[la][lb]


# =========================
# 数据结构
# =========================

@dataclass
class WordEntry:
    wid: int
    text: str
    pinyin: List[str]
    freq: float
    log_freq: float


@dataclass
class CandidateHit:
    wid: int
    text: str
    relation: str          # "equal_sub" | "super"
    pos_type: str          # "prefix" | "suffix" | "middle" | "na"
    len_diff: int
    edit_no_tone: float
    log_freq: float
    log_freq_norm: float
    align_start: Optional[int]  # query 在候选中的起始音节位置（仅 super）
    align_end: Optional[int]    # 结束位置（不含）（仅 super）
    score: float


# =========================
# 索引 & 检索器
# =========================

class Model1Indexer:
    """
    Model 1 pilot:
    - 词表 -> 拼音
    - 完整拼音映射（equal_sub）
    - bigram 倒排表（super）
    """

    def __init__(self, lexicon: List[Tuple[str, float]]):
        self.words: List[WordEntry] = []
        self.exact_map: Dict[Tuple[str, ...], List[int]] = {}
        self.bigram_index: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        self._build(lexicon)

    def _build(self, lexicon: List[Tuple[str, float]]) -> None:
        # 预计算 max_freq 用于归一化
        freqs = [f for _, f in lexicon if f > 0]
        max_freq = max(freqs) if freqs else 1.0

        for wid, (word, freq) in enumerate(lexicon):
            if not word:
                continue
            if freq <= 0:
                continue

            py = text_to_pinyin_list(word)
            if not py:
                continue

            logf = math.log(1.0 + freq)
            entry = WordEntry(
                wid=wid,
                text=word,
                pinyin=py,
                freq=freq,
                log_freq=logf,
            )
            self.words.append(entry)

            # 完整拼音映射
            seq = tuple(py)
            self.exact_map.setdefault(seq, []).append(entry.wid)

            # bigram 倒排
            if len(py) >= 2:
                for pos in range(len(py) - 1):
                    bg = (py[pos], py[pos + 1])
                    self.bigram_index.setdefault(bg, []).append((entry.wid, pos))

        # 再次算 max_log_freq 用于归一化
        self.max_log_freq = max((w.log_freq for w in self.words), default=1.0)

    # ---------- 辅助函数 ----------

    def _find_equal_sub_candidates(
        self,
        q_py: List[str],
        min_sub_len: int,
        max_sub_len: Optional[int] = None,
        exclude_text: Optional[str] = None,
    ) -> Dict[int, CandidateHit]:
        """
        equal_sub：候选拼音 == query 拼音的某个连续子串（>= min_sub_len）。
        返回 wid -> CandidateHit 的 dict（score 先不算）。
        """
        m = len(q_py)
        if m < min_sub_len:
            return {}

        if max_sub_len is None:
            max_sub_len = m

        hits: Dict[int, CandidateHit] = {}

        for sub_len in range(min_sub_len, min(m, max_sub_len) + 1):
            for start in range(m - sub_len + 1):
                sub_seq = tuple(q_py[start:start + sub_len])
                wids = self.exact_map.get(sub_seq)
                if not wids:
                    continue

                for wid in wids:
                    we = self.words[wid]
                    if exclude_text is not None and we.text == exclude_text:
                        continue
                    # equal_sub 的 pos_type 暂时设为 "na"
                    hits[wid] = CandidateHit(
                        wid=wid,
                        text=we.text,
                        relation="equal_sub",
                        pos_type="na",
                        len_diff=len(we.pinyin) - len(q_py),
                        edit_no_tone=0.0,   # 之后统一计算
                        log_freq=we.log_freq,
                        log_freq_norm=we.log_freq / (self.max_log_freq + 1e-9),
                        align_start=None,
                        align_end=None,
                        score=0.0,
                    )
        return hits

    def _find_super_candidates(
        self,
        q_py: List[str],
        exclude_text: Optional[str] = None,
    ) -> Dict[int, CandidateHit]:
        """
        super：候选拼音中包含 query 全段拼音。
        使用 bigram 倒排粗筛，然后逐词检查连续子串。
        """
        m = len(q_py)
        if m == 0:
            return {}

        # m == 1 时，没有 bigram，直接全词扫一遍（pilot 简化）。
        if m == 1:
            q0 = q_py[0]
            hits: Dict[int, CandidateHit] = {}
            for we in self.words:
                if exclude_text is not None and we.text == exclude_text:
                    continue
                for pos, syl in enumerate(we.pinyin):
                    if syl == q0:
                        pos_type = _pos_type(pos, pos + 1, len(we.pinyin))
                        hits[we.wid] = CandidateHit(
                            wid=we.wid,
                            text=we.text,
                            relation="super",
                            pos_type=pos_type,
                            len_diff=len(we.pinyin) - m,
                            edit_no_tone=0.0,  # 后面统一算
                            log_freq=we.log_freq,
                            log_freq_norm=we.log_freq / (self.max_log_freq + 1e-9),
                            align_start=pos,
                            align_end=pos + 1,
                            score=0.0,
                        )
                        break
            return hits

        # m >= 2，使用 bigram 倒排求交集再精检
        bigrams = [(q_py[i], q_py[i + 1]) for i in range(m - 1)]
        posting_sets: List[Set[int]] = []
        for bg in bigrams:
            postings = self.bigram_index.get(bg)
            if not postings:
                return {}  # 任一 bigram 无命中，则不可能有 super
            posting_sets.append({wid for wid, _ in postings})

        # 取交集得到候选 wid 集合
        candidate_ids = set.intersection(*posting_sets)

        hits: Dict[int, CandidateHit] = {}
        for wid in candidate_ids:
            we = self.words[wid]
            if exclude_text is not None and we.text == exclude_text:
                continue

            # 精检：q_py 是否为 we.pinyin 的连续子串
            pos = _find_subsequence_start(we.pinyin, q_py)
            if pos is None:
                continue

            align_start = pos
            align_end = pos + m
            pos_type = _pos_type(align_start, align_end, len(we.pinyin))

            hits[wid] = CandidateHit(
                wid=we.wid,
                text=we.text,
                relation="super",
                pos_type=pos_type,
                len_diff=len(we.pinyin) - m,
                edit_no_tone=0.0,  # 后面统一计算
                log_freq=we.log_freq,
                log_freq_norm=we.log_freq / (self.max_log_freq + 1e-9),
                align_start=align_start,
                align_end=align_end,
                score=0.0,
            )

        return hits

    def search(
        self,
        query_text: str,
        topk: int = 20,
        min_sub_len: int = 2,
        max_sub_len: Optional[int] = None,
        include_equal_sub: bool = True,
        include_super: bool = True,
        exclude_self: bool = True,
    ) -> List[CandidateHit]:
        """
        统一接口：
        - query_text: 原词
        - 返回经过轻量打分排序的候选
        """
        q_py = text_to_pinyin_list(query_text)

        exclude_text = query_text if exclude_self else None

        all_hits: Dict[int, CandidateHit] = {}

        if include_equal_sub:
            eq_hits = self._find_equal_sub_candidates(
                q_py=q_py,
                min_sub_len=min_sub_len,
                max_sub_len=max_sub_len,
                exclude_text=exclude_text,
            )
            all_hits.update(eq_hits)

        if include_super:
            sp_hits = self._find_super_candidates(
                q_py=q_py,
                exclude_text=exclude_text,
            )
            # 如果同一个 wid 同时出现在 equal_sub 和 super 中，保留“关系更强”的 super
            for wid, hit in sp_hits.items():
                prev = all_hits.get(wid)
                if prev is None or prev.relation != "super":
                    all_hits[wid] = hit

        # 为所有候选计算编辑距离 & 总分
        for wid, hit in all_hits.items():
            we = self._get_word_by_id(wid)
            edit = weighted_edit_distance(q_py, we.pinyin)
            hit.edit_no_tone = edit
            hit.score = self._light_score(hit)

        # 排序 & 截断
        hits_sorted = sorted(all_hits.values(), key=lambda h: h.score, reverse=True)
        return hits_sorted[:topk]

    def _get_word_by_id(self, wid: int) -> WordEntry:
        # 这里假设 wid 与 self.words 的顺序一致
        # 如果未来做过滤/重排，可以维护 id->entry 的 dict
        return self.words[wid]

    def _light_score(self, hit: CandidateHit) -> float:
        """
        轻量打分：
        score = 0.3 * position_bonus
              - 0.25 * |len_diff|
              - 0.4  * edit_no_tone
              + 0.2  * log_freq_norm
        """
        pos_b = position_bonus(hit.pos_type)
        len_pen = abs(hit.len_diff)

        score = 0.0
        score += 0.3 * pos_b
        score -= 0.25 * float(len_pen)
        score -= 0.4 * float(hit.edit_no_tone)
        score += 0.2 * float(hit.log_freq_norm)
        return score


# =========================
# 小工具函数
# =========================

def _find_subsequence_start(seq: List[str], sub: List[str]) -> Optional[int]:
    """在 seq 中查找 sub 的连续子序列起点，找不到则返回 None。"""
    n, m = len(seq), len(sub)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if seq[i:i + m] == sub:
            return i
    return None


def _pos_type(start: int, end: int, total_len: int) -> str:
    """
    决定“prefix/suffix/middle”（用于 side > middle 的奖励）
    start, end: 命中片段在候选中的 [start, end)
    """
    if start == 0 and end == total_len:
        # 完全一致，算 prefix（也可以单独标 equal，但这里简单点）
        return "prefix"
    if start == 0:
        return "prefix"
    if end == total_len:
        return "suffix"
    return "middle"


def position_bonus(pos_type: str) -> float:
    if pos_type == "prefix":
        return 1.0
    if pos_type == "suffix":
        return 0.8
    if pos_type == "middle":
        return 0.3
    return 0.0


def load_lexicon_tsv(path: str) -> List[Tuple[str, float]]:
    """
    读取 TSV 词表：
        词语<TAB>频次
    并对相同词语的频次进行累加去重。
    返回 [(word, freq), ...]，按 freq 降序。
    """
    agg: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word = parts[0]
            try:
                freq = float(parts[1])
            except ValueError:
                continue
            agg[word] = agg.get(word, 0.0) + freq

    items = list(agg.items())
    # 方便后面使用高频词，先按频次降序排序
    items.sort(key=lambda x: x[1], reverse=True)
    return items


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Model 1 pilot for 谢彬烂梗生成器（拼音检索 + 轻量打分）"
    )
    parser.add_argument("--lex", required=True, help="词表路径（TSV: word<TAB>freq）")
    parser.add_argument("--query", required=True, help="原词（中文）")
    parser.add_argument("--topk", type=int, default=20, help="返回候选数量")
    parser.add_argument("--min_sub_len", type=int, default=2, help="equal_sub 最小音节长度")
    parser.add_argument("--max_sub_len", type=int, default=None, help="equal_sub 最大音节长度（默认=query长度）")
    parser.add_argument("--no_equal_sub", action="store_true", help="不召回 equal_sub 候选")
    parser.add_argument("--no_super", action="store_true", help="不召回 super 候选")
    parser.add_argument("--include_self", action="store_true", help="是否保留与原词文本完全相同的候选")
    args = parser.parse_args()

    lexicon = load_lexicon_tsv(args.lex)
    indexer = Model1Indexer(lexicon)

    hits = indexer.search(
        query_text=args.query,
        topk=args.topk,
        min_sub_len=args.min_sub_len,
        max_sub_len=args.max_sub_len,
        include_equal_sub=not args.no_equal_sub,
        include_super=not args.no_super,
        exclude_self=not args.include_self,
    )

    result = {
        "query": args.query,
        "query_pinyin": text_to_pinyin_list(args.query),
        "items": [
            {
                "text": h.text,
                "relation": h.relation,
                "pos_type": h.pos_type,
                "len_diff": h.len_diff,
                "edit_no_tone": round(h.edit_no_tone, 4),
                "log_freq": round(h.log_freq, 4),
                "log_freq_norm": round(h.log_freq_norm, 4),
                "align": None if h.align_start is None else [h.align_start, h.align_end],
                "score": round(h.score, 4),
            }
            for h in hits
        ],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
