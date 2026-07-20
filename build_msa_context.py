#!/usr/bin/env python
"""为 ProtGPT3-MSA 重建同源 few-shot 上下文。

2026-07-19 重建原因(旧文件三个问题):
  1. **泄漏**:旧 context 的同源里混着 854 序列本身 —— 实测 1347 条同源中 247 条是 854 的
     hazardous、其中 209 条属 test split;更直接的证据是 33/527(6.3%)生成序列与 prompt 里
     某条同源**逐字符完全相同**。攻击可以靠"复读 prompt"达成,而复读能力不受 unlearning 影响,
     该 baseline 的结论不成立。
  2. **难例污染**:train context 1244 条里 710 条不在 854 内(难例衍生),而难例概念已废除。
  3. **缺 benign**:test context 96 条**全是 hazardous**,效用轴无数据。

对齐 ProtGPT3-MSA 官方规范(models/ProtGPT3-MSA/README.md + msa_model_pretraining/
mini_clust_extract.py):
  - 推理时 **至多 15 条**同源(训练是 16 条一组,留一位给生成)
  - 官方从同源池中 **random.sample**,不是取 e-value 前 N —— 本脚本对齐
  - 官方 no_gap 模式的 process_style 只去 gap 与 X,**保留 B/Z/U/O 等非标准残基**
  - 官方训练数据要求 ≥16 条同源,但那是**筛训练数据**用的;推理无下界。
    本脚本**不设门槛**(有多少给多少),因为硬门槛会不成比例地砍掉 hazardous
    (Swiss-Prot 下 ≥15 条的比例:hazardous 74% vs benign 90%),而那正是危害轴。
    改为记录 ``n_homologs`` 供评测时分层。

用法:
  python build_msa_context.py --msa <msaDB> --csv dataset/dataset_dataset_854_full.csv \\
      --out ../dbs/homolog_context_854.json
"""
import argparse
import csv
import json
import os
import random
import re
from collections import Counter

MAX_HOMOLOGS = 15  # 官方上界


def process_style_no_gap(seq):
    """官方 no_gap 模式:去 gap、去 X、转大写。**保留 B/Z/U/O**(与官方一致)。"""
    return re.sub(r"[X]", "", seq.replace("-", "").upper())


def parse_msa_blocks(path):
    raw = open(path, "rb").read().decode("utf-8", "replace")
    for block in raw.split("\x00"):
        block = block.strip()
        if not block:
            continue
        headers, seqs, cur = [], [], []
        for line in block.split("\n"):
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
                headers.append(line[1:].strip())
            elif line:
                cur.append(line.strip())
        if cur:
            seqs.append("".join(cur))
        if headers and seqs and len(headers) == len(seqs):
            yield headers, seqs


def acc_of(header):
    h = header.split()[0]
    return h.split("|")[1] if "|" in h else h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msa", required=True)
    ap.add_argument("--csv", required=True, help="854 全集,其 accession 与序列都要从同源池排除")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-homologs", type=int, default=MAX_HOMOLOGS)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    lab = {r["accession"]: r["label"] for r in rows}
    # 双重排除:accession 与序列本身(同一条蛋白可能以不同 accession 出现在 Swiss-Prot)
    ban_acc = set(lab)
    ban_seq = {r["sequence"].strip().upper() for r in rows}

    rng = random.Random(args.seed)
    out, stats = {}, Counter()
    n_removed = 0
    for headers, seqs in parse_msa_blocks(args.msa):
        q = acc_of(headers[0])
        if q not in lab:
            stats["非854查询"] += 1
            continue
        pool = []
        for h, s in zip(headers[1:], seqs[1:]):
            a = acc_of(h)
            clean = process_style_no_gap(s)
            if a in ban_acc or clean in ban_seq:
                n_removed += 1
                continue          # ← 泄漏修复:排除全部 854
            if len(clean) < 20:
                continue
            pool.append(clean)
        # 去重(同一序列可能多次出现)
        seen, uniq = set(), []
        for s in pool:
            if s not in seen:
                seen.add(s); uniq.append(s)
        k = min(args.max_homologs, len(uniq))
        picked = rng.sample(uniq, k) if k else []
        out[q] = {"label": lab[q], "n_homologs": len(picked), "n_pool": len(uniq),
                  "homologs": picked}
        stats["总计"] += 1
        stats[f"{lab[q]}_有同源" if picked else f"{lab[q]}_无同源"] += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"写出 {len(out)} 条 → {args.out}")
    print(f"排除的 854 泄漏同源: {n_removed} 条")
    print("统计:", dict(stats))
    for L in ("hazardous", "benign"):
        ns = sorted(v["n_homologs"] for v in out.values() if v["label"] == L)
        if ns:
            print(f"  {L}: n={len(ns)} 同源数 中位={ns[len(ns)//2]} "
                  f"满15条的={sum(1 for x in ns if x >= 15)} 零同源的={sum(1 for x in ns if x == 0)}")


if __name__ == "__main__":
    main()
