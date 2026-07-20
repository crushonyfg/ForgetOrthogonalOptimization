#!/usr/bin/env python
"""从 mmseqs MSA 计算逐位保守性(Shannon 熵),供 conservation mask 使用。

背景(2026-07-19):SafeProtein 用 PDBe API 的 conservation 注释做保守位点掩码,但实测
PDBe 对我们的数据覆盖极不均衡(hazardous 50% / benign 仅 8%),且长度与 UniProt 全长差 1~72 aa,
直接用会按 label 不均衡地丢样本 + 位置对不齐。故改为本地 MSA 自算:100% 覆盖、长度天然对齐、
两类一视同仁。与 SafeProtein 的差异需在论文中说明。

保守性定义:对每个 query 位点,统计 MSA 该列的氨基酸分布,算 Shannon 熵 H,
conservation = 1 - H/log(20)。越大越保守。gap 单独计一类但不计入分母(只统计有残基的行)。

用法:
  python compute_conservation.py --msa <msaDB> --out ../dbs/conservation_854.json \\
      --csv dataset/dataset_dataset_854_full.csv
"""
import argparse
import json
import math
import os
from collections import Counter

AA = "ACDEFGHIKLMNPQRSTVWY"
AASET = set(AA)


def parse_msa_blocks(path):
    """mmseqs result2msa(--msa-format-mode 2) 的库文件:每个 query 一段,\\0 分隔。"""
    raw = open(path, "rb").read().decode("utf-8", "replace")
    for block in raw.split("\x00"):
        block = block.strip()
        if not block:
            continue
        headers, seqs, cur = [], [], []
        for line in block.split("\n"):
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                headers.append(line[1:].strip())
            elif line:
                cur.append(line.strip())
        if cur:
            seqs.append("".join(cur))
        if headers and seqs and len(headers) == len(seqs):
            yield headers, seqs


def conservation_from_msa(seqs, min_depth=3):
    """seqs[0] 是 query(可能含 gap)。返回按 query 残基位置排列的 conservation 列表。

    只统计 query 非 gap 的列;每列只对有残基(非 gap)的行统计分布。
    深度不足 min_depth 的列标为 None(调用方决定如何处理)。
    """
    query = seqs[0]
    others = seqs[1:] if len(seqs) > 1 else []
    out = []
    logn = math.log(len(AA))
    for col, qchar in enumerate(query):
        if qchar == "-":
            continue  # 该列不对应 query 残基
        column = [s[col] for s in others if col < len(s) and s[col] in AASET]
        if len(column) < min_depth:
            out.append(None)
            continue
        cnt = Counter(column)
        total = sum(cnt.values())
        h = -sum((c / total) * math.log(c / total) for c in cnt.values())
        out.append(round(1.0 - h / logn, 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msa", required=True, help="mmseqs result2msa 产出的库文件")
    ap.add_argument("--csv", required=True, help="854 CSV,用于校验长度")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-depth", type=int, default=3)
    args = ap.parse_args()

    import csv as _csv

    lens = {r["accession"]: int(r["length"]) for r in _csv.DictReader(open(args.csv))}

    result, stats = {}, Counter()
    for headers, seqs in parse_msa_blocks(args.msa):
        acc = headers[0].split()[0].split("|")[-1] if "|" in headers[0] else headers[0].split()[0]
        if acc not in lens:
            stats["未知accession"] += 1
            continue
        cons = conservation_from_msa(seqs, args.min_depth)
        if len(cons) != lens[acc]:
            # query 在 MSA 中的非 gap 残基数应等于原序列长度;不等说明解析或建库有问题
            stats["长度不匹配"] += 1
            continue
        n_valid = sum(1 for c in cons if c is not None)
        result[acc] = {"conservation": cons, "n_homologs": len(seqs) - 1, "n_valid_pos": n_valid}
        stats["成功"] += 1
        if len(seqs) - 1 < args.min_depth:
            stats["同源不足"] += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"))
    print(f"写出 {len(result)}/{len(lens)} 条 → {args.out}")
    print("统计:", dict(stats))
    if result:
        depths = sorted(v["n_homologs"] for v in result.values())
        cov = sorted(v["n_valid_pos"] / max(1, len(v["conservation"])) for v in result.values())
        print(f"同源深度 中位={depths[len(depths)//2]} 最小={depths[0]} 最大={depths[-1]}")
        print(f"有效位点占比 中位={cov[len(cov)//2]:.3f} 最小={cov[0]:.3f}")


if __name__ == "__main__":
    main()
