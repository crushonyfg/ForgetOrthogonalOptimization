#!/usr/bin/env python
"""ProteinMPNN 评测 harness —— 生成 K 个候选 / 折叠 / 存**完整原始记录**到 JSONL。

关键设计:把"生成+折叠(贵)"和"算指标(便宜)"解耦。
本脚本只负责生成 K 个候选、折叠、把每个候选的原始字段(生成序列、mask 位置、
native 序列、identity、rmsd、TM、pLDDT、PDB 路径)落盘为 JSONL。
ASR@k / SDSR 等阈值型指标由 compute_metrics.py 从 JSONL 重算,**不用重折叠**。

用法:
  export ESMFOLD_MODEL=.../esmfold_v1 FOLDSEEK_BIN=.../foldseek
  python eval_harness_mpnn.py --weights vanilla_model_weights/v_48_020.pt \
     --test-csv dataset/dataset_cluster_split_test.csv \
     --mask-ratios 0.1,0.3 --k 8 --max-len 1000 --out results/raw_mpnn.jsonl [--limit N]
"""
import os, sys, json, time, math, hashlib, random, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--mask-ratios", default="0.1,0.3")
    ap.add_argument("--k", type=int, default=8, help="每个(蛋白,mask)生成的候选数 (pass@k)")
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--mask-mode", default="conservation",
                    choices=["conservation", "random_half", "last_half", "structure_only"],
                    help="掩码策略。默认 conservation(对齐 SafeProtein §2.1：掩最保守位点)。"
                         "SafeProtein Table 4 显示 conservation 的 ASR 远高于 random"
                         "(mask0.3: 42.19 vs 8.63)，且更能说明模型内化了危害知识。")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个蛋白(调试)")
    ap.add_argument("--mask-seed", type=int, default=123)
    ap.add_argument("--resume", action="store_true", help="续跑:跳过输出里已有的(蛋白,mask)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # env 必须在 import 前设好(TrainingConfig 在类体里读 os.environ)
    import torch
    from unified_training_dpo import (
        TrainingConfig, load_model_and_ref, ensure_reference_pdb_for_sample,
        parse_PDB, select_design_positions, generate_sequences_with_mpnn,
        evaluate_cluster_variant, load_cluster_csv_samples, normalize_label,
        resolve_local_accession_pdb, generated_pdb_cache_path, ESMFoldSubprocessRunner,
        REFERENCE_SOURCE,
    )
    cfg = TrainingConfig()
    cfg.GRPO_CLUSTER_MAX_LEN = args.max_len
    cfg.MAX_LEN = args.max_len
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ratios = [float(x) for x in args.mask_ratios.split(",")]

    samples = load_cluster_csv_samples(args.test_csv, split_name="test", max_len=args.max_len)
    # 2026-07-19 修 P2-25：**不再前置过滤掉"本地无 PDB"的蛋白**。
    # 此前这一行静默丢弃 21/203 条 test 蛋白,且两类剔除率不均(hazardous 7.7% / benign 14.0%),
    # 使危害轴与效用轴的基准不是同一总体。而 ensure_reference_pdb_for_sample 本就有
    # 三级兜底(本地 → AlphaFold 下载 → ESMFold 折叠),前置过滤是多余的。
    # 参考结构来源记入 ref_source 字段,供"仅实验结构子集"的敏感性分析。
    _n_before = len(samples)
    _n_local = sum(1 for s in samples if resolve_local_accession_pdb(s.accession, cfg))
    print(f"[harness] {_n_before} 个蛋白, 其中 {_n_local} 有本地 PDB, "
          f"{_n_before - _n_local} 将现场折叠补齐", flush=True)
    if args.limit: samples = samples[:args.limit]
    print(f"[harness] {len(samples)} 个蛋白, mask={ratios}, k={args.k}, out={args.out}", flush=True)

    model, _, _ = load_model_and_ref(dev, args.weights)
    # 2026-07-19：评测必须 eval()。ProteinMPNN 每层 dropout=0.1，train() 会给每个候选叠一层
    # 随机噪声：结果不可复现，且噪声抬高 RMSD/压低 identity，正好朝"脱毒成功"方向偏。
    # 采样随机性由 SAMPLING_TEMPERATURE 提供，不需要 dropout。
    model.eval()
    model_esm = ESMFoldSubprocessRunner(cfg.ESM_DIR, device=dev,
                 timeout_s=int(getattr(cfg, "GRPO_ESMFOLD_SUBPROCESS_TIMEOUT", 180)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # 续跑:读已完成的 (accession, mask_ratio)
    done = set()
    if args.resume and os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line); done.add((r["accession"], float(r["mask_ratio"])))
            except Exception:
                pass
        print(f"[harness] resume: 已有 {len(done)} 个(蛋白,mask)记录,跳过", flush=True)
    n_done, n_fail, t_start = 0, 0, time.time()
    try:
        with open(args.out, "a" if args.resume else "w") as fout:
            for mr in ratios:
                for si, s in enumerate(samples):
                    if (s.accession, float(mr)) in done:
                        continue
                    t0 = time.time()
                    try:
                        ref_path, chain_id = ensure_reference_pdb_for_sample(s, model_esm, cfg)
                        pdb = parse_PDB(ref_path, ca_only=False)
                        L = len(pdb[0][f"seq_chain_{chain_id}"])
                        # 每(蛋白,mask)确定性 mask —— 保证可复现且各方法看同一个 mask
                        seed = int(hashlib.md5(f"{s.accession}|{mr}|{args.mask_seed}".encode()).hexdigest()[:8], 16)
                        design_pos = select_design_positions(
                            L, args.mask_mode, random.Random(seed), mr, accession=s.accession)
                        variants, *_ = generate_sequences_with_mpnn(
                            model, ref_path, chain_id, design_pos, args.k, device=dev)
                        for ki, var in enumerate(variants):
                            m = evaluate_cluster_variant(s, var, ref_path, model_esm, cfg,
                                    step=0, sample_index=ki, num_design_positions=len(design_pos))
                            rec = {
                                "accession": s.accession, "label": s.label,
                                "cluster_id": s.cluster_id, "length": L,
                                "mask_ratio": mr, "mask_seed": args.mask_seed,
                                # 溯源字段:换阈值/换判据重算时必须能区分是哪次跑、什么配置产生的
                                "mask_mode": args.mask_mode, "arch": "proteinmpnn",
                                "ref_source": REFERENCE_SOURCE.get(s.accession, "unknown"),
                                # local = 实验结构;其余(alphafold/esmfold,含缓存)都是预测结构
                                "ref_is_predicted": REFERENCE_SOURCE.get(s.accession, "unknown") != "local",
                                "weights": args.weights,
                                "sampling_temperature": float(getattr(cfg, "SAMPLING_TEMPERATURE", 0.3)),
                                "design_positions": design_pos, "num_design": len(design_pos),
                                "candidate_idx": ki,
                                "native_seq": s.sequence, "gen_seq": var,
                                "seq_identity": m["seq_identity"], "rmsd": m["rmsd"],
                                "qtmscore": m["qtmscore"], "ttmscore": m["ttmscore"],
                                "alntmscore": m["alntmscore"], "plddt": m["plddt"],
                                "refusal": m.get("refusal", False), "num_x": m.get("num_x", 0),
                                "gen_pdb_path": generated_pdb_cache_path(s, var, cfg),
                                "ref_pdb_path": ref_path,
                            }
                            fout.write(json.dumps(rec) + "\n")
                        fout.flush()
                        n_done += 1
                        dt = time.time() - t0
                        print(f"[{n_done}/{len(samples)*len(ratios)}] {s.accession} mask={mr} "
                              f"len={L} k={args.k} 用时={dt:.1f}s "
                              f"(累计 {(time.time()-t_start)/60:.1f}min)", flush=True)
                    except Exception as e:
                        # 2026-07-19：失败必须落盘。此前只 print 不记录，失败样本完全不进分母，
                        # 指标只在"成功子集"上算；而 unlearning 后模型更易产出垃圾/折叠失败，
                        # 被静默丢弃会让剩余质量虚高、ASR 分母变小，两个方向都朝有利结论偏。
                        fout.write(json.dumps({
                            "accession": s.accession, "label": s.label,
                            "cluster_id": s.cluster_id, "length": len(s.sequence),
                            "mask_ratio": mr, "mask_seed": args.mask_seed,
                            "mask_mode": args.mask_mode, "arch": "proteinmpnn",
                            "weights": args.weights,
                            "status": "failed", "error": f"{type(e).__name__}: {e}",
                        }) + "\n")
                        fout.flush()
                        n_fail += 1
                        print(f"[skip] {s.accession} mask={mr}: {e}", flush=True)
    finally:
        if isinstance(model_esm, ESMFoldSubprocessRunner):
            model_esm.close()
    print(f"[harness] 完成 {n_done} 蛋白-mask (失败 {n_fail}), 总用时 {(time.time()-t_start)/60:.1f}min → {args.out}", flush=True)

if __name__ == "__main__":
    main()
