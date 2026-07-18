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
     --mask-ratios 0.1,0.3 --k 8 --max-len 300 --out results/raw_mpnn.jsonl [--limit N]
"""
import os, sys, json, time, math, hashlib, random, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--mask-ratios", default="0.1,0.3")
    ap.add_argument("--k", type=int, default=8, help="每个(蛋白,mask)生成的候选数 (pass@k)")
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个蛋白(调试)")
    ap.add_argument("--mask-seed", type=int, default=123)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # env 必须在 import 前设好(TrainingConfig 在类体里读 os.environ)
    import torch
    from unified_training_dpo import (
        TrainingConfig, load_model_and_ref, ensure_reference_pdb_for_sample,
        parse_PDB, select_design_positions, generate_sequences_with_mpnn,
        evaluate_cluster_variant, load_cluster_csv_samples, normalize_label,
        resolve_local_accession_pdb, generated_pdb_cache_path, ESMFoldSubprocessRunner,
    )
    cfg = TrainingConfig()
    cfg.GRPO_CLUSTER_MAX_LEN = args.max_len
    cfg.MAX_LEN = args.max_len
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ratios = [float(x) for x in args.mask_ratios.split(",")]

    samples = load_cluster_csv_samples(args.test_csv, split_name="test", max_len=args.max_len)
    samples = [s for s in samples if resolve_local_accession_pdb(s.accession, cfg)]
    if args.limit: samples = samples[:args.limit]
    print(f"[harness] {len(samples)} 个蛋白, mask={ratios}, k={args.k}, out={args.out}", flush=True)

    model, _, _ = load_model_and_ref(dev, args.weights)
    model.train()
    model_esm = ESMFoldSubprocessRunner(cfg.ESM_DIR, device=dev,
                 timeout_s=int(getattr(cfg, "GRPO_ESMFOLD_SUBPROCESS_TIMEOUT", 180)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_done, t_start = 0, time.time()
    try:
        with open(args.out, "w") as fout:
            for mr in ratios:
                for si, s in enumerate(samples):
                    t0 = time.time()
                    try:
                        ref_path, chain_id = ensure_reference_pdb_for_sample(s, model_esm, cfg)
                        pdb = parse_PDB(ref_path, ca_only=False)
                        L = len(pdb[0][f"seq_chain_{chain_id}"])
                        # 每(蛋白,mask)确定性 mask —— 保证可复现且各方法看同一个 mask
                        seed = int(hashlib.md5(f"{s.accession}|{mr}|{args.mask_seed}".encode()).hexdigest()[:8], 16)
                        design_pos = select_design_positions(L, "random_half", random.Random(seed), mr)
                        variants, *_ = generate_sequences_with_mpnn(
                            model, ref_path, chain_id, design_pos, args.k, device=dev)
                        for ki, var in enumerate(variants):
                            m = evaluate_cluster_variant(s, var, ref_path, model_esm, cfg,
                                    step=0, sample_index=ki, num_design_positions=len(design_pos))
                            rec = {
                                "accession": s.accession, "label": s.label,
                                "cluster_id": s.cluster_id, "length": L,
                                "mask_ratio": mr, "mask_seed": args.mask_seed,
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
                        print(f"[skip] {s.accession} mask={mr}: {e}", flush=True)
    finally:
        if isinstance(model_esm, ESMFoldSubprocessRunner):
            model_esm.close()
    print(f"[harness] 完成 {n_done} 蛋白-mask, 总用时 {(time.time()-t_start)/60:.1f}min → {args.out}", flush=True)

if __name__ == "__main__":
    main()
