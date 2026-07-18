#!/usr/bin/env python
"""ProtGPT3 评测 harness —— 前缀续写攻击,存**完整原始记录**到 JSONL(与 MPNN 同格式)。

攻击(序列级,无结构):给毒/良蛋白前 (1-mask_ratio) 段作前缀 → ProtGPT3 续写剩余
mask_ratio 段 → 生成 K 个候选。identity 只在**续写区**上算(前缀是给的,不计)。
折叠/foldseek/pLDDT/JSONL 全复用 unified_training_dpo 的函数 → 同一个 compute_metrics.py。

用法:
  export ESMFOLD_MODEL=.../esmfold_v1 FOLDSEEK_BIN=.../foldseek
  python eval_harness_protgpt.py --model /path/ProtGPT3-112M \
     --test-csv dataset/dataset_cluster_split_test.csv \
     --mask-ratios 0.1,0.3 --k 8 --max-len 300 --out results/raw_protgpt.jsonl [--limit N]
"""
import os, json, time, hashlib, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--mask-ratios", default="0.1,0.3")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from unified_training_dpo import (
        TrainingConfig, ensure_reference_pdb_for_sample, load_cluster_csv_samples,
        resolve_local_accession_pdb, ensure_generated_pdb_for_sequence,
        safe_foldseek_tmscore, rmsd_CA, seq_identity, generated_pdb_cache_path,
        ESMFoldSubprocessRunner,
    )
    cfg = TrainingConfig(); cfg.GRPO_CLUSTER_MAX_LEN = args.max_len; cfg.MAX_LEN = args.max_len
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ratios = [float(x) for x in args.mask_ratios.split(",")]

    samples = load_cluster_csv_samples(args.test_csv, split_name="test", max_len=args.max_len)
    samples = [s for s in samples if resolve_local_accession_pdb(s.accession, cfg)]
    if args.limit: samples = samples[:args.limit]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    lm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    bos = tok.bos_token_id
    print(f"[protgpt] {len(samples)} 蛋白, mask={ratios}, k={args.k}", flush=True)

    model_esm = ESMFoldSubprocessRunner(cfg.ESM_DIR, device=dev,
                 timeout_s=int(getattr(cfg, "GRPO_ESMFOLD_SUBPROCESS_TIMEOUT", 180)))

    def gen_candidates(prefix, total_len, k):
        ids = [bos] + tok.encode(prefix, add_special_tokens=False)
        inp = torch.tensor([ids], device=dev)
        out = lm.generate(inp, do_sample=True, temperature=args.temperature,
                          num_return_sequences=k, max_new_tokens=max(1, total_len - len(prefix)),
                          pad_token_id=tok.pad_token_id or tok.eos_token_id)
        seqs = []
        for o in out:
            txt = tok.decode(o[1:], skip_special_tokens=True).replace(" ", "")
            full = (prefix + txt[len(prefix):]) if txt.startswith(prefix) else (prefix + txt)
            seqs.append(full[:total_len])
        return seqs

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n, t0 = 0, time.time()
    try:
        with open(args.out, "w") as fout:
            for mr in ratios:
                for s in samples:
                    tt = time.time()
                    try:
                        ref_path, _ = ensure_reference_pdb_for_sample(s, model_esm, cfg)
                        L = len(s.sequence)
                        plen = max(1, int(L * (1 - mr)))          # 前缀长度 = (1-mask) 段
                        prefix = s.sequence[:plen]
                        cands = gen_candidates(prefix, L, args.k)
                        for ki, gen in enumerate(cands):
                            if len(gen) < 5: continue
                            gen_pdb, plddt = ensure_generated_pdb_for_sequence(s, gen, model_esm, cfg)
                            tm = safe_foldseek_tmscore(gen_pdb, ref_path, cfg)
                            try: rmsd = rmsd_CA(gen_pdb, ref_path)
                            except Exception: rmsd = float("nan")
                            # identity 只在续写区(plen:) 上算
                            gid = seq_identity(gen[plen:], s.sequence[plen:]) if len(gen) > plen else float("nan")
                            rec = {
                                "accession": s.accession, "label": s.label, "cluster_id": s.cluster_id,
                                "length": L, "mask_ratio": mr, "prefix_len": plen,
                                "num_design": L - plen, "candidate_idx": ki,
                                "native_seq": s.sequence, "gen_seq": gen,
                                "seq_identity": gid, "rmsd": rmsd,
                                "qtmscore": tm.get("qtmscore"), "ttmscore": tm.get("ttmscore"),
                                "alntmscore": tm.get("alntmscore"), "plddt": plddt,
                                "gen_pdb_path": gen_pdb, "ref_pdb_path": ref_path,
                                "arch": "protgpt3",
                            }
                            fout.write(json.dumps(rec) + "\n")
                        fout.flush(); n += 1
                        print(f"[{n}/{len(samples)*len(ratios)}] {s.accession} mask={mr} L={L} "
                              f"用时={time.time()-tt:.1f}s (累计 {(time.time()-t0)/60:.1f}min)", flush=True)
                    except Exception as e:
                        print(f"[skip] {s.accession} mask={mr}: {e}", flush=True)
    finally:
        if isinstance(model_esm, ESMFoldSubprocessRunner): model_esm.close()
    print(f"[protgpt] 完成 {n}, 总用时 {(time.time()-t0)/60:.1f}min → {args.out}", flush=True)

if __name__ == "__main__":
    main()
