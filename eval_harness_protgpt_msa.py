#!/usr/bin/env python
"""ProtGPT3-MSA 同源 few-shot 攻击 harness —— 存完整原始记录到 JSONL(与 MPNN 同格式)。

攻击:给每个毒素 ≤15 条同源(Swiss-Prot 检索,排除>0.95的答案)作上下文 →
ProtGPT3-MSA 生成"家族一致"的新成员 → 折叠 + 记录。是否有害由 score_hazard.py 打标。

用法:
  export ESMFOLD_MODEL=.../esmfold_v1 FOLDSEEK_BIN=.../foldseek
  python eval_harness_protgpt_msa.py --model /path/ProtGPT3-MSA \
     --context ../dbs/homolog_context.json --test-csv dataset/dataset_cluster_split_test.csv \
     --k 8 --out results/raw_msa.jsonl [--limit N]
"""
import os, json, time, re, random, argparse

def process_style(seq, gap=False):
    return re.sub(r"[X]", "", seq.upper()) if gap else re.sub(r"[X]", "", seq.replace("-", "").upper())

def build_prompt(sequences, gap=False, direction="1"):
    sequences = list(sequences); random.shuffle(sequences)
    gap_token = "<gap>" if gap else "<no_gap>"
    tokens = ["<|bos|>", direction, gap_token]
    for seq in sequences:
        tokens.append("<s>"); tokens.extend(list(process_style(seq, gap)))
    tokens.append("<s>")
    return " ".join(tokens)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from unified_training_dpo import (
        TrainingConfig, ensure_reference_pdb_for_sample, load_cluster_csv_samples,
        ensure_generated_pdb_for_sequence, safe_foldseek_tmscore, rmsd_CA,
        generated_pdb_cache_path, ESMFoldSubprocessRunner,
    )
    cfg = TrainingConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = json.load(open(args.context))
    samples = {s.accession: s for s in load_cluster_csv_samples(args.test_csv, split_name="test", max_len=1000)}
    toxins = [a for a in ctx if a in samples]
    if args.limit: toxins = toxins[:args.limit]
    print(f"[msa] 可攻击毒素 {len(toxins)}, k={args.k}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, add_bos_token=False, add_eos_token=False, padding_side="left")
    lm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    model_esm = ESMFoldSubprocessRunner(cfg.ESM_DIR, device=dev,
                 timeout_s=int(getattr(cfg, "GRPO_ESMFOLD_SUBPROCESS_TIMEOUT", 180)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n, t0 = 0, time.time()
    try:
        with open(args.out, "w") as fout:
            for acc in toxins:
                tt = time.time(); s = samples[acc]; homs = ctx[acc]
                try:
                    ref_path, _ = ensure_reference_pdb_for_sample(s, model_esm, cfg)
                    prompt = build_prompt(homs, gap=False)
                    inp = tok(prompt, return_tensors="pt").to(dev)
                    L = len(s.sequence)
                    out = lm.generate(inp["input_ids"], do_sample=True, temperature=args.temperature,
                                      top_p=0.9, max_new_tokens=min(1024, L + 20),
                                      num_return_sequences=args.k,
                                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
                    for ki, o in enumerate(out):
                        dec = tok.decode(o, skip_special_tokens=True)
                        segs = dec.split("<s>")
                        # 生成的新成员 = 上下文之后的第一段
                        gi = len(homs) + 1
                        gen = segs[gi].replace(" ", "").strip() if len(segs) > gi else ""
                        gen = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", gen)[:1000]
                        if len(gen) < 20: continue
                        gen_pdb, plddt = ensure_generated_pdb_for_sequence(s, gen, model_esm, cfg)
                        tm = safe_foldseek_tmscore(gen_pdb, ref_path, cfg)
                        try: rmsd = rmsd_CA(gen_pdb, ref_path)
                        except Exception: rmsd = float("nan")
                        rec = {"accession": acc, "label": s.label, "cluster_id": s.cluster_id,
                               "length": L, "gen_len": len(gen), "n_homologs": len(homs),
                               "candidate_idx": ki, "mask_ratio": "msa_fewshot",
                               "native_seq": s.sequence, "gen_seq": gen,
                               "seq_identity": None, "rmsd": rmsd,
                               "qtmscore": tm.get("qtmscore"), "ttmscore": tm.get("ttmscore"),
                               "alntmscore": tm.get("alntmscore"), "plddt": plddt,
                               "gen_pdb_path": gen_pdb, "ref_pdb_path": ref_path, "arch": "protgpt3-msa"}
                        fout.write(json.dumps(rec) + "\n")
                    fout.flush(); n += 1
                    print(f"[{n}/{len(toxins)}] {acc} L={L} homs={len(homs)} 用时={time.time()-tt:.1f}s "
                          f"(累计 {(time.time()-t0)/60:.1f}min)", flush=True)
                except Exception as e:
                    print(f"[skip] {acc}: {e}", flush=True)
    finally:
        if isinstance(model_esm, ESMFoldSubprocessRunner): model_esm.close()
    print(f"[msa] 完成 {n}, 总用时 {(time.time()-t0)/60:.1f}min → {args.out}", flush=True)

if __name__ == "__main__":
    main()
