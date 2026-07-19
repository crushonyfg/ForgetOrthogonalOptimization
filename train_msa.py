#!/usr/bin/env python
"""ProtGPT3-MSA baseline 训练:NPO+RT / DPO-unlearning / GRPO(reconstruction reward A)。
和 MPNN 侧共用:同一 base、同一 forget 曝光预算、KL 锚 base、难例进 retain、log 间隔 checkpoint。

- 例子 = (同源上下文 prompt, 目标蛋白序列);teacher-forcing 算序列 log-prob(只在目标段)。
- NPO+RT: hazard 用 NPO 有界遗忘;retain(benign+训练难例)用 NLL+KL。
- DPO:  prompt=毒素家族上下文, chosen=该毒素的良性同源(训练难例), rejected=毒素序列。
- GRPO: 给上下文生成 K 个成员 → ESMFold+foldseek → reward A(发散×foldability)→ GRPO。

用法:
  export ESMFOLD_MODEL=... FOLDSEEK_BIN=...
  python train_msa.py --method npo --model /path/ProtGPT3-MSA \
     --train-csv dataset/dataset_cluster_split_train.csv \
     --hardneg-csv dataset/hard_negatives_train.csv \
     --context ../dbs/homolog_context_train.json --epochs 3 --out outputs/msa_npo
"""
import os, re, json, time, math, random, argparse, copy

AA = set("ACDEFGHIKLMNPQRSTVWY")
def clean_seq(s): return "".join(c for c in s.upper() if c in AA)

def build_prompt(seqs, direction="1"):
    seqs = list(seqs); random.shuffle(seqs)
    toks = ["<|bos|>", direction, "<no_gap>"]
    for s in seqs:
        toks.append("<s>"); toks.extend(list(clean_seq(s)))
    toks.append("<s>")
    return " ".join(toks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["npo", "dpo", "grpo"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--hardneg-csv", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--npo-beta", type=float, default=0.1)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--kl-beta", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=4, help="GRPO 每 prompt rollout 数")
    ap.add_argument("--hazard-frac", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch, torch.nn.functional as F
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    ctx = json.load(open(args.context))

    tr = pd.read_csv(args.train_csv); tr["accession"] = tr["accession"].astype(str)
    hn = pd.read_csv(args.hardneg_csv); hn["accession"] = hn["accession"].astype(str)
    hn["source_toxin"] = hn["source_toxin"].astype(str)
    hazard = [(r.accession, r.sequence) for r in tr[tr.label == "hazardous"].itertuples() if r.accession in ctx]
    retain = [(r.accession, r.sequence) for r in tr[tr.label == "benign"].itertuples() if r.accession in ctx]
    retain += [(r.accession, r.sequence) for r in hn.itertuples() if r.accession in ctx]
    tox2hn = {}  # 毒素 → 一个良性同源序列(DPO chosen)
    for r in hn.itertuples():
        tox2hn.setdefault(r.source_toxin, r.sequence)
    print(f"[msa-{args.method}] hazard={len(hazard)} retain={len(retain)} (含难例)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, add_bos_token=False, add_eos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev)
    model.train()
    ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev).eval()
    for p in ref.parameters(): p.requires_grad = False
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def ids_labels(acc, target_seq):
        """prompt=上下文 + 目标序列;labels 只在目标段"""
        prompt = build_prompt(ctx[acc])
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        t_ids = tok(" ".join(list(clean_seq(target_seq))), add_special_tokens=False).input_ids
        ids = (p_ids + t_ids)[:4096]
        lab = ([-100]*len(p_ids) + t_ids)[:4096]
        return torch.tensor([ids], device=dev), torch.tensor([lab], device=dev)

    def seq_logp(m, ids, lab):
        out = m(input_ids=ids)
        lp = F.log_softmax(out.logits[0, :-1], dim=-1)
        tgt = ids[0, 1:]
        tl = lp[range(tgt.shape[0]), tgt]
        mask = (lab[0, 1:] != -100).float()
        return (tl * mask).sum(), mask.sum()

    # ---- 训练循环 ----
    from unified_training_dpo import compute_npo_loss, compute_dpo_loss
    total = args.epochs * (len(hazard) + len(retain))
    milestones = sorted(set([1] + [2**i for i in range(20) if 2**i <= total] + [total]))
    g, rng = 0, random.Random(456)

    def save(tag):
        model.save_pretrained(os.path.join(args.out, tag)); tok.save_pretrained(os.path.join(args.out, tag))

    if args.method in ("npo", "dpo"):
        for ep in range(args.epochs):
            n_steps = len(hazard) + len(retain); tl = 0.0; nu = 0
            for _ in range(n_steps):
                g += 1; is_h = rng.random() < args.hazard_frac
                try:
                    if args.method == "npo":
                        if is_h:
                            acc, seq = rng.choice(hazard)
                            ids, lab = ids_labels(acc, seq)
                            slp, _ = seq_logp(model, ids, lab)
                            with torch.no_grad(): slp_r, _ = seq_logp(ref, ids, lab)
                            loss = compute_npo_loss(slp.unsqueeze(0), slp_r.unsqueeze(0), args.npo_beta)
                        else:
                            acc, seq = rng.choice(retain)
                            ids, lab = ids_labels(acc, seq)
                            slp, m = seq_logp(model, ids, lab)
                            with torch.no_grad(): slp_r, _ = seq_logp(ref, ids, lab)
                            nll = -slp / m.clamp(min=1)
                            loss = nll + args.kl_beta * ((slp_r - slp) / m.clamp(min=1)).abs()
                    else:  # dpo:prompt=毒素上下文, chosen=良性同源, rejected=毒素
                        acc, tox = rng.choice(hazard)
                        if acc not in tox2hn:  # 无配对难例 → 退化为 retain 一步
                            racc, rseq = rng.choice(retain); ids, lab = ids_labels(racc, rseq)
                            slp, m = seq_logp(model, ids, lab); loss = -slp / m.clamp(min=1)
                        else:
                            ir, lr = ids_labels(acc, tox)          # rejected=毒素(用毒素自己的上下文)
                            ic, lc = ids_labels(acc, tox2hn[acc])  # chosen=良性同源(同上下文)
                            lp_c, _ = seq_logp(model, ic, lc); lp_r, _ = seq_logp(model, ir, lr)
                            with torch.no_grad():
                                rp_c, _ = seq_logp(ref, ic, lc); rp_r, _ = seq_logp(ref, ir, lr)
                            loss = compute_dpo_loss(lp_c.unsqueeze(0), lp_r.unsqueeze(0),
                                                    rp_c.unsqueeze(0), rp_r.unsqueeze(0), args.dpo_beta)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    tl += float(loss.detach()); nu += 1
                    if g in milestones: save(f"step{g}")
                except Exception as e:
                    print(f"[skip] {e}", flush=True)
            print(f"[msa-{args.method}] epoch {ep+1} mean_loss={tl/max(1,nu):.4f}", flush=True)
            save(f"ep{ep+1}")
    else:
        # GRPO:给上下文生成 K 成员 → reward A(折叠)→ group advantage
        from unified_training_dpo import (TrainingConfig, ESMFoldSubprocessRunner,
            ensure_generated_pdb_for_sequence, safe_foldseek_tmscore, rmsd_CA,
            compute_reconstruction_reward, seq_identity,
            ensure_reference_pdb_for_sample, SequenceSample)
        cfg = TrainingConfig()
        esm = ESMFoldSubprocessRunner(cfg.ESM_DIR, device=dev, timeout_s=180)
        try:
            for ep in range(args.epochs):
                for _ in range(len(hazard)):
                    g += 1; acc, tox = rng.choice(hazard)
                    try:
                        prompt = build_prompt(ctx[acc]); inp = tok(prompt, return_tensors="pt").to(dev)
                        out = model.generate(inp.input_ids, do_sample=True, temperature=0.8, top_p=0.9,
                                max_new_tokens=min(1024, len(tox)+20), num_return_sequences=args.k,
                                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
                        gens, rewards = [], []
                        for o in out:
                            dec = tok.decode(o, skip_special_tokens=True); segs = dec.split("<s>")
                            gi = len(ctx[acc]) + 1
                            gen = clean_seq(segs[gi].replace(" ","")) if len(segs) > gi else ""
                            if len(gen) < 20: continue
                            # 折叠 gen 取 pLDDT + 对毒素参考结构算 TM → reward A
                            samp = SequenceSample(accession=acc, label="hazardous", sequence=tox, length=len(tox))
                            ref_pdb, _ = ensure_reference_pdb_for_sample(samp, esm, cfg)
                            gpdb, plddt = ensure_generated_pdb_for_sequence(samp, gen, esm, cfg)
                            tm = safe_foldseek_tmscore(gpdb, ref_pdb, cfg).get("qtmscore", 0.0)
                            sid = seq_identity(gen, tox)
                            r = compute_reconstruction_reward("hazardous", sid, tm, plddt)
                            gens.append(gen); rewards.append(r)
                        if len(rewards) < 2: continue
                        rt = torch.tensor(rewards, device=dev); adv = (rt - rt.mean())/(rt.std()+1e-8)
                        # policy gradient:∑ adv * logp(gen | prompt)
                        loss = 0.0
                        for gen, a in zip(gens, adv):
                            ids, lab = ids_labels(acc, gen); slp, m = seq_logp(model, ids, lab)
                            loss = loss - a * slp / m.clamp(min=1)
                        loss = loss / len(gens)
                        opt.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                        if g in milestones: save(f"step{g}")
                        print(f"[msa-grpo] step={g} {acc} meanR={rt.mean():.3f}", flush=True)
                    except Exception as e:
                        print(f"[skip] {acc}: {e}", flush=True)
                save(f"ep{ep+1}")
        finally:
            esm.close()
    print(f"[msa-{args.method}] done → {args.out}", flush=True)

if __name__ == "__main__":
    main()
