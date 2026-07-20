#!/usr/bin/env python
"""ProtGPT3-MSA baseline 训练:NPO+RT / DPO / GRPO。

与 MPNN 侧(unified_training_dpo.py)保持同构 —— 同一 base、同一 forget 曝光预算、
KL 锚 base、log 间隔 checkpoint,**DPO 与 GRPO 共用同一批 rollout 与同一个 reward**,
只有更新方式不同(成对偏好损失 vs 组内优势策略梯度)。这样二者差异只能归因于机制本身。

数据规定(2026-07-19 定死):只用 854 数据集的 hazardous/benign;**hard_negative 概念已废除**
(其构造 label 实测 52% 是真毒素,见 EVALUATION.md §0)。

reward(与 MPNN 同形,只有 hazard 的相似度来源不同):
  hazard: σ(k(τ_sim − sim_hazDB)) · pLDDT^γ
      —— ProtGPT 自由生成长度可变,对"特定目标"的 id/RMSD 无意义(生成的是家族新成员),
         故危害轴改用**对毒素库的 mmseqs 相似度**(KPO 式,长度无关、免折叠)。
         实测(200 条 base MPNN 生成,排除自命中):mmseqs bits hazardous 290 vs benign 0,
         而 Pfam(1.45 vs 1.72,方向反)与 ToxinPred3(0.286 vs 0.237)判别力不足,
         故 reward 只用 mmseqs,三个 KPO 指标全部进**评测**(score_kpo.py)。
  benign: pLDDT^γ · TM · σ(k(id − τ_s))    —— 与 MPNN benign 完全一致

用法:
  export ESMFOLD_MODEL=... FOLDSEEK_BIN=...
  python train_msa.py --method dpo --model /path/ProtGPT3-MSA \
     --train-csv dataset/dataset_cluster_split_train.csv \
     --context ../dbs/homolog_context_train.json --epochs 3 --out outputs/msa_dpo
"""
import os, json, math, random, argparse, subprocess, tempfile

AA = set("ACDEFGHIKLMNPQRSTVWY")

def process_style(seq, gap=False):
    """对齐 ProtGPT3-MSA 官方 build_prompt 的 process_style(见模型卡)。

    官方 no_gap:``re.sub(r"[X]", "", seq.replace("-","").upper())`` —— 只去 gap 与 X,
    **保留 B/Z/U/O 等非标准残基**。此前我们用只留 20 种标准氨基酸的 clean_seq,会多删字符,
    造成与训练分布的偏离。
    """
    import re as _re
    return _re.sub(r"[X]", "", seq.upper() if gap else seq.replace("-", "").upper())

def clean_seq(s):
    """仅用于**目标序列**的清洗(需要严格 20 AA,因为要送 ESMFold 折叠)。"""
    return "".join(c for c in s.upper() if c in AA)

HERE = os.path.dirname(os.path.abspath(__file__))


def build_prompt(seqs, direction="1", rng=None):
    """构造同源 few-shot prompt。

    2026-07-19 **修 DPO/GRPO 梯度错误**:此前这里做就地 random.shuffle,而 ids_labels 每次
    调用都重建 prompt —— chosen / rejected / ref_chosen / ref_rejected 四次前向用的是
    **四个不同顺序的 prompt**。DPO 的推导前提是同一 prompt 下的成对偏好;GRPO 同理
    (采样用 prompt A、算 logp 用 prompt B)。现在调用方构造一次并复用同一个 prompt。
    """
    seqs = list(seqs)
    (rng or random).shuffle(seqs)
    toks = ["<|bos|>", direction, "<no_gap>"]
    for s in seqs:
        toks.append("<s>"); toks.extend(list(process_style(s)))
    toks.append("<s>")
    return " ".join(toks)


def mmseqs_max_similarity(seqs, mmseqs_bin, hazard_db, exclude_acc=None, threads=8):
    """一批序列对毒素库的**最大相似度**(fident × qcov,有界 0–1)。返回 list[float]。

    KPO 用的是 alignment score(bit score),但 bits 无界且随长度增长,不适合直接进 reward;
    fident × qcov 保留了"比对多长 + 多像"两个维度且有界。评测侧仍按 KPO 报 bits(score_kpo.py)。
    ``exclude_acc``:排除命中目标自身(MPNN 侧实测自命中率 99.8%,不排除则指标恒满分)。
    """
    if not seqs:
        return []
    tmp = tempfile.mkdtemp(prefix="msa_sim_")
    fa = os.path.join(tmp, "q.fasta")
    with open(fa, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">s{i}\n{s}\n")
    out = os.path.join(tmp, "hits.m8")
    subprocess.run(
        [mmseqs_bin, "easy-search", fa, hazard_db, out, os.path.join(tmp, "t"),
         "--max-seqs", "10", "-e", "1e-3", "--threads", str(threads), "-v", "0",
         "--format-output", "query,target,fident,qcov"],
        capture_output=True, check=False,
    )
    best = [0.0] * len(seqs)
    if os.path.exists(out):
        for line in open(out):
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            q, t, fid, qcov = p[0], p[1], float(p[2]), float(p[3])
            if exclude_acc and str(t) == str(exclude_acc):
                continue
            i = int(q[1:])
            best[i] = max(best[i], fid * qcov)
    import shutil as _sh
    _sh.rmtree(tmp, ignore_errors=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["npo", "dpo", "grpo"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--context", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--npo-beta", type=float, default=0.1)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--dpo-margin", type=float, default=0.02)
    ap.add_argument("--dpo-max-pairs", type=int, default=3)
    ap.add_argument("--kl-beta", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=6,
                    help="每 prompt 的 rollout 数。**DPO 与 GRPO 必须相同**(公平对比)。")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    # 2026-07-19 弃用:改为无放回遍历后,hazard/benign 比例由数据集本身决定(全局比例),
    # 不再由这个参数控制。保留仅为兼容旧命令行,值被忽略。
    ap.add_argument("--max-ctx", type=int, default=4096,
                    help="prompt+目标的最大 token 数。超长时削减同源条数而非截断目标段。")
    ap.add_argument("--tau-sim", type=float, default=0.5,
                    help="hazard reward 的相似度 sigmoid 中心(对毒素库的 fident×qcov)")
    ap.add_argument("--tau-seqid", type=float, default=0.7,
                    help="benign reward 的 identity sigmoid 中心")
    ap.add_argument("--sigmoid-k", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=1.0, help="pLDDT 指数")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mmseqs", default=os.path.join(HERE, "bin/mmseqs/bin/mmseqs"))
    ap.add_argument("--hazard-db", default=os.path.join(HERE, "../dbs/hazardDB"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch, torch.nn.functional as F
    import pandas as pd
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from unified_training_dpo import (
        compute_npo_loss, compute_dpo_loss, _build_dpo_pairs_from_rewards,
        compute_group_relative_advantages, mean_plddt_to_quality_0_1, sigmoid,
        TrainingConfig, ESMFoldSubprocessRunner, ensure_generated_pdb_for_sequence,
        safe_foldseek_tmscore, seq_identity, ensure_reference_pdb_for_sample,
        SequenceSample, assert_foldseek_available, disable_dropout,
    )

    # 全局播种(此前全脚本无 torch/numpy 播种,run 不可复现)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    ctx_raw = json.load(open(args.context))
    # 新格式(build_msa_context.py):{acc: {label, n_homologs, n_pool, homologs}}
    # 兼容旧格式 {acc: [seq, ...]}
    ctx = {a: (v["homologs"] if isinstance(v, dict) else v) for a, v in ctx_raw.items()}
    ctx = {a: v for a, v in ctx.items() if v}   # 零同源的无法构造 prompt,跳过
    print(f"[ctx] {len(ctx)}/{len(ctx_raw)} 个可用上下文(零同源的已排除)", flush=True)
    cfg = TrainingConfig()
    assert_foldseek_available(cfg)

    # **只吃 854**:难例已废除,不再有 --hardneg-csv
    tr = pd.read_csv(args.train_csv); tr["accession"] = tr["accession"].astype(str)
    hazard = [(r.accession, r.sequence) for r in tr[tr.label == "hazardous"].itertuples()
              if r.accession in ctx]
    retain = [(r.accession, r.sequence) for r in tr[tr.label == "benign"].itertuples()
              if r.accession in ctx]
    print(f"[msa-{args.method}] hazard={len(hazard)} retain={len(retain)} (仅 854,无难例)", flush=True)
    if not hazard or not retain:
        raise RuntimeError("hazard 与 retain 都必须非空。benign 上下文缺失时请先重建 context。")

    # 对齐官方模型卡:trust_remote_code + padding_side="left"(BOS 在 build_prompt 里手动加)
    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
        add_bos_token=False, add_eos_token=False, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev)
    model.train()
    ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev).eval()
    for p in ref.parameters():
        p.requires_grad = False
    # policy 与 reference 的 dropout 必须一致关闭,否则 step0 的隐式 reward 就有系统性偏差
    disable_dropout(model); disable_dropout(ref)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    rng = random.Random(args.seed)

    def make_prompt(acc):
        """构造 prompt 字符串。**同一步内的多次前向必须复用同一个返回值。**"""
        return build_prompt(ctx[acc], rng=rng)

    def ids_labels(acc, target_seq, prompt=None):
        """prompt + 目标序列;labels 只在目标段。

        2026-07-19 修 **4096 截断导致的静默零梯度**:实测 prompt token 中位数 2202、最大 7773,
        96 个 test 上下文里 14 个超 4096;超长时 labels 被整段截掉 → mask.sum()==0 →
        loss 恒 0 → backward 无梯度且完全静默。现在优先削减同源条数,仍放不下则抛错。
        """
        if prompt is None:
            prompt = make_prompt(acc)
        t_ids = tok(" ".join(list(clean_seq(target_seq))), add_special_tokens=False).input_ids
        if len(t_ids) + 8 > args.max_ctx:
            raise ValueError(f"{acc}: 目标 token 数 {len(t_ids)} 超出 max_ctx={args.max_ctx}")
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        if len(p_ids) + len(t_ids) > args.max_ctx:
            homs = list(ctx[acc])
            while len(homs) > 1:
                homs = homs[:-1]
                prompt = build_prompt(homs, rng=rng)
                p_ids = tok(prompt, add_special_tokens=False).input_ids
                if len(p_ids) + len(t_ids) <= args.max_ctx:
                    break
            if len(p_ids) + len(t_ids) > args.max_ctx:
                raise ValueError(f"{acc}: 即使只保留 1 条同源仍超长")
        ids = p_ids + t_ids
        lab = [-100] * len(p_ids) + t_ids
        assert sum(1 for x in lab if x != -100) == len(t_ids), "目标段被截断,拒绝静默零梯度"
        return torch.tensor([ids], device=dev), torch.tensor([lab], device=dev)

    def seq_logp(m, ids, lab):
        out = m(input_ids=ids)
        lp = F.log_softmax(out.logits[0, :-1], dim=-1)
        tgt = ids[0, 1:]
        tl = lp[range(tgt.shape[0]), tgt]
        mask = (lab[0, 1:] != -100).float()
        return (tl * mask).sum(), mask.sum()

    # ---------- rollout(DPO 与 GRPO 共用) ----------
    esm = None

    def rollout(acc, native_seq, label, prompt):
        """从 prompt 采 K 条,算 reward。返回 (gens, rewards)。

        **DPO 与 GRPO 调用同一个函数、同一组参数** —— 只变机制,其余锁死。
        prompt 由调用方传入并在后续打分时复用(保证 on-policy)。
        """
        inp = tok(prompt, return_tensors="pt").to(dev)
        out = model.generate(
            inp.input_ids, do_sample=True, temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=min(1024, len(native_seq) + 20), num_return_sequences=args.k,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        )
        gens = []
        n_ctx = len(ctx[acc])
        for o in out:
            dec = tok.decode(o, skip_special_tokens=True)
            segs = dec.split("<s>")
            gen = clean_seq(segs[n_ctx + 1].replace(" ", "")) if len(segs) > n_ctx + 1 else ""
            gens.append(gen if len(gen) >= 20 else None)
        valid_idx = [i for i, g in enumerate(gens) if g]
        if len(valid_idx) < 2:
            return [], []

        samp = SequenceSample(accession=acc, label=label, sequence=native_seq,
                              length=len(native_seq))
        ref_pdb, _ = ensure_reference_pdb_for_sample(samp, esm, cfg)
        rewards = [float("nan")] * len(gens)
        if label == "hazardous":
            sims = mmseqs_max_similarity([gens[i] for i in valid_idx], args.mmseqs,
                                         args.hazard_db, exclude_acc=acc)
            for j, i in enumerate(valid_idx):
                _, plddt = ensure_generated_pdb_for_sequence(samp, gens[i], esm, cfg)
                q = mean_plddt_to_quality_0_1(plddt) ** args.gamma
                # 危害轴:对毒素库越不像越好;质量门防"塌成垃圾"拿满分
                rewards[i] = sigmoid(args.sigmoid_k * (args.tau_sim - sims[j])) * q
        else:
            for i in valid_idx:
                gpdb, plddt = ensure_generated_pdb_for_sequence(samp, gens[i], esm, cfg)
                tm = safe_foldseek_tmscore(gpdb, ref_pdb, cfg).get("qtmscore", float("nan"))
                if tm != tm:
                    continue
                q = mean_plddt_to_quality_0_1(plddt) ** args.gamma
                sid = seq_identity(gens[i], native_seq)
                # benign 与 MPNN 完全一致:质量 × 结构 × 忠实度
                rewards[i] = q * float(np.clip(tm, 0, 1)) * sigmoid(
                    args.sigmoid_k * (sid - args.tau_seqid))
        keep = [i for i in valid_idx if rewards[i] == rewards[i]]
        return [gens[i] for i in keep], [rewards[i] for i in keep]

    from collections import Counter as _Counter
    skip_reasons = _Counter()
    total = args.epochs * (len(hazard) + len(retain))
    milestones = sorted(set([1] + [2 ** i for i in range(20) if 2 ** i <= total] + [total]))
    g = 0
    n_updates = 0

    def save(tag):
        model.save_pretrained(os.path.join(args.out, tag))
        tok.save_pretrained(os.path.join(args.out, tag))

    try:
        if args.method in ("dpo", "grpo"):
            esm = ESMFoldSubprocessRunner(cfg.ESM_DIR, device=dev, timeout_s=180)
        for ep in range(args.epochs):
            # 2026-07-19 与 MPNN 侧一致:**无放回遍历**,一个 epoch 每条蛋白恰好一次。
            # 此前是 rng.choice 有放回抽样,一个 epoch 只覆盖约 63% 的蛋白,
            # forget 曝光次数由随机种子决定 —— METHODS.md §3 的主对齐轴形同虚设。
            h_ord, b_ord = list(hazard), list(retain)
            rng.shuffle(h_ord); rng.shuffle(b_ord)
            nh, nb = len(h_ord), len(b_ord)
            epoch_order, ih, ib = [], 0, 0
            for _t in range(nh + nb):
                if ih >= nh:
                    epoch_order.append((b_ord[ib], "benign")); ib += 1
                elif ib >= nb or (ih / max(1, ih + ib)) < (nh / (nh + nb)):
                    epoch_order.append((h_ord[ih], "hazardous")); ih += 1
                else:
                    epoch_order.append((b_ord[ib], "benign")); ib += 1
            assert len(epoch_order) == nh + nb
            n_steps = len(epoch_order)
            tl, nu = 0.0, 0
            print(f"[msa-{args.method}] epoch {ep+1}: 遍历 {n_steps} 条"
                  f"(hazard {nh} / benign {nb}),每条恰好一次", flush=True)
            for (acc, seq), label in epoch_order:
                g += 1
                is_h = label == "hazardous"
                try:
                    if args.method == "npo":
                        ids, lab = ids_labels(acc, seq)
                        slp, n_tok = seq_logp(model, ids, lab)
                        with torch.no_grad():
                            slp_r, _ = seq_logp(ref, ids, lab)
                        if is_h:
                            loss = compute_npo_loss(slp.unsqueeze(0), slp_r.unsqueeze(0),
                                                    args.npo_beta, n_tokens=n_tok.unsqueeze(0))
                        else:
                            nll = -slp / n_tok.clamp(min=1)
                            loss = nll + args.kl_beta * ((slp_r - slp) / n_tok.clamp(min=1)).abs()
                    else:
                        # DPO 与 GRPO:同一 rollout、同一 reward、同一 K,只有更新方式不同
                        prompt = make_prompt(acc)
                        gens, rewards = rollout(acc, seq, label, prompt)
                        if len(rewards) < 2:
                            skip_reasons["rollout<2有效生成"] += 1
                            continue
                        rt = torch.tensor(rewards, dtype=torch.float32, device=dev)
                        if args.method == "dpo":
                            pairs = _build_dpo_pairs_from_rewards(
                                rt, margin=args.dpo_margin, max_pairs=args.dpo_max_pairs)
                            if not pairs:
                                skip_reasons[f"无有效偏好对(gap<{args.dpo_margin})"] += 1
                                continue
                            loss = 0.0
                            for ci, ri in pairs:
                                ic, lc = ids_labels(acc, gens[ci], prompt=prompt)
                                ir, lr = ids_labels(acc, gens[ri], prompt=prompt)
                                lp_c, mc = seq_logp(model, ic, lc)
                                lp_r, mr = seq_logp(model, ir, lr)
                                with torch.no_grad():
                                    rp_c, _ = seq_logp(ref, ic, lc)
                                    rp_r, _ = seq_logp(ref, ir, lr)
                                loss = loss + compute_dpo_loss(
                                    lp_c.unsqueeze(0), lp_r.unsqueeze(0),
                                    rp_c.unsqueeze(0), rp_r.unsqueeze(0), args.dpo_beta)
                            loss = loss / len(pairs)
                        else:
                            # GRPO:Shao 2024 组内 z-score,无任何非线性 reward 变换
                            adv = compute_group_relative_advantages(rt)
                            loss = 0.0
                            for gen, a in zip(gens, adv):
                                ids, lab = ids_labels(acc, gen, prompt=prompt)
                                slp, m = seq_logp(model, ids, lab)
                                loss = loss - a * slp / m.clamp(min=1)
                            loss = loss / max(1, len(gens))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    tl += float(loss.detach()); nu += 1; n_updates += 1
                    if g in milestones:
                        save(f"step{g}")
                except Exception as e:
                    skip_reasons[f"异常:{type(e).__name__}"] += 1
                    print(f"[skip] {acc} ({label}): {type(e).__name__}: {e}", flush=True)
            # 2026-07-19：跳过必须归因。此前两处 continue 无任何输出,
            # smoke 里 updates=4/6 时看不出是 rollout 不足还是没配上对。
            print(f"[msa-{args.method}] epoch {ep+1} mean_loss={tl/max(1,nu):.4f} "
                  f"updates={nu}/{n_steps}"
                  + (f"  跳过原因={dict(skip_reasons)}" if skip_reasons else ""), flush=True)
            save(f"ep{ep+1}")
    finally:
        if esm is not None:
            esm.close()

    # 零更新即失败:拒绝把未经训练的模型当产物保存(MPNN 侧同样的守卫)
    if n_updates == 0:
        raise RuntimeError(
            f"msa-{args.method}: 完成时更新数为 0 —— 没有任何梯度更新。"
            f"常见原因:rollout 全被 skip、reward 全相同导致无有效偏好对、上下文缺失。"
        )
    print(f"[msa-{args.method}] done, updates={n_updates} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
