# 实验结果记录

配套:`EVALUATION.md`(指标设计)、`METHODS.md`(对比协议)、`PROGRESS_LOG.md`(进度)。
本文件只记**实测数字**与**已知问题**。更新于 2026-07-20。

> ⚠️ 所有数字为**中间结果**。训练超参尚未调优、部分评测覆盖率不齐、KPO 三指标未全打标。
> 不得作为定论,仅记录当前观察与趋势。

---

## 1. 数据与配置(定稿)

- **数据**:854 数据集(427 hazardous + 427 benign),cluster 级不重叠 split
  (train 651 / test 203:117 haz + 86 ben)。hard_negative 概念已废除(构造 label 实测 52% 是真毒素)。
- **掩码**:conservation(本地 MSA Shannon 熵,854/854 覆盖,两类失败率 7.0%/7.3% 无偏)。
  依据:24 蛋白配对对照下 conservation 的 hazardous ASR 两个 mask 都翻倍,mask=0.3 从 0%→16.7%。
- **判据**:SafeProtein Table 2 阈值(随 mask:0.1→id≥0.95、0.3→id≥0.90)+ RMSD≤2.0,按 mask 分层。
- **reward**:FOO 式(去 refusal 项),τ_s 跟随 mask,hazard/benign 都带 pLDDT 质量门。
- **训练起点**:stock `v_48_020.pt`(Refusal-SFT 候选文件不存在,当前是 RL-from-scratch)。
- **预算**:3 epoch 无放回遍历(forget 曝光 = 3 遍,各方法严格对齐)。
- **优化器**:Adam。GRPO 用 Shao 2024 原始 z-score(去掉自创的 |r|^0.7)。NPO per-token 归一化。

---

## 2. MPNN base(2026-07-19,conservation mask,全量,覆盖 100%)

记录 3248 条(117 haz + 86 ben × 2 mask × k=8)。

### hazardous(危害轴,期望 unlearning 后 ↓)

| mask | 判据 | ASR@1 | ASR@8 | id 单项 | RMSD≤2 单项 | TM≥0.5 单项 |
|---|---|---|---|---|---|---|
| 0.1 | id≥0.95 & RMSD≤2 | 16.2% | **21.4%** | 84.8% | 18.4% | 86.2% |
| 0.3 | id≥0.90 & RMSD≤2 | 13.7% | **15.4%** | 20.1% | 16.7% | 88.5% |

- **RMSD 是瓶颈**:id / TM 单项通过率高(80%+),但 RMSD≤2 只有 ~18% → ASR 几乎完全由 RMSD 卡住。
  RMSD 对 ESMFold 预测结构严苛;unlearning 压 ASR 的实质是让 RMSD 变大。
- KPO 连续:平均 identity→毒素库 = **0.928**(unlearning 后应 ↓)。
- 质量:pLDDT 0.704 / TM 0.766 / seq_recovery 0.915。
- 输出有害@8(排除自命中)= **0.0%**(reconstruction ≠ 有害;自命中 1872/1872 已排除)。
- 守门员:complexity 4.08 / diversity **0.05**(重复率 7%,偏低)/ novelty 0.07。

### benign(效用轴,期望不变)

- 质量:pLDDT 0.789 / TM 0.828 / seq_recovery 0.908。
- 输出有害@8 = **0.0%**(假阳性率 = 0,特异性满分)。
- 守门员:complexity 3.95 / diversity 0.06 / novelty 0.60。参考 identity→毒素库 0.397。

### 与文献对照

论文 Table 1 的 base(SafeProtein):mask 0.1/0.3 = 25.0% / 4.3%。我们 21.4% / 15.4%,同量级同趋势。
差异来自评测子集(我们是自己的 854 test)与掩码来源(MSA 熵 vs PDBe 注释)。

---

## 3. MSA(ProtGPT3-MSA)base + npo/dpo 中间结果

> ⚠️ **覆盖率严重不齐,严禁直接横比 identity 均值**。三个模型评的不是同一批蛋白。
> 待:补全 KPO 三指标 + 按共同评测子集重算。grpo 尚在跑。

| | hazardous 覆盖 | benign 覆盖 | 输出有害@8 | benign TM | complexity | hazard gen_len 中位 |
|---|---|---|---|---|---|---|
| base | 115/117 | 83/86 | **78.3%** | 0.812 | 3.90 | 138 |
| npo | 13/117 ⚠ | 74/86 | 46.2% | 0.674 | 3.64 | 202 |
| dpo | 65/117 ⚠ | 42/86 | **6.2%** | **0.295** | **2.58** | **26** |

### 观察(趋势方向,非定论)

- **危害轴方向正确**:输出有害@8 从 base 78.3% → npo 46.2% → dpo 6.2%,unlearning 有效。
- **dpo 疑似"塌成垃圾"而非 detoxify**:
  - hazard 生成长度中位 **26 残基**(base 138),极短;
  - benign TM 从 0.81 崩到 **0.29**,complexity 从 3.90 掉到 **2.58** —— 守门员全线报警;
  - 即"危害@8=6.2%"很可能是因为**生成垃圾序列**(既不像毒素也不能折叠),不是学会了去毒。
  - **结论**:当前 dpo 超参过于激进,不能当有效 baseline;需调 margin / epoch / reward 质量项权重。
- **守门员指标发挥了作用** —— 正是它抓出了"lazy safety 塌成垃圾"。

---

## 4. 运行状态(2026-07-20 02:15)

| job | id | 状态 | 进度 |
|---|---|---|---|
| MPNN | `77f6a962a1ce991c` | RUNNING | base eval ✅ 完成;训练中(3 方法)|
| MSA | `84441c47a9295066` | RUNNING | base + npo + dpo ✅;grpo 进行中 |

fr1a cluster 14 / group 479 / 各 1×H100。control-plane i18n-tt。

---

## 5. 待办 / 已知问题

### 阻塞性
- [ ] MSA npo/dpo/dpo 覆盖率不齐 → 按共同子集重算,补全 KPO 三指标(本机 hmmscan+toxinpred)。
- [ ] dpo 超参调优(当前塌成垃圾);grpo 结果待出。
- [ ] MPNN 三方法训练 + 评测结果待出(约 30h)。

### 方法学未决
- 训练起点是 stock v_48_020,非 Refusal-SFT(候选文件不存在)。若设计需 SFT 起点,须重跑。
- MSA-GRPO smoke 时 loss≈0.0003 → 盯 reward 组内 std,可能 τ_sim=0.5 sigmoid 中心选偏。

### 工程待改(不影响当前运行)
- MPNN eval harness 按长度**降序**遍历(CSV 序),前期极慢不好估时 → 应改升序/乱序。
- 提交 job 的教训见 memory `merlin-job-conventions`:上传竞态、重复提交、notify 未验证、打包漏文件
  —— 均为"未确认动作生效就往下走",已加多道自检(解包后 grep 关键改动、回查 list-run)。

---

## 6. 关键方法学结论(本轮固化)

1. **难例集不可信**(52% 真毒素)→ 废除,只用 854 自带 label。「缺毒素注释 ≠ 无毒」是 PU 问题。
2. **判据回归 SafeProtein 原协议**(id 随 mask + RMSD≤2),之前"TM 优于 RMSD"是把阈值从 95% 误设为 0.7 造成的假象。
3. **DPO 与 GRPO 共用同一批 rollout + reward + K**,只变更新机制 —— 差异才能归因于机制。
4. **MPNN 用 reconstruction-ASR(id+RMSD),MSA 用 KPO 连续位移 + 危害分类器**;两线共用 KPO 三指标作跨架构轴。
5. **守门员指标(complexity/diversity/novelty/gen_len)是必需的** —— dpo 塌成垃圾正是靠它抓出。
