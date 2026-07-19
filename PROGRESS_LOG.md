# 进度日志(ProteinMPNN + ProtGPT3 安全 unlearning)

配套:`EVALUATION.md`(指标发现)、`METHODS.md`(对比协议)。更新于 2026-07-19。

---

## 一、当前状态

### 已完成
- **环境**:本地 venv(torch 2.9.1+cu126,含 Muon);fr1a NAS env(base 镜像 torch + pypi extras,`/mnt/bn/feed-quality-issues/yxu/proteinmpnn_fr1a/`,可复用)。
- **数据**:cluster split(同源聚类、cluster 级不重叠、0 泄漏);**难例两套**——train `hard_negatives_train.csv`(727,retain 用)/ test `hard_negatives.csv`(185,过度改造评测),两者 cluster 0 交集;PDB **1620**(毒/良/难例,~90% 覆盖);hazardDB(427 毒素);训练同源上下文 `homolog_context_train.json`(1244)。
- **评测基础设施**:`eval_harness_mpnn.py` / `eval_harness_protgpt_msa.py`(生成 K 候选+折叠,存**完整原始 JSONL**,支持 `--resume`);`compute_metrics.py`(全矩阵 + 守门员 complexity/diversity/novelty);`score_hazard.py`(mmseqs 打"输出有害"标);`eval_trajectory.py`(帕累托轨迹)。
- **指标定稿**:毒性主判据 **reconstruction-ASR(id + TM,TM 不用 RMSD≤2)**;ProtGPT 走 KPO 式连续位移;**难例 FP 对照做成贡献**(揭示同源/Pfam/ToxinPred 都分不开毒素与良性同源:实测 70%/97% FP);守门员防"塌成垃圾"。唯一 limitation:novel 绝对毒性计算上不可测。
- **Baseline 代码(6 个 = 3 方法 × 2 架构,已推 GitHub)**:
  - MPNN(`unified_training_dpo.py`):`--method npo`(NPO+RT,难例进 retain,log 间隔 ckpt)、`--method grpo`(`GRPO_REWARD_TYPE=reconstruction_A`,`(1−id)(1−TM)·pLDDT`,弃 FOO reward)、`--method dpo`+`seqonly_A`(fold-free)。
  - ProtGPT3-MSA(`train_msa.py`):npo/dpo/grpo,teacher-forcing 序列 logp,DPO chosen=难例良性同源/rejected=毒素。
- **MPNN-NPO smoke 通过**:修了 `hard_negative` label bug(normalize_label 映射为 benign);checkpoint 正常存(step 1/2/4/8/…)。
- **GitHub**:`crushonyfg/ForgetOrthogonalOptimization`,最新含全部上述。

### 正在跑
- **fr1a resume-eval MPNN**(job `67aa4d67f4ea6368`):续跑 base eval 剩 ~12%(跳过已有 2208)+ score + 出 base 指标,带 notify。
- **本地 GPU 空**(留给 smoke)。

### 已知数字
- base MPNN(部分):reconstruction-ASR@8 宽 96-98% / 严 92%;benign 质量 pLDDT 0.85 TM 0.83;守门员 complexity 4.1。
- MSA 攻击:生成对最近毒素 id~0.55(novel)、pLDDT~0.46。

---

## 二、设计规划

### 对比协议(见 METHODS.md)
- 三 test 集(hazardous 危害轴 / benign 灾难性遗忘 / hard-neg 过度改造),同一指标不同集含义不同;hard-neg 上相似度=FP 对照不作安全信号。
- **主轴 = (reconstruction-ASR, 效用) 帕累托轨迹,按 forget 曝光对齐**;wall-clock/训练折叠数作效率表(NPO/DPO 不折叠训练=省,GRPO 折叠=贵)。
- 动态分层记录:密集免折叠指标(每 ckpt)+ 稀疏折叠(小子集,~8 log 间隔 ckpt)+ 终点全评带 CI。
- baseline 阶梯:NPO+RT(Zhang24)/ DPO-unlearning(cite KPO)/ GRPO(reward A,cite Shao24);公平=只变机制,base/数据含难例/曝光预算/评测/optimizer(Adam)全锁死。预算=**3 遍 hazard train ≤300aa**。
- 自研方法未定(候选:retain-正交投影 / RMU 表征级 / 难例锚定 detoxify);baseline 越标准,赢得越干净。

### fr1a job 规范(关键)
- **fr1a egress 只有 pypi 通;github clone 和 download.pytorch.org 都不通** → 代码/权重/数据/DB 走"maliva 打包 → NAS → 解压",torch 用镜像自带(baseline 用 Adam 不需 Muon)。
- **fork 继承镜像**(`source_job_run_id=7b9812728897459e`),不 from_scratch 猜 image。
- 资源在 `resource_config.arnold_config`(group 479/cluster 14/queue…/bytenas feed-quality-issues),gpuv 下划线 `H100_SXM_80GB`,control-plane `i18n-tt`。
- **必带 `notify_settings`**(APP,JOB_FAILED/DONE/QUEUE_TIMEOUT),否则 Lark 无消息。
- entrypoint 用 `set -e`(否则中途失败也 DONE),或逐步 check marker。

---

## 三、待办

- [ ] fr1a resume-eval 完成 → 看 base 全矩阵指标(带 CI)。
- [ ] MSA base eval 续跑(需把 ProtGPT3-MSA 弄到 NAS/fr1a;527/768 已有)。
- [ ] DPO、GRPO 的 smoke(验证另两个方法,尤其 GRPO 折叠 reward)。
- [ ] 训练难例 AF 补齐(655/727)。
- [ ] 按对比协议正式跑 6 个 baseline(fr1a 并行,notify);MSA-GRPO ~20h。
- [ ] 定"我们的方法"并实现(可能用 Muon → 届时解决 fr1a torch 2.9:内网 pip mirror 或自定义镜像)。
- [ ] 补:conservation mask、更大 pass@k、预训练污染检查、自适应再攻击鲁棒性。

---

## 四、经验 / 教训

- **不私下偷懒改/省参数**:torch 版本、notify_settings、进程——改任何参数都明说,不静默省略/替换。
- **不批量 kill 进程**:PPID=1 ≠ 可杀(nohup/detach 也是 PPID=1);曾误杀用户 pearl 训练进程。只杀能 100% 确认是自己的(精确命令名 + 自己 spawn)。
- **写入只限 `/mnt/bn/feed-quality-issues/yxu/`**,绝不碰别人目录;tar 用相对路径 + `-C` 解压。
- **GPU 泄漏**:killed job 的 ESMFold worker 会留僵尸占显存,不自动释放;清理要精确到自己的进程树。
- **Merlin**:fork 继承镜像最省事;notify 必带;资源结构 arnold_config;dry-run 校验后再提交(dry-run 不做全校验,真提交才报 arnold_config 缺失等)。
- **fr1a egress 限制**(只 pypi);**base 镜像启动带 code-server**(日志有扩展安装噪声)。
- **smoke 的价值**:NPO smoke 直接抓出 hard_negative label bug——正式跑前必 smoke。
- **NAS 读慢**(fr1a 卷跨区,maliva 侧 ~9MB/s):验证优先看 NAS 产物文件而非拉日志;小文件多用 tar。
