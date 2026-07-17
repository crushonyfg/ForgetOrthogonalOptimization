# ForgetOrthogonalOptimization — ProteinMPNN DPO / GRPO alignment

ProteinMPNN（蛋白反折叠模型）的偏好对齐实验代码：**DPO**，与 **GRPO** 做 apples-to-apples 对照。
生成序列用 **ESMFold** 折叠、**foldseek** 算结构 TM-score 作为 refusal-aware cluster reward，
优化器支持 **Muon / Adam**。

> 本仓库只放**自定义训练代码 + 数据集 CSV**。ProteinMPNN 官方代码/权重、ESMFold 权重、
> foldseek 二进制、venv 均不入库（见 `.gitignore`），按下方步骤各自获取。

## 仓库内容

| 文件 | 说明 |
|---|---|
| `unified_training_dpo.py` | 训练主脚本（`--method dpo/grpo/sft/sft_refusal/sft_ga/eval_cluster/prefetch_pdbs`）|
| `run_dpo_autoresume.sh` | DPO 启动器，遇 segfault(139) 自动从最新 checkpoint 续跑；所有路径由 `ROOT` 自动推导，可用环境变量覆盖 |
| `data/*.csv` | 数据集：cluster/random split 的 train/test + 854 全表 + 若干 eval 结果 |
| `requirements.txt` / `setup_env.sh` | 依赖锁定 + 一键重建 venv |

## 搭建步骤

```bash
# 1) 取 ProteinMPNN 官方代码 + 权重
git clone https://github.com/dauparas/ProteinMPNN.git
# 2) 把本仓库的训练代码放进 ProteinMPNN（与 protein_mpnn_utils.py 同级）
cp unified_training_dpo.py run_dpo_autoresume.sh ProteinMPNN/
mkdir -p ProteinMPNN/dataset && cp data/dataset_cluster_split_*.csv ProteinMPNN/dataset/
# 3) 建 venv（torch 2.9.1+cu126，含内置 torch.optim.Muon）
bash setup_env.sh
# 4) ESMFold 权重（~8G，放在 ProteinMPNN 的上级目录 models/ 下）
hf download facebook/esmfold_v1 --local-dir models/esmfold_v1
# 5) foldseek 二进制
mkdir -p ProteinMPNN/bin && (cd ProteinMPNN/bin && \
  wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz && tar xzf foldseek-linux-avx2.tar.gz)
```

## 运行 DPO

```bash
cd ProteinMPNN
source ../.venv/bin/activate
bash run_dpo_autoresume.sh
```

- 默认用官方 `vanilla_model_weights/v_48_020.pt` 初始化；有 refusal SFT ckpt 时 `export INIT_CKPT=/path/ckpt.pt` 覆盖。
- 优化器默认 `muon`（需 torch≥2.9），可 `GRPO_OPTIMIZER=adam`。
- 关键可覆盖环境变量：`ROOT` `ESMFOLD_MODEL` `FOLDSEEK_BIN` `INIT_CKPT` `RUN_DIR` `ORIG_TRAIN_CSV` `ORIG_TEST_CSV`。

## 依赖要点

torch **2.9.1+cu126**（`torch.optim.Muon` 自 torch 2.9 起内置，2.8 及以下没有）、
transformers 4.53.3、biopython、biotite、einops。完整见 `requirements.txt`。
