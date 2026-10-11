# full-duplex-turn-taking-eval

全双工语音模型（full-duplex speech-to-speech）**turn-taking 行为**的可复现评测工具与实验记录。

当前已包含 **PersonaPlex** 的评测代码与 10 事件实验（Open Yap 1K 样本）；架构预留多模型（后续将纳入 FireRedChat），代码按 `code/<model>/` 组织。

---

## 实验目标

测量全双工模型在真实人类对话中，对**对方犹豫现象**（停顿 pause、填充词 filler、重复 repetition、延长 lengthen、重叠 overlap）的 turn-taking 反应。当前协议将行为拆分为**四类独立指标**（定义见 `docs/protocol.md`）：

| 指标 | 含义 |
|---|---|
| `continuation` | 模型处于话轮中途被释放 → 延续自己（被回放的）语音 |
| `initiation` | 静默 ≥0.5s 后的**新开口**（真正的"决定说话"） |
| `floor_yielding` | 对方（A）发声进入时，模型**让出**话轮的延迟 |
| `no_response` | 整个测量窗内未开口（中性标签，不作对错判断） |

## 数据与模型来源

| 资源 | 来源 | 版本/许可 |
|---|---|---|
| 模型权重 | `nvidia/personaplex-7b-v1`（HF） | revision `fdaf4090…`（见 `results/run10/run_config.json`）；**NVIDIA Open Model License**，权重不入库 |
| 模型代码 | [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex)（moshi fork） | commit `3428dfd9…`；MIT |
| 数据集 | `TheAgenticDataCompany/open-yap-1k`（HF 公开样本） | 16 段双人对话 / 8.9h / 48kHz 双轨 / 词级转写；**CC-BY-4.0**；音频不入库，校验和见 `provenance/dataset_files.sha256` |
| 依赖锁定 | `provenance/requirements-frozen.txt` + `provenance/environment.md` | python 3.12.3 / torch 2.8.0+cu128 / RTX 5090 |

引用要求：Open Yap 1K（The Agentic Data Company, 2026）、PersonaPlex（Roy et al., 2026）、Moshi（Défossez et al., 2024）。

## 仓库结构

```
code/personaplex/
  smoke_harness.py                 # 核心 harness：probe1(teacher-forcing 验证) + probe2(解码延迟)
                                   #   + 10 事件回放（interleave 主 / official 对照）→ 逐帧日志+音频+文本
  probe_text_force.py              # 文本流 teacher-forcing 可行性 probe（T1 官方组合 / T2 回放扩展组合）
  diagnostics/                     # diag_mem.py / diag_mimi.py：显存泄漏定位（no_grad 缺失问题）
tools/
  openyap_event_index.py           # 事件抽取与筛选（11,595 事件 → selection_50）
  audit_protocol.py                # 声学活动检测 + onset/continuation/yielding 指标分离
  build_case_index.py              # 生成含声学修正的 case 索引
events/
  events_all.csv                   # 全量事件池（8.92h, 11,595 事件）
  selection_50.csv                 # 50 个代表事件（5 类 × 配额，含 clip 窗口）
  run10_case_index.csv             # ★ 实验 case ↔ 原始事件定位表（含声学修正时间戳、切换点）
results/
  run10/run_config.json            # 运行参数 + 版本锁定 + 对齐验证结论（协议版本 pp-replay-v2）
  run10/run10_results.json         # 12 case 汇总（onset/真人参考/文本/活动占比…）
  run10/frames/*.csv               # 逐帧时间序列（时间/A,B活动/文本/token/RMS）
  run10/text/*.txt                 # 模型输出文本（可读版 + raw token）
  probes/                          # teacher-forcing 与文本强制的验证输出
  audit/PROTOCOL_AUDIT.txt         # 每 case 的 ±2s 声学状态串等审计原始数据
docs/
  protocol.md                      # ★ 实验协议与指标定义（含已知偏差登记表）
  RUN10_REPORT.md                  # 10 事件实验报告
  PROTOCOL_AUDIT_REPORT.md         # 协议有效性审计报告（6 项检查）
  smoke_report_v1.md               # 历史：首轮冒烟（v1，token 判据，已被 v2 取代）
  listening_manifest.md            # 听检材料清单与听检顺序
provenance/
  dataset_files.sha256             # 数据集 tar/metadata 校验和（定位同一数据版本）
  requirements-frozen.txt          # 依赖锁定
  environment.md                   # 环境与版本记录
```

## 依赖安装

```bash
# 系统依赖
apt-get install -y libopus-dev ffmpeg

# 1) 获取并安装 PersonaPlex 代码（moshi fork）
git clone --depth 1 https://github.com/NVIDIA/personaplex.git /opt/personaplex
# 2) Python 环境（关键：不要动 torch —— moshi 的 pyproject 钉 torch<2.5，
#    直接带依赖安装会降级 torch 导致 Blackwell GPU 不可用）
python3 -m venv --system-site-packages .venv     # 复用已装好的 torch>=2.7(cu128)
source .venv/bin/activate
pip install --no-cache-dir --no-deps /opt/personaplex/moshi
pip install --no-cache-dir \
  "numpy>=1.26,<2.2" "safetensors>=0.4.0,<0.5" "huggingface-hub>=0.24,<0.25" \
  "einops==0.7" "sentencepiece==0.2" "sounddevice==0.5" "sphn>=0.1.4,<0.2" \
  "aiohttp>=3.10.5,<3.11" soundfile
# 精确复现可用 provenance/requirements-frozen.txt 对照
```

## 数据准备

```bash
# 1) Open Yap 1K 公开样本（约 3.8GB；非 gated）
pip install -U "huggingface_hub[cli]"
huggingface-cli download --repo-type dataset TheAgenticDataCompany/open-yap-1k \
  --local-dir data/open-yap --include "*.tar" "metadata.jsonl"
cd data/open-yap && for f in shard-*.tar; do tar -xf $f -C extracted; done  # a.flac/b.flac/json 每段对话
# 校验：对照 provenance/dataset_files.sha256
# 2) PersonaPlex 权重（gated：需在 HF 接受许可 + Read token）
export HF_TOKEN=hf_xxx
python -c "from huggingface_hub import hf_hub_download; \
  from moshi.models import loaders; \
  hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)"   # 约 17GB
```

## 运行方式

所有路径通过环境变量配置（无写死绝对路径）：

```bash
export OY_DATA_DIR=$PWD/data/open-yap       # 数据集根目录
export PP_OUT_DIR=$PWD/outputs/pp_smoke     # 运行输出目录（clip/日志/音频/结果）

# ① 事件抽取（→ events_all.csv / selection_50.csv）
python tools/openyap_event_index.py $OY_DATA_DIR/extracted outputs/events

# ② 回放实验（需先有 ① 的 selection_50.csv；输出逐帧 CSV + out.wav + text.txt）
python code/personaplex/smoke_harness.py

# ③ 协议审计（声学活动 + 指标分离；需 ② 的产物）
python tools/audit_protocol.py

# ④ 生成含声学修正的 case 索引
python tools/build_case_index.py --out events/run10_case_index.csv
```

## 当前实验协议（pp-replay-v2）

PP 扮演 B（任选一侧）。每个事件裁剪 `[t_start−45s, t_end+15s]`：

- **阶段 1（历史回放）**：B 的音频**逐帧 teacher-force 进模型自身音频流**（`moshi_tokens`，官方 voice-prompt 同款机制）；A 的音频走用户流（overlap 帧两路同时）；文本流强制 PAD。
- **阶段 2（自由测量）**：仅停止强制 B；**A 的真实音频继续逐帧输入**；模型自由生成自身流。
- **对齐保证**（三重验证）：输出↔输入延迟 = 1 帧 = max_delay（已校正）；解码延迟 δ=0；帧数一致性断言 12/12 通过；CSV↔WAV 一致差 <0.75dB。绝对时间不确定度 ≤80ms（切换点量化）。

`interleave` 为主实验模式；`official`（仅 B 历史、用户流 sine，严格贴合官方 voice-prompt 组合）为技术对照。指标定义与已知偏差登记见 **`docs/protocol.md`**。

## 已知局限（重要）

1. **⚠️ offline replay ≠ 真实互动评估**：历史对话是通过 teacher-forcing **回放**给模型的（不是真实流式互动）；模型被释放瞬间的状态与真实在线状态之间存在人为切换。本框架测量的是"**给定回放上下文后的生成行为**"，不能直接外推为在线交互质量。
2. **人类 B 的接话时刻仅作行为参考**，不是唯一正确答案（个体方差大）。
3. **转写边界 vs 声学边界 ±100–200ms 抖动**（已用声学复算修正，见 `run10_case_index.csv`）。
4. **文本流 PAD 约定**：历史回放中 B 的文本流为 PAD（官方 voice-prompt 约定）→ 可能出现"无文本音频延续"（overlap2 实例）。文本回放变体已验证可行（`results/probes/probe_text_force_output.txt`），尚未纳入协议。
5. **样本偏差**：公开样本仅 16 段对话（官方手工挑选），统计结论需扩展数据后再说。
6. `no_response` 为中性标签；本仓库不对行为做"正确/错误"判定。
7. 本仓库**不含**任何音频、模型权重与密钥；听检材料需在本地按上述流程重新生成。

## 复现元数据

随机种子 `42424242`；采样温度 0.8（音频）/0.7（文本）、top-k 250/25；帧长 80ms（12.5Hz）；
模型 revision `fdaf4090a61cb315c138a1faee287ffd6c716309`；代码 commit `3428dfd9…`；
数据集校验和见 `provenance/`。完整运行参数：`results/run10/run_config.json`。
