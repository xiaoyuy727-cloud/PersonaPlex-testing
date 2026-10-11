# 协议有效性审计报告（10 事件）

> 2026-10-11。回答你提出的 6 项检查。原始数据：`results/audit/PROTOCOL_AUDIT.txt`（逐 case 的 ±2s 声学状态串）。

---

## 1. 切换点 ±2s 的 A/B 真实声学活动（第 1 项）

逐 case 状态串已写入 `PROTOCOL_AUDIT.txt`（80ms/格，左 A 右 B，`.`=静音，阈值=各自底噪+15dB，底噪取全窗 20 分位）。**汇总**：

| case | 切换点 A | 切换点 B | B 连续说话段（跨切换点） | 备注 |
|---|---|---|---|---|
| pause1 (×2 模式) | 说 | 静 | — | 转写停顿 1.2s；**声学实际静音 328.02–328.90（0.88s）** |
| pause2 | 静 | 静 | — | 转写间隙 1.2s；声学静音 ~0.56s |
| filler1 | 说 | 静 | — | — |
| filler2 | 说 | **说** | 722.66–722.98（+0.08s 即尽） | — |
| repetition1 | 说 | **说** | 1161.67–1162.79（+0.16s） | — |
| repetition2 | 说 | **说** | 2179.30–2179.62（+0.08s） | — |
| lengthen1 | 说 | **说** | 117.49–**117.57（恰在切换点结束）** | — |
| lengthen2 | 说 | 静 | — | — |
| overlap1 | 静* | **说** | 72.73–75.21（+2.40s） | *A 声学插话点 73.05s（转写 72.85 → **200ms 提前抖动**） |
| overlap2 | 静* | **说** | 228.89–252.97（+0.56s） | *A 声学插话点 252.49s |

**再次确认转写边界 vs 声学边界存在 ±100~200ms 抖动**（pause1、overlap1/2、pause2 均可见）→ 正式实验建议事件边界以声学为准重定义。

## 2. overlap1/2：+0.04s onset 的本质（第 2 项）——**是延续，不是发起**

- 切换点 B（真人）**确实正在说话**（声学确认，上表）；
- PP 输出**前 5 帧全部为语音级能量（-13~-27dB，阈值 -53dB）**——从自由窗第一帧起就是连续语音，无静音间隙 → 是**延续被 teacher-force 的 B 语音**，不是新的 turn-taking 决策；
- 延续时长：overlap1 = **0.40s**（真实剩余 2.40s，模型提前收口）；overlap2 = **0.96s**（真实剩余 0.56s，模型多说 ~0.4s）。overlap2 延续期间**文本流恒为 PAD**（§4 解释）；
- 对照：另外 4 个"切换时 B 在说"的 case（filler2/lengthen1/repetition1/2，B 剩余 ≤0.16s）**没有延续段**（首帧 -58dB = 已知首帧解码瞬态，随后 -73dB 静音）→ "释放本身不触发说话"的结论再次成立；
- **真正的 turn-taking 事件在之后**：overlap1 在 +4.68s、overlap2 在 +3.80s 才出现新的发起段（且 A 的声学插话在延续段内 → **让出延迟 = 0.16s / 0.88s**，见 §3）。

**结论：此前的 "+0.04s onset" 把延续误并入了首 onset——你的担忧成立，指标已拆开。**

## 3. 指标分离（第 3 项）

三类指标独立记录（完整数据在 PROTOCOL_AUDIT.txt，每条含说话段列表）：

| case | continuation | initiation（延迟） | floor yielding（A 进入 → PP 停止，延迟） | 标签 |
|---|---|---|---|---|
| pause1 interleave | — | +6.04s | 0.40/0.88/2.24s（3 次） | 发起 |
| pause1 official | — | +4.68s | — | 发起 |
| pause2 | — | — | — | **no-response** |
| filler1 | — | +3.08s | 0.64s | 发起 |
| filler2 | 无（≤1 帧） | — | — | **no-response** |
| repetition1 | 无（≤2 帧） | +8.92s | 4.32/5.76s | 发起 |
| repetition2 | 无（≤1 帧） | +3.96s | 0.32/0.56/1.04/1.28/0.00s | 发起 |
| lengthen1 | 无（≤1 帧） | +5.48s | 1.44s | 发起 |
| lengthen2 | — | +7.96s | — | 发起 |
| overlap1 | **0.40s** | +4.68s | 0.24/0.72/0.40/1.76/2.80s（5 次） | 延续+让出+再发起 |
| overlap2 | **0.96s** | +3.80s | 0.08s | 延续+让出 |

（yielding 数值 =（A 进入 PP 段内的时刻 − 段起点, PP 继续的时长）；仅列前 5 个。）

## 4. text_token=PAD 的状态一致性 + 文本强制可行性（第 4 项）

**源码结论**（`moshi/moshi/models/lm.py`）：
- 文本流**可逐帧 teacher-force**：`prepare_step_input(text_token=...)` 写入 cache 第 0 行并置 `provided=True`；`process_transformer_output` L892 `next_text_token = where(provided, target, sampled)` 强制回显，且 L895-897 显示**被强制的文本 token 会条件化 depformer 的音频生成**；
- **官方已有此组合**：`_step_text_prompt_core`（L1096-1104）就是"逐帧喂文本 token + moshi=静音 + input=sine"——即官方系统提示词相位；
- 官方**未**演练的组合：在强制 moshi 音频（B 语音）的**同时**强制文本——与 interleave 同类，属 API 级延伸。

**最小 probe（probe_text_force.py，已完成、未改正式协议）**：

| 组合 | 回显验证 | 释放后行为 |
|---|---|---|
| T1 官方组合（moshi=静音 + 文本="Well, I mean, you know…"） | ✅ **逐帧与输入完全一致** | 自由生成正常（"Hello?"） |
| T2 回放扩展组合（**moshi=真实 B 音频强制 + 文本=词 token 同步**） | ✅ **逐帧完全一致、零报错** | 自由生成自然续写（"Well, I mean…"） |

**PAD 不一致的具体表现**：历史回放中 B 的文本流恒为 PAD（官方 voice-prompt 约定），导致模型"说了但不记得自己的词"——可见证据即 overlap2 的**无文本音频延续**。**当前协议保持不动**；"文本回放变体"（用 T2 组合回放 B 的真实词）技术上已验证可行，是否纳入矩阵待你决策。

## 5. no-response 标签（第 5 项）

pause2 / filler2 一律标记 **`no-response（未响应）`**，不写作"正确"。其"真人 B 也沉默 26–30s"仅作为语境参考单列。

## 6. 听检材料（第 6 项）

- 听检包（preview/out 音频 + 文本，约 109MB）**不入库**（含数据集音频衍生物，且仓库禁止提交 WAV）；可用 tools 目录脚本从原始数据重新生成。清单与建议听检顺序保留在 `docs/listening_manifest.md`
- 清单与建议听检顺序：`LISTENING_MANIFEST.md`

## 7. 对"是否扩大实验"的输入

协议在**测量层面**现已合格：continuation / initiation / yielding 已分离；声学状态可复算；文本强制可行性已证。仍存在的**协议层面**待决项：(a) 事件边界是否按声学重定义；(b) 文本回放变体是否纳入；(c) 若扩大，建议同步记录 yielding 分布（这是本批新出现的、信息量最大的指标）。
