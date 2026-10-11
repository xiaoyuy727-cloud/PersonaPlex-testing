# PersonaPlex 10 事件运行报告（harness v2）

> 2026-10-11。10 事件（每类 2 个：pause/filler/repetition/lengthen/overlap，均 A 说话人）+ 2 个 official 对照（pause1、overlap1）。
> harness：`smoke_harness.py`（v2）；产物：`results/run10/`；听检索引：`results/run10/RUN_INDEX.md`

---

## 1. 输入协议与对齐验证（你关心的第 1-3 点）

**A 的真实音频在自由窗口全程逐帧输入**——`run_case()` 阶段二：

```python
# 阶段2：仅停止强制 B；A 的真实音频继续逐帧喂入（直到 clip_end，含事件后 15s）
o = lm_gen.step(input_tokens=a_codes[:, :, i:i+1])
```

- 阶段二的输入是 `[P1, n_frames)` 的**全部** A 编码帧；只有 `moshi_tokens` 不再传入（B 强制停止）。每个 case 的 `*_frames.csv` 中 `a_act` 列可复盘整段 A 的时间结构。
- interleave 模式阶段一 = A→用户流 + B→moshi 流（重叠时同时）；official 模式阶段一用户流为 sine（仅 B 历史）。

**对齐校验（三重，全部通过）**：

| 校验 | 方法 | 结果 |
|---|---|---|
| 帧数一致 | 断言 `A编码帧 == B编码帧 == n_frames`（词活动数组同长） | 12/12 通过（752–766 帧） |
| 输出↔输入对齐 | probe1：强制 token 出现在输出的步数差 | **恰 1 帧 = max_delay**（80ms，已在校正中扣除） |
| 解码器延迟 δ | probe2：SINE 内容帧 20–39 → 解码音调出现于内容帧 20–40 | **δ=0 帧**（解码帧与内容帧严格同帧） |
| 日志↔音频一致 | 逐帧 RMS vs 写出 wav 复算 | 最大差 0.58–0.74 dB（仅 wav 写路径量化） |

**切换点**：`P1 = floor((t_start − clip_start)/0.08)`，量化到 80ms 网格（相对事件起点最多早 80ms，已记录在 summary 的 `switch_clip_time`）。**综上：系统性的绝对时间不确定度 ≤80ms；其余环节零偏差。**

## 2. 10 事件结果（interleave 主实验 + official 对照）

| case | 事件 | **PP onset 延迟** | 真人 B 参考 | PP 事件后活跃占比 | PP 输出（首段） |
|---|---|---|---|---|---|
| pause1 | A 停顿 1.2s | +6.04s | +4.70s | 0.47 | "No, no, no, it's fine. It's fine. Just, yeah…" |
| pause1_official | 同上（对照） | +4.68s | +4.70s | 0.15 | "Yeah, yeah, it's it's a bit annoying" |
| pause2 | A 停顿 1.2s | **全程未开口** | +29.62s | 0.00 | （无） |
| filler1 | "uh" | +3.08s | +3.61s | 0.62 | "Oh, yeah, definitely, the helicopters, the drones…" |
| filler2 | "uh" | **全程未开口** | +26.36s | 0.00 | （无） |
| repetition1 | "I I" | +8.92s | +1.96s | 0.40 | "Oh, no way. My mom's in her early 70s…" |
| repetition2 | "I I" | +3.96s | +8.36s | 0.43 | "Oh, there you go. That's great. That's great…" |
| lengthen1 | 延长词 0.96s | +5.48s | +4.93s | 0.52 | "Yeah, I understand, but the main thing…" |
| lengthen2 | 延长词 0.88s | +7.96s | +6.51s | 0.38 | "Oh, man, I' gonna go find this movie…" |
| overlap1 | A 插话 B | +0.04s（续说） | +0.10s | 0.56（8 个说话段） | "Yeah, I know, I know, I'm not that into…" |
| overlap1_official | 同上（对照） | +0.04s | +0.10s | 0.47 | "Yeah, they make cool stuff. Yeah…" |
| overlap2 | A 插话 B | +0.04s（续说 ~1.0s 后让出） | +0.14s | 0.04 | （文本流 PAD——见 §3.3） |

## 3. 观察（全部只作为"行为样本"陈述，不作对错判断）

1. **停顿/填充词/重复/延长四类**：onset 分布在 **+3.0 ~ +8.9s**，且同一类型内部差异大于类型之间差异（如 repetition1 慢 7s、repetition2 快 4.4s）→ **单事件对比无意义，必须看分布**（这正是收紧到 50×3 的理由）。
2. **"全程未开口"两例（pause2/filler2）**：输出最大 RMS ≈ -59dB（确为静音）、文本为空；**真人 B 在对应处也沉默了 26–30s**（下一词分别在 +29.6/+26.4s）——PP 的沉默与真实对话状态一致。这类样本要保留，是"正确不抢话"的对照。
3. **overlap 两类行为都出现了**：overlap1 持续介入（事件窗内 8 个说话段、活跃占比 0.56），overlap2 续说约 1s 后让出（活跃 0.04）。**overlap2 的文本流全程 PAD 但音频有续说**——原因：回放协议中历史 B 的文本流被强制为 PAD（官方 voice-prompt 约定），模型被释放时处于"词说到一半"，出现了无文本的音频续说。这是协议的已知产物，正式实验可加一个"文本回放"变体（把 B 的真实词 token 也喂进文本流）作对照——**待你决策**。
4. **协议敏感性实锤**：pause1 的 interleave vs official —— onset 6.04 vs 4.68s、活跃占比 0.47 vs 0.15。比第一次冒烟（Δ0.16s）大得多，说明 official 对照不能省。
5. 真人 B 参考始终单列（本批 PP 与真人差值 -7.0 ~ +4.4s 不等）。

## 4. 听检材料（你问的第 5 点）

12 个 case × 3 个文件（未入库，见 README「听检材料」一节的结构说明）：

```
case_{pause1,pause2,filler1,filler2,repetition1,repetition2,lengthen1,lengthen2,overlap1,overlap2}/
    preview.wav                        # 裁剪窗立体声（左=A 右=B），原始对话
    {case}_interleave_out.wav          # PP 输出音频（24kHz）
    {case}_interleave_text.txt         # PP 输出文本（可读版 + raw token 序列）
    {case}_interleave_frames.csv       # 完整逐帧时间序列（时间/A,B活动/文本/token/RMS）
case_pause1/pause1_official_*      case_overlap1/overlap1_official_*   # 两个对照
```
另：旧冒烟 3 个 case 的文本已补齐（`case_pause/*_text.txt`、`case_filler/filler_interleave_text.txt`）。
汇总索引 `RUN_INDEX.md`（每行的 onset/真人参考/文件路径）。

## 5. 状态与下一步

- 显存全程稳定（~19G 峰值），12 case 总耗时约 25 分钟，可安全扩量。
- 待你：① 听检；② 拍板"文本回放变体"是否纳入矩阵；③ 确认后扩 50×3（或先 20×2 看分布）。
