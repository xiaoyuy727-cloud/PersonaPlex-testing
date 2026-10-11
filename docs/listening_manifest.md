# 听检材料清单（未入库；tarball 需自行打包）

材料位于运行输出目录（`$PP_OUT_DIR`，本实验为 `pp_smoke/`）。每个 case 目录包含 4 类文件：

| 文件 | 内容 | 听检要点 |
|---|---|---|
| `preview.wav` | 原始对话裁剪窗（48kHz 立体声，**左=A 右=B**） | 先听它建立原始语境 |
| `{case}_interleave_out.wav` | **PP 输出音频**（24kHz 单声道，从事件起点开始） | 对比 PP 何时开口、与 A 的重叠 |
| `{case}_interleave_text.txt` | PP 输出的文本（可读版 + raw token 序列） | 对照内容合理性 |
| `{case}_interleave_frames.csv` | 逐帧时间序列（clip 时间/A、B 活动/文本/token/RMS） | 精查任何一帧 |

官方对照：`case_pause1/pause1_official_*`、`case_overlap1/overlap1_official_*`。
旧冒烟 case（v1）：`case_pause/`、`case_filler/`（注意其 onset 数值为已弃用的 token 判据，仅听检用）。

## 建议听检顺序

1. **overlap1 / overlap2**：切换点 B（真人）正在说话 → PP 的 +0.04s "onset" 实为**延续**（0.40s / 0.96s），随后 A 插话 → PP **让出**（延迟 0.16s / 0.88s）。听：延续是否像"说完这句话"、让出是否干净。
2. **pause1（两模式对比）**：interleave 输出 "No, no, no, it's fine…"（onset +6.04s、活跃 0.47）vs official "Yeah, yeah, it's it's a bit annoying"（+4.68s、活跃 0.15）——同一事件、不同回放协议，听差异。
3. **repetition1 / repetition2**：同一类型、相反行为（PP 比真人慢 7.0s / 快 4.4s）。
4. **pause2 / filler2（no-response）**：PP 全程静音；对照 preview 听当时对话状态（真人 B 也沉默了 26–30s）。**标签为"未响应"，不作对错判断**。
5. **filler1 / lengthen1 / lengthen2**：常规模态。

汇总索引：`RUN_INDEX.md`；协议审计（含 ±2s 声学状态串）：`PROTOCOL_AUDIT.txt`。
