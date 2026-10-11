#!/usr/bin/env python3
"""PersonaPlex 流式回放 harness v2 —— 10 事件运行（interleave 主 + official 对照）

对齐保证（v2 新增）：
  - 帧数断言：词活动数组 / 编码帧 / 输出步数三者严格一致
  - probe2：实测 Mimi 解码流的延迟 δ（解码帧 ↔ 内容帧对齐）
  - CSV↔WAV 一致性：逐帧 RMS 与写出的 wav 复核
输入协议（interleave）：
  阶段1 [clip_start, 事件起点)：B 轨→moshi 流(teacher-force)，A 轨→用户流；重叠时同时
  阶段2 [事件起点, clip_end)：**仅停止强制 B**；A 轨继续逐帧喂入用户流，模型自由生成自身流
用法: python smoke_harness.py   （输出 $PP_OUT_DIR，默认 outputs/pp_smoke）
"""
import csv, gc, json, glob, os, re, subprocess, collections
import numpy as np
import torch
import sentencepiece
from huggingface_hub import hf_hub_download
from moshi.models import loaders, LMGen
from moshi.models.lm import (load_audio as lm_load_audio, _iterate_audio,
                             encode_from_sphn, SILENCE_TOKENS, SINE_TOKENS)

DATA = os.environ.get("OY_DATA_DIR", "data/open-yap")   # 数据集根目录（可配置）
OUT  = os.environ.get("PP_OUT_DIR", "outputs/pp_smoke")  # 实验结果输出目录（可配置）
DEV  = "cuda"
FRAME_S = 0.08

def log(*a): print(*a, flush=True)

def ffmpeg_cut(flac, ss, dur, dst, sr=24000):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:.3f}",
                    "-i", flac, "-t", f"{dur:.3f}", "-ar", str(sr), "-ac", "1", dst], check=True)

def ffmpeg_preview(flac_a, flac_b, ss, dur, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:.3f}", "-i", flac_a,
                    "-ss", f"{ss:.3f}", "-i", flac_b, "-t", f"{dur:.3f}",
                    "-filter_complex", "amerge=inputs=2", "-ar", "48000", dst], check=True)

def frame_rms_db(samples, fs=1920):
    n = len(samples) // fs
    return np.array([20*np.log10(float(np.sqrt(np.mean(samples[i*fs:(i+1)*fs]**2)) + 1e-12))
                     for i in range(n)])

def activity_from_words(words, t0, n_frames):
    act = np.zeros(n_frames, dtype=bool)
    for w in words:
        i0 = max(0, int(np.floor((w["start"] - t0) / FRAME_S)))
        i1 = min(n_frames - 1, int(np.ceil((w["end"] - t0) / FRAME_S)))
        if i1 >= i0 and i0 < n_frames:
            act[i0:i1+1] = True
    return act

def select_events(sel):
    """10 个事件：每类 2 个 A-speaker 事件（PP 扮 B）；repetition 取 function 型"""
    spec = [("pause", ""), ("filler", ""), ("repetition", "function"), ("lengthen", ""), ("overlap", "")]
    picked = []
    for etype, sub in spec:
        pool = [r for r in sel if r["etype"] == etype and r["speaker"] == "a" and (not sub or r["sub"] == sub)]
        for k, r in enumerate(pool[:2], 1):
            picked.append((f"{etype}{k}", r))
    return picked

def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(42424242); np.random.seed(42424242)

    sel = list(csv.DictReader(open(f"{DATA}/events/selection_50.csv")))
    for r in sel:
        for k in ("t_start", "t_end", "clip_start", "clip_end"):
            r[k] = float(r[k])
    events = select_events(sel)
    assert len(events) == 10, f"期望 10 个事件, 实得 {len(events)}"
    log("[1/6] 10 个事件（每类 2 个, A 说话人）:")
    for name, r in events:
        log(f"   {name:12s} {r['etype']:10s} conv={r['conv'][:14]} t=[{r['t_start']:.2f},{r['t_end']:.2f}] "
            f"clip=[{r['clip_start']:.1f},{r['clip_end']:.1f}] {r['context'][:60]}")

    clipmeta = {}
    for name, r in events:
        d = f"{OUT}/case_{name}"; os.makedirs(d, exist_ok=True)
        dur = r["clip_end"] - r["clip_start"]
        conv = f"{DATA}/extracted/{r['conv']}"
        ffmpeg_cut(f"{conv}.a.flac", r["clip_start"], dur, f"{d}/clip_a.wav")
        ffmpeg_cut(f"{conv}.b.flac", r["clip_start"], dur, f"{d}/clip_b.wav")
        ffmpeg_preview(f"{conv}.a.flac", f"{conv}.b.flac", r["clip_start"], dur, f"{d}/preview.wav")
        clipmeta[name] = dict(row=r, dir=d, dur=dur, conv_path=conv)
    log("[2/6] 音频裁剪完成")

    # ---------- 模型 ----------
    log("[3/6] 加载模型 …")
    mimi_w  = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    moshi_w = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    tok_w   = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
    tokenizer = sentencepiece.SentencePieceProcessor(tok_w)
    mimi   = loaders.get_mimi(mimi_w, DEV)
    mimi_b = loaders.get_mimi(mimi_w, DEV)
    omimi  = loaders.get_mimi(mimi_w, DEV)
    lm     = loaders.get_moshi_lm(moshi_w, device=DEV)
    lm.eval()
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(lm, audio_silence_frame_cnt=int(0.5*mimi.frame_rate), sample_rate=mimi.sample_rate,
                   device=DEV, frame_rate=mimi.frame_rate, use_sampling=True,
                   temp=0.8, temp_text=0.7, top_k=250, top_k_text=25)
    for m in (mimi, mimi_b, omimi): m.streaming_forever(1)
    lm_gen.streaming_forever(1)
    for _ in range(4):
        z = torch.zeros(1, 1, frame_size, device=DEV)
        c = mimi.encode(z); mimi_b.encode(z); omimi.encode(z)
        for k in range(c.shape[-1]):
            t = lm_gen.step(c[:, :, k:k+1])
            if t is not None: omimi.decode(t[:, 1:9])
    torch.cuda.synchronize()
    log("     预热完成")

    def reset_all():
        mimi.reset_streaming(); mimi_b.reset_streaming(); omimi.reset_streaming()
        lm_gen.reset_streaming()

    zero_f = lm_gen._encode_zero_frame(); sine_f = lm_gen._encode_sine_frame()
    sil_np, sine_np = SILENCE_TOKENS.astype(np.int64), SINE_TOKENS.astype(np.int64)
    probe_lines = [f"mimi: sr={mimi.sample_rate} frame_rate={mimi.frame_rate} frame_size={frame_size} "
                   f"({frame_size/mimi.sample_rate*1000:.1f} ms/帧)",
                   f"lm: delays={lm.delays} max_delay={lm_gen.max_delay}"]

    # ================= PROBE 1: teacher-forcing 语义 =================
    reset_all()
    b_samp = lm_load_audio(f"{OUT}/case_pause1/clip_b.wav", mimi.sample_rate)
    b_codes = torch.cat([x for x in encode_from_sphn(mimi_b, _iterate_audio(b_samp, frame_size, pad=True), 1)], dim=-1)
    seg = b_codes[:, :, 80:100]
    outs = []
    for i in range(140):
        m_tok = sine_f if 50 <= i < 60 else (seg[:, :, i-60:i-60+1] if 60 <= i < 80 else zero_f)
        o = lm_gen.step(input_tokens=sine_f, moshi_tokens=m_tok, text_token=3)
        if o is not None:
            outs.append((i, o[0, 1:9, 0].cpu().numpy().copy(), int(o[0, 0, 0].item())))
    lag_obs = next((s - 50 for (s, row, _) in outs if np.array_equal(row, sine_np)), None)
    checks = [f"feed帧{60+k} 真实B行: " +
              ("匹配" if (r := next((row for (s, row, _) in outs if s == 60+k+lm_gen.max_delay), None)) is not None
               and np.array_equal(r, seg[0, :, k].cpu().numpy()) else "不匹配") for k in (0, 5, 10, 19)]
    probe_lines += [f"[PROBE1] SINE 强制后出现在输出: 延迟={lag_obs} 帧 (max_delay={lm_gen.max_delay})",
                    f"[PROBE1] {checks}", f"[PROBE1] 强制期文本流输出={set(t for _, _, t in outs)}"]

    # ================= PROBE 2: 解码器流式延迟 δ =================
    reset_all()
    dec_rows = []
    for i in range(80):
        m_tok = sine_f if 20 <= i < 40 else zero_f
        o = lm_gen.step(input_tokens=sine_f, moshi_tokens=m_tok, text_token=3)
        if o is not None:
            pcm = omimi.decode(o[:, 1:9]).detach().cpu().numpy()[0, 0]
            rms = 20*np.log10(float(np.sqrt(np.mean(pcm**2)) + 1e-12))
            dec_rows.append((i, i - lm_gen.max_delay, rms))   # (循环步, 内容帧, 解码RMS)
    thr = -40.0
    tone_frames = [cf for (_, cf, r) in dec_rows if r > thr]
    delta_start = (min(tone_frames) - 20) if tone_frames else None
    delta_end   = (40 - (max(tone_frames) + 1)) if tone_frames else None
    probe_lines += [f"[PROBE2] SINE 内容帧 20..39 → 解码音调出现于内容帧 {min(tone_frames) if tone_frames else '?'}"
                    f"..{max(tone_frames) if tone_frames else '?'}  ⇒ 解码延迟 δ≈{delta_start} 帧",
                    f"[PROBE2] 原始 (步, 内容帧, RMS) 抽样: " +
                    ", ".join(f"({s},{c},{r:.0f})" for s, c, r in dec_rows[15:28:3])]
    open(f"{OUT}/probe_forcing.txt", "w").write("\n".join(probe_lines))
    log("[4/6] PROBE 完成：\n   " + "\n   ".join(probe_lines[-4:]))

    # ================= 回放 =================
    def run_case(tag, r, clip, mode):
        d = clip["dir"]
        a_samp = lm_load_audio(f"{d}/clip_a.wav", mimi.sample_rate)
        b_samp = lm_load_audio(f"{d}/clip_b.wav", mimi.sample_rate)
        n_frames = int(np.ceil(a_samp.shape[-1] / frame_size))
        t0 = r["clip_start"]
        js = json.load(open(f"{clip['conv_path']}.json"))
        words_a, words_b = js["transcripts"]["a"]["words"], js["transcripts"]["b"]["words"]
        act_a, act_b = activity_from_words(words_a, t0, n_frames), activity_from_words(words_b, t0, n_frames)
        P1 = int(np.floor((r["t_start"] - t0) / FRAME_S))
        reset_all()
        a_codes = torch.cat([x for x in encode_from_sphn(mimi,   _iterate_audio(a_samp, frame_size, pad=True), 1)], dim=-1)
        b_codes = torch.cat([x for x in encode_from_sphn(mimi_b, _iterate_audio(b_samp, frame_size, pad=True), 1)], dim=-1)
        # ---- 对齐断言 ----
        assert a_codes.shape[-1] == b_codes.shape[-1] == n_frames, \
            f"帧数不一致: a={a_codes.shape[-1]} b={b_codes.shape[-1]} frames={n_frames}"

        rows_csv, out_pcm, text_pieces, onset_tok = [], [], [], None
        for i in range(n_frames):
            if i < P1:   # 阶段1：历史回放（A 全程在用户流；B 强制在自己的流）
                have_a, have_b = act_a[i], act_b[i]
                if mode == "official":
                    m_tok, i_tok = (b_codes[:, :, i:i+1] if have_b else zero_f), sine_f
                else:
                    if have_a and have_b: m_tok, i_tok = b_codes[:, :, i:i+1], a_codes[:, :, i:i+1]
                    elif have_b:          m_tok, i_tok = b_codes[:, :, i:i+1], sine_f
                    elif have_a:          m_tok, i_tok = zero_f, a_codes[:, :, i:i+1]
                    else:                 m_tok, i_tok = zero_f, sine_f
                lm_gen.step(input_tokens=i_tok, moshi_tokens=m_tok, text_token=3)
                continue
            # 阶段2：仅停止强制 B；A 的真实音频继续逐帧喂入
            o = lm_gen.step(input_tokens=a_codes[:, :, i:i+1])
            if o is None: continue
            cont_frame = i - lm_gen.max_delay
            if cont_frame < P1: continue
            cont_time = t0 + cont_frame * FRAME_S
            row = o[0, 1:9, 0].cpu().numpy()
            is_sil = bool(np.array_equal(row, sil_np))
            pcm = omimi.decode(o[:, 1:9]).detach().cpu().numpy()[0, 0]
            out_pcm.append(pcm)
            tid = int(o[0, 0, 0].item())
            piece = {0: "EPAD", 1: "BOS", 2: "EOS", 3: "PAD"}.get(tid)
            if piece is None: piece = tokenizer.id_to_piece(tid).replace("▁", " ")
            text_pieces.append(piece)
            rms = 20*np.log10(float(np.sqrt(np.mean(pcm**2)) + 1e-12))
            rows_csv.append(dict(step=i, cont_frame=cont_frame, clip_t=round(cont_time, 2),
                                 a_act=int(act_a[i]), b_act=int(act_b[i]), text=piece,
                                 silence=int(is_sil), rms=round(rms, 1),
                                 tok=",".join(str(x) for x in row)))
            if onset_tok is None and not is_sil: onset_tok = cont_time
        # ---- 写出 ----
        import sphn
        if out_pcm:
            sphn.write_wav(f"{d}/{tag}_out.wav", np.concatenate(out_pcm), mimi.sample_rate)
        with open(f"{d}/{tag}_frames.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step","cont_frame","clip_t","a_act","b_act","text","silence","rms","tok"])
            w.writeheader(); w.writerows(rows_csv)
        # 可读文本
        words_only = [p for p in text_pieces if p not in ("PAD", "EPAD", "BOS", "EOS")]
        readable = re.sub(r"\s+", " ", "".join(words_only)).strip()
        with open(f"{d}/{tag}_text.txt", "w") as f:
            f.write(readable + "\n\n[raw pieces]\n" + "|".join(text_pieces))
        # ---- onset（RMS 判据；token 判据仅记录）----
        onset_rms, floor, thr = None, None, None
        if rows_csv:
            arr = np.array([x["rms"] for x in rows_csv])
            floor = float(np.percentile(arr, 20)); thr = floor + 20.0
            for j in range(1, len(arr)-2):
                if arr[j] > thr and arr[j+1] > thr and arr[j+2] > thr:
                    onset_rms = float(rows_csv[j]["clip_t"]); break
        # ---- CSV↔WAV 一致性 ----
        wav_diff = None
        if out_pcm:
            import soundfile as sf
            d2, sr2 = sf.read(f"{d}/{tag}_out.wav")
            wr = frame_rms_db(d2)
            k = min(len(wr), len(rows_csv))
            wav_diff = float(np.max(np.abs(wr[:k] - np.array([x["rms"] for x in rows_csv])[:k])))
        # ---- 活动占比 ----
        frac = lambda lo, hi: (float(np.mean((np.array([x["rms"] for x in rows_csv]) > thr)
                                             & (np.array([x["clip_t"] for x in rows_csv]) >= lo)
                                             & (np.array([x["clip_t"] for x in rows_csv]) < hi))) if rows_csv else None)
        b_ref = next((w0["start"] for w0 in words_b if w0["start"] >= r["t_start"]), None)
        summary = dict(case=tag, mode=mode, event=dict(r), n_frames=n_frames, phase1_frames=P1,
                       switch_clip_time=round(t0 + P1*FRAME_S, 2), max_delay=lm_gen.max_delay,
                       pp_onset_rms_clip_time=onset_rms,
                       pp_onset_rms_delay_s=(round(onset_rms - r["t_start"], 2) if onset_rms else None),
                       pp_onset_token_clip_time=onset_tok,
                       rms_floor_db=floor, rms_threshold_db=thr, csv_wav_rms_maxdiff_db=wav_diff,
                       pp_active_frac_after_event=frac(r["t_end"], r["clip_end"]),
                       human_b_onset_abs=b_ref,
                       human_b_onset_delay_s=(round(b_ref - r["t_start"], 2) if b_ref else None),
                       output_frames=len(out_pcm), text=readable)
        json.dump(summary, open(f"{d}/{tag}_summary.json", "w"), ensure_ascii=False, indent=1, default=str)
        log(f"   [{tag:18s}] onset(RMS)={onset_rms} (事件起点 {r['t_start']:.2f})  真人B参考={summary['human_b_onset_delay_s']}s  "
            f"帧={len(out_pcm)} wav校差={wav_diff}dB 文本={readable[:80]}")
        del a_codes, b_codes, a_samp, b_samp, out_pcm
        gc.collect(); torch.cuda.empty_cache()
        return summary

    log("\n[5/6] === 回放（10 interleave + 2 official 对照）===")
    results = []
    official_controls = {"pause1", "overlap1"}
    for name, r in events:
        results.append(run_case(f"{name}_interleave", r, clipmeta[name], "interleave"))
        if name in official_controls:
            results.append(run_case(f"{name}_official", r, clipmeta[name], "official"))
    json.dump(results, open(f"{OUT}/run10_results.json", "w"), ensure_ascii=False, indent=1, default=str)

    # ---------- RUN_INDEX（供人工听检）----------
    idx = ["# 人工听检索引（onset 为 RMS 判据；真人 B 仅为行为参考）", "",
           "| case | 事件 | 窗口 | onset(RMS) | 真人B参考 | 音频 | 文本 |",
           "|---|---|---|---|---|---|---|"]
    for s in results:
        e = s["event"]; d = s["case"].rsplit("_", 1)[0]
        idx.append(f"| {s['case']} | {e['etype']}/{e['sub'] or '-'} @ {e['conv'][:14]} | "
                   f"[{e['clip_start']:.0f}, {e['clip_end']:.0f}] | {s['pp_onset_rms_clip_time']} "
                   f"({s['pp_onset_rms_delay_s']}s) | {s['human_b_onset_delay_s']}s | "
                   f"preview: case_{d}/preview.wav, out: case_{d}/{s['case']}_out.wav | case_{d}/{s['case']}_text.txt |")
    open(f"{OUT}/RUN_INDEX.md", "w").write("\n".join(idx))
    log("[6/6] 完成。索引: RUN_INDEX.md")
    for s in results:
        log(f"  {s['case']:18s} onset_delay={s['pp_onset_rms_delay_s']}s  human_ref={s['human_b_onset_delay_s']}s  "
            f"text={s['text'][:60]}")

if __name__ == "__main__":
    # 官方 offline.py / server.py 均在 torch.no_grad() 下运行；缺失会致 mimi 编码每帧 4.2MB 梯度图泄漏
    with torch.no_grad():
        main()
