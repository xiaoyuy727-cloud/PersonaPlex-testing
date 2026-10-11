#!/usr/bin/env python3
"""协议有效性审计：切换点声学活动 + turn-taking 指标分离 + no-response 标签

用法: python audit_protocol.py
输出: $PP_OUT_DIR/PROTOCOL_AUDIT.txt
"""
import csv, json, glob, os, subprocess
import numpy as np

SMOKE = os.environ.get("PP_OUT_DIR", "outputs/pp_smoke")
DATA = os.path.join(os.environ.get("OY_DATA_DIR", "data/open-yap"), "extracted")
FS, FRAME_S = 1280, 0.08   # 16k 下 80ms

def cut_track(conv, spk, t0, dur, sr=16000):
    dst = f"/tmp/aud_{spk}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.3f}",
                    "-i", f"{DATA}/{conv}.{spk}.flac", "-t", f"{dur:.3f}",
                    "-ar", str(sr), "-ac", "1", dst], check=True)
    import wave
    with wave.open(dst) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768
    n = len(x)//FS
    return np.array([20*np.log10(float(np.sqrt(np.mean(x[i*FS:(i+1)*FS]**2)))+1e-12) for i in range(n)])

def active(prof, floor):
    return prof > floor + 15.0   # 声学活动：高于底噪 15dB

def segs_from_bool(b, merge=2, min_len=3):
    segs, i = [], 0
    while i < len(b):
        if b[i]:
            j = i
            while j+1 < len(b) and (b[j+1] or (j+2 < len(b) and b[j+2])): j += 1
            if j - i + 1 >= min_len: segs.append((i, j))
            i = j + 1
        else: i += 1
    return segs

def main():
    out = []
    for sf in sorted(glob.glob(f"{SMOKE}/case_*/*_summary.json")):
        s = json.load(open(sf))
        if "switch_clip_time" not in s: continue
        e = s["event"]; case, conv = s["case"], e["conv"]
        t0, t1 = e["clip_start"], e["clip_end"]
        sw = s["switch_clip_time"]
        # ---- 裁剪整个 clip 的声学包络 ----
        profA, profB = cut_track(conv, "a", t0, t1-t0), cut_track(conv, "b", t0, t1-t0)
        flA, flB = np.percentile(profA, 20), np.percentile(profB, 20)
        actA, actB = active(profA, flA), active(profB, flB)
        # ---- ±2s 窗口状态串 ----
        c0 = int(round((sw - t0)/FRAME_S))
        i0, i1 = c0 - 25, c0 + 25
        state = "".join(("A" if actA[i] else ".") + ("B" if actB[i] else ".") for i in range(i0, i1))
        B_at_switch, A_at_switch = bool(actB[c0]), bool(actA[c0])
        # B 在切换点的连续说话段
        k = c0
        while k > 0 and actB[k-1]: k -= 1
        j = c0
        while j+1 < len(actB) and actB[j+1]: j += 1
        B_run = (t0 + k*FRAME_S, t0 + j*FRAME_S) if B_at_switch else None
        # A 在窗口内的声学开始/结束
        A_trans = [f"{t0+i*FRAME_S:.2f}s({'起' if actA[i] and not actA[i-1] else '止'})"
                   for i in range(max(1,i0), min(len(actA), i1)) if actA[i] != actA[i-1]]
        # ---- PP 输出段（RMS 判据 + 段化）----
        rows = list(csv.DictReader(open(f"{SMOKE}/case_{case.rsplit('_',1)[0]}/{case}_frames.csv")))
        rms = np.array([float(r["rms"]) for r in rows]); t = np.array([float(r["clip_t"]) for r in rows])
        thr = np.percentile(rms, 20) + 20.0
        pp_segs = [(t[i], t[jj]) for (i, jj) in segs_from_bool(rms > thr)]
        # 分类：continuation（切换后 ≤0.24s 起始 且 B 当时在说）；initiation（PP 静音 ≥0.5s 后的首个新段）
        cont, init, yields = None, None, []
        prev_end = sw
        for (a, b) in pp_segs:
            if a <= sw + 0.24 and B_at_switch and cont is None:
                cont = (a, b); prev_end = b; continue
            if a - prev_end >= 0.5 and init is None and a > sw + 0.24:
                init = (a, b); prev_end = b; continue
            prev_end = b
        # yield：A 声学活动开始于某个 PP 段内 → PP 段结束时刻 - A 开始 = 让出延迟
        for (a, b) in pp_segs:
            ia, ib = int((a-t0)/FRAME_S), int((b-t0)/FRAME_S)
            for i in range(max(1, ia), min(len(actA), ib)):
                if actA[i] and not actA[i-1]:
                    yields.append((round(t0+i*FRAME_S - a, 2), round(b - (t0+i*FRAME_S), 2)))
        out += [f"\n{'='*78}", f"[{case}] {e['etype']}/{e['sub'] or '-'} @ {conv[:14]}"]
        out += [f"  切换点={sw:.2f}s（事件起点 {e['t_start']:.2f}s，量化差 {sw-e['t_start']:+.2f}s）"]
        out += [f"  ±2s 声学状态串（80ms/格, 左A右B, '.'=静音）:"]
        out += [f"    -2.0s {state} +2.0s"]
        out += [f"  切换点: A={'说' if A_at_switch else '静'} B={'说' if B_at_switch else '静'}"
                + (f"；B 连续说话段 {B_run[0]:.2f}–{B_run[1]:.2f}s (跨切换点)" if B_run else "")]
        out += [f"  窗口内 A 声学变化: {', '.join(A_trans[:6]) if A_trans else '无'}"]
        kind = "continuation(延续被强制的B语音)" if cont else ("no-response" if init is None else "initiation(新发起)")
        out += [f"  PP: {kind}"
                + (f"；延续段 {cont[0]:.2f}–{cont[1]:.2f}s（时长 {cont[1]-cont[0]:.2f}s）" if cont else "")
                + (f"；发起段 {init[0]:.2f}–{init[1]:.2f}s（延迟 +{init[0]-e['t_start']:.2f}s）" if init else "")]
        if yields:
            out += [f"  让出（A 在 PP 说话时进入 → PP 停止）: 延迟/让出时长的列表 {yields[:5]}"]
        out += [f"  PP 说话段共 {len(pp_segs)} 个: {[(round(a,1), round(b,1)) for a,b in pp_segs][:6]}"]
    txt = "\n".join(out) + "\n"
    open(f"{SMOKE}/PROTOCOL_AUDIT.txt", "w").write(txt)
    print(txt)

if __name__ == "__main__":
    main()
