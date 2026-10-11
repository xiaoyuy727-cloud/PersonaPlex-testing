#!/usr/bin/env python3
"""生成实验 case 索引（含声学修正时间戳），用于从原始数据定位实验事件。

对每个 v2 summary（含 switch_clip_time）：
  - 复算切换点 ±2s 的 A/B 声学活动（阈值 = 各自底噪(20分位)+15dB）
  - 记录 B 是否在切换点说话、其连续说话段
  - 记录 A 的声学起止变化；对 A 侧事件给"声学修正"（真实静音窗/插话点）

用法:
  PP_OUT_DIR=outputs/pp_smoke OY_DATA_DIR=data/open-yap \
  python build_case_index.py --out events/run10_case_index.csv
"""
import argparse, csv, json, glob, os, subprocess, wave
import numpy as np

FS = 1280  # 16k 下 80ms

def cut_rms(conv_path, spk, t0, dur, sr=16000):
    dst = f"/tmp/bci_{spk}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0,t0):.3f}",
                    "-i", f"{conv_path}.{spk}.flac", "-t", f"{dur:.3f}",
                    "-ar", str(sr), "-ac", "1", dst], check=True)
    with wave.open(dst) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768
    n = len(x)//FS
    return np.array([20*np.log10(float(np.sqrt(np.mean(x[i*FS:(i+1)*FS]**2)))+1e-12) for i in range(n)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-dir", default=os.environ.get("PP_OUT_DIR", "outputs/pp_smoke"))
    ap.add_argument("--data-dir",  default=os.environ.get("OY_DATA_DIR", "data/open-yap"))
    ap.add_argument("--out", default="events/run10_case_index.csv")
    a = ap.parse_args()

    rows = []
    for sf in sorted(glob.glob(f"{a.smoke_dir}/case_*/*_summary.json")):
        s = json.load(open(sf))
        if "switch_clip_time" not in s:      # 跳过 v1 旧格式
            continue
        e = s["event"]; conv = e["conv"]; sw = s["switch_clip_time"]
        conv_path = f"{a.data_dir}/extracted/{conv}"
        w0, w1 = max(0, sw - 30), sw + 30
        profA, profB = cut_rms(conv_path, "a", w0, w1 - w0), cut_rms(conv_path, "b", w0, w1 - w0)
        flA, flB = np.percentile(profA, 20), np.percentile(profB, 20)
        actA, actB = profA > flA + 15, profB > flB + 15
        csw = int(round((sw - w0) / 0.08))
        b_at = bool(actB[csw])
        if b_at:
            k = csw
            while k > 0 and actB[k-1]: k -= 1
            j = csw
            while j+1 < len(actB) and actB[j+1]: j += 1
            b_run = f"{w0+k*0.08:.2f}-{w0+j*0.08:.2f}"
        else:
            b_run = ""
        i0, i1 = max(1, csw-25), min(len(actA), csw+25)
        a_trans = ";".join(f"{w0+i*0.08:.2f}{'起' if actA[i] else '止'}"
                           for i in range(i0, i1) if actA[i] != actA[i-1])
        # 声学修正
        note = ""
        if e["etype"] == "overlap":
            first_on = next((w0+i*0.08 for i in range(max(1,csw-8), min(len(actA), csw+19))
                             if actA[i] and not actA[i-1]), None)
            if first_on is not None:
                note = f"A 声学插话点 {first_on:.2f}(转写 {e['t_start']:.2f}, 差 {first_on-e['t_start']:+.2f}s)"
        else:
            # A 侧事件：找 [t_start-0.6, t_end+0.6] 内最长的 A 静音段
            s0, s1 = int((e["t_start"]-w0-0.6)/0.08), int((e["t_end"]-w0+0.6)/0.08)
            best = (0, None)
            i = max(0, s0)
            while i < min(len(actA), s1):
                if not actA[i]:
                    j = i
                    while j+1 < min(len(actA), s1) and not actA[j+1]: j += 1
                    if j - i + 1 > best[0]: best = (j - i + 1, (i, j))
                    i = j + 1
                else: i += 1
            if best[1]:
                i, j = best[1]
                note = (f"A 声学静音 {w0+i*0.08:.2f}-{w0+(j+1)*0.08:.2f}"
                        f" ({(j-i+1)*0.08:.2f}s; 转写间隙 {e['t_start']:.2f}-{e['t_end']:.2f})")
            else:
                note = "A 连续有声（事件窗内无静音帧，无需静音修正）"
        rows.append(dict(case_id=s["case"], conv_id=conv, model_role="B", event_speaker=e["speaker"],
                         etype=e["etype"], sub=e["sub"], t_start_original=e["t_start"],
                         t_end_original=e["t_end"], clip_start=e["clip_start"], clip_end=e["clip_end"],
                         switch_clip_time=sw, switch_offset_s=round(sw - e["t_start"], 3),
                         b_speaking_at_switch=int(b_at), b_run_span_at_switch=b_run,
                         a_acoustic_transitions_pm2s=a_trans, acoustic_correction_note=note))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"写出 {len(rows)} 行 -> {a.out}")

if __name__ == "__main__":
    main()
