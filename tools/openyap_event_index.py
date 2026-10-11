#!/usr/bin/env python3
"""Open Yap 1K 样本 → 犹豫/不流畅事件索引

从每段对话的词级转写（{word, start, end}，两位说话人共用时间轴）中抽取：
  - filler     填充词 (uh/um/ah/...)
  - repetition 相邻同词重复（区分 function/content 词类）
  - pause      说话人词间停顿 >=0.5s（对方不在说 → 犹豫停顿）
  - turn_gap   词间停顿期间对方接过话头（轮次切换）
  - overlap    跨说话人并发语音 >=0.3s（含谁先起头=插话方向）
  - lengthen   词时长离群（>mean+2.5σ 且 >=0.7s）作为"延长"代理

用法: python openyap_event_index.py <extracted_dir> <out_dir>
输出: events_all.csv / selection_50.csv / summary.txt
"""
import json, sys, csv, glob, os, collections, statistics

FILLERS   = {"uh","um","ah","eh","hm","hmm","mmm","uhh","umm","er","erm"}
FUNC_WORDS = {"i","i'm","you","we","the","a","an","to","and","but","so","it","that",
              "is","was","my","your","he","she","they","of","in","on","for","with",
              "do","did","does","can","could","would","will","just","like","get","got"}
PAUSE_MIN   = 0.5
REP_MAX_GAP = 0.6
OVERLAP_MIN = 0.3
SPEAKER_MERGE_GAP = 0.15   # 词间隔 <=0.15s 视为连续说话段


def norm(w: str) -> str:
    return w.strip(".,?!;:'\"").lower()


def ctx(ws, i, k=3):
    lo, hi = max(0, i - k), min(len(ws), i + k + 1)
    return " ".join(x["word"] for x in ws[lo:hi])


def speaking_runs(ws):
    """把词序列合并成连续说话段 [(start,end), ...]"""
    runs = []
    for w in ws:
        if runs and w["start"] - runs[-1][1] <= SPEAKER_MERGE_GAP:
            runs[-1][1] = max(runs[-1][1], w["end"])
        else:
            runs.append([w["start"], w["end"]])
    return [(s, e) for s, e in runs]


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rows, summary = [], collections.Counter()

    for fp in sorted(glob.glob(os.path.join(src, "*.json"))):
        d = json.load(open(fp))
        cid, dur = d["conversation_id"], d["duration_seconds"]
        words = {s: d["transcripts"][s]["words"] for s in ("a", "b")}

        for s in ("a", "b"):
            ws = words[s]
            other = words["b" if s == "a" else "a"]
            # ---- filler & repetition ----
            for i, w in enumerate(ws):
                n = norm(w["word"])
                if n in FILLERS:
                    rows.append(dict(conv=cid, speaker=s, etype="filler", sub="",
                                     t_start=round(w["start"],2), t_end=round(w["end"],2),
                                     dur=round(w["end"]-w["start"],2), text=w["word"], context=ctx(ws, i)))
                    summary["filler"] += 1
                if i > 0:
                    p = ws[i-1]; pn = norm(p["word"])
                    if n and n == pn and 0 <= w["start"] - p["end"] <= REP_MAX_GAP:
                        sub = "function" if n in FUNC_WORDS else "content"
                        rows.append(dict(conv=cid, speaker=s, etype="repetition", sub=sub,
                                         t_start=round(p["start"],2), t_end=round(w["end"],2),
                                         dur=round(w["end"]-p["start"],2),
                                         text=f"{p['word']} {w['word']}",
                                         context=f"gap={w['start']-p['end']:.2f}s | " + ctx(ws, i-1)))
                        summary[f"repetition_{sub}"] += 1
            # ---- pause / turn_gap ----
            for i in range(1, len(ws)):
                g0, g1 = ws[i-1]["end"], ws[i]["start"]
                gap = g1 - g0
                if gap >= PAUSE_MIN:
                    crossed = any(o["start"] < g1 and o["end"] > g0 for o in other)
                    etype = "turn_gap" if crossed else "pause"
                    rows.append(dict(conv=cid, speaker=s, etype=etype, sub="",
                                     t_start=round(g0,2), t_end=round(g1,2), dur=round(gap,2),
                                     text=f"[{gap:.2f}s]",
                                     context=ctx(ws, i-1, 2) + "  ⏸  " + ctx(ws, i, 2)))
                    summary[etype] += 1
            # ---- lengthening proxy ----
            cand = [w for w in ws if len(norm(w["word"])) >= 3 and norm(w["word"]) not in FILLERS]
            if len(cand) > 30:
                durs = [w["end"] - w["start"] for w in cand]
                mu, sd = statistics.mean(durs), statistics.pstdev(durs)
                thr = max(mu + 2.5 * sd, 0.7)
                for w in cand:
                    wd = w["end"] - w["start"]
                    if wd >= thr:
                        rows.append(dict(conv=cid, speaker=s, etype="lengthen", sub="",
                                         t_start=round(w["start"],2), t_end=round(w["end"],2),
                                         dur=round(wd,2), text=w["word"],
                                         context=f"thr={thr:.2f}s | " + ctx(ws, ws.index(w))))
                        summary["lengthen"] += 1

        # ---- overlap (跨说话人) ----
        ra, rb = speaking_runs(words["a"]), speaking_runs(words["b"])
        for (s1, e1) in ra:
            for (s2, e2) in rb:
                ov = min(e1, e2) - max(s1, s2)
                if ov >= OVERLAP_MIN:
                    st = max(s1, s2)
                    who = "a" if s1 > s2 else "b"      # 后起者 = 插话方
                    rows.append(dict(conv=cid, speaker=who, etype="overlap", sub=f"by_{who}",
                                     t_start=round(st,2), t_end=round(st+ov,2), dur=round(ov,2),
                                     text=f"overlap {ov:.2f}s", context=""))
                    summary["overlap"] += 1

    # ---- 写全量事件表 ----
    cols = ["conv","speaker","etype","sub","t_start","t_end","dur","text","context"]
    with open(os.path.join(out, "events_all.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    # ---- 选 50 个代表事件（配额 + 跨会话分散）----
    QUOTA = {"pause": 12, "repetition_function": 12, "filler": 10, "overlap": 8, "lengthen": 8}
    picks, per_conv = [], collections.Counter()
    for key, n in QUOTA.items():
        if key == "repetition_function":
            pool = [r for r in rows if r["etype"]=="repetition" and r["sub"]=="function"]
        else:
            pool = [r for r in rows if r["etype"]==key]
        # 优先分数：pause 偏好 0.8-2.0s；overlap 偏好 0.3-2s；其余按时间
        def score(r):
            if key=="pause":    return abs(r["dur"]-1.2)
            if key=="overlap":  return abs(r["dur"]-0.8)
            return r["dur"]
        for r in sorted(pool, key=score):
            if n == 0: break
            if per_conv[r["conv"]] >= 4: continue
            picks.append(r); per_conv[r["conv"]] += 1; n -= 1
    picks.sort(key=lambda r: (r["conv"], r["t_start"]))
    dur_by_conv = {}
    for fp in glob.glob(os.path.join(src, "*.json")):
        d = json.load(open(fp)); dur_by_conv[d["conversation_id"]] = d["duration_seconds"]
    with open(os.path.join(out, "selection_50.csv"), "w", newline="") as f:
        cols2 = cols + ["clip_start","clip_end"]
        w = csv.DictWriter(f, fieldnames=cols2); w.writeheader()
        for i, r in enumerate(picks, 1):
            r = dict(r); total = dur_by_conv[r["conv"]]
            r["clip_start"] = round(max(0, r["t_start"]-45), 2)
            r["clip_end"]   = round(min(total, r["t_end"]+15), 2)
            w.writerow({k: r.get(k, "") for k in cols2})

    # ---- 摘要 ----
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(f"事件总数: {len(rows)}\n")
        for k, v in summary.most_common():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n选中事件: {len(picks)}（覆盖 {len(per_conv)} 段对话）\n")
        for k, v in per_conv.most_common():
            f.write(f"  {k}: {v}\n")
    print(open(os.path.join(out, "summary.txt")).read())


if __name__ == "__main__":
    main()
