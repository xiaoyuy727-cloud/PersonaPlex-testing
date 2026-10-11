#!/usr/bin/env python3
"""最小 probe：文本流 teacher-forcing 可行性验证（不改正式协议）

测试两个组合：
  T1（官方组合，对应 _step_text_prompt_core）：moshi=静音帧 + input=sine + 文本=词 token
  T2（回放扩展组合）：moshi=真实B音频(强制) + input=sine + 文本=词 token（回放中同步喂 B 的转写）
验证：无报错 / 输出文本流在 lag=1 回显强制 token / 释放后自由生成正常。
"""
import json, os, torch, sentencepiece, numpy as np
from huggingface_hub import hf_hub_download
from moshi.models import loaders, LMGen
from moshi.models.lm import load_audio as lla, _iterate_audio, encode_from_sphn

DEV = "cuda"
def log(*a): print(*a, flush=True)

with torch.no_grad():
    mimi_w  = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    moshi_w = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    tok_w   = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
    tokenizer = sentencepiece.SentencePieceProcessor(tok_w)
    mimi  = loaders.get_mimi(mimi_w, DEV); mimi_b = loaders.get_mimi(mimi_w, DEV); omimi = loaders.get_mimi(mimi_w, DEV)
    lm    = loaders.get_moshi_lm(moshi_w, device=DEV); lm.eval()
    fs    = int(mimi.sample_rate / mimi.frame_rate)
    lg    = LMGen(lm, audio_silence_frame_cnt=int(0.5*mimi.frame_rate), sample_rate=mimi.sample_rate,
                  device=DEV, frame_rate=mimi.frame_rate, use_sampling=True)
    for m in (mimi, mimi_b, omimi): m.streaming_forever(1)
    lg.streaming_forever(1)
    for _ in range(4):
        z = torch.zeros(1, 1, fs, device=DEV)
        c = mimi.encode(z); mimi_b.encode(z); omimi.encode(z)
        for k in range(c.shape[-1]):
            t = lg.step(c[:, :, k:k+1])
            if t is not None: omimi.decode(t[:, 1:9])
    torch.cuda.synchronize(); log("预热完成")

    zero_f, sine_f = lg._encode_zero_frame(), lg._encode_sine_frame()
    texts = ["Well, I mean,", " you know,", " that's", " a", " good", " point."]
    ALL = texts
    ids = [tokenizer.encode(t) for t in ALL]
    flat = [i for sub in ids for i in sub]
    log(f"待强制文本: {ALL}\n  编码: {flat} -> {[tokenizer.id_to_piece(i) for i in flat]}")

    def reset():
        mimi.reset_streaming(); mimi_b.reset_streaming(); omimi.reset_streaming(); lg.reset_streaming()

    def piece(tid):
        return {0:"EPAD",1:"BOS",2:"EOS",3:"PAD"}.get(tid) or tokenizer.id_to_piece(tid).replace("▁"," ")

    # ---------- T1: 官方组合（moshi=静音）----------
    reset()
    forced, outs = [], []
    n1 = len(flat) + 10
    for i in range(n1):
        tt = flat[i % len(flat)] if i < len(flat) else 3       # 后半段强制 PAD
        forced.append(tt)
        o = lg.step(input_tokens=sine_f, moshi_tokens=zero_f, text_token=tt)
        if o is not None:
            outs.append((i, int(o[0,0,0].item())))
    echo = all(outs[k][1] == forced[outs[k][0]-lg.max_delay] for k in range(len(outs)) if outs[k][0]-lg.max_delay >= 0)
    out_txt = "".join(piece(t) for _, t in outs)
    log(f"[T1] 步数={n1} 输出帧={len(outs)} 强制回显(逐帧与输入一致)={echo}")
    log(f"[T1] 输出文本流: {out_txt}")
    # 释放：自由生成 12 步
    free = []
    for i in range(12):
        o = lg.step(input_tokens=sine_f)
        if o is not None: free.append(piece(int(o[0,0,0].item())))
    log(f"[T1] 释放后自由生成（前12帧）: {''.join(free)}")

    # ---------- T2: 回放扩展组合（moshi=真实B音频）----------
    reset()
    OUT = os.environ.get("PP_OUT_DIR", "outputs/pp_smoke")
    b_samp = lla(os.path.join(OUT, "case_overlap2/clip_b.wav"), mimi.sample_rate)
    b_codes = torch.cat([x for x in encode_from_sphn(mimi_b, _iterate_audio(b_samp, fs, pad=True), 1)], dim=-1)
    seg = b_codes[:, :, 100:160]          # 4.8s 的 B 语音
    forced, outs = [], []
    for i in range(seg.shape[-1] + 12):
        if i < seg.shape[-1]:
            tt = flat[i % len(flat)]
            o = lg.step(input_tokens=sine_f, moshi_tokens=seg[:, :, i:i+1], text_token=tt)
        else:
            tt = 3
            o = lg.step(input_tokens=sine_f, moshi_tokens=seg[:, :, -1:], text_token=tt)
        forced.append(tt)
        if o is not None: outs.append((i, int(o[0,0,0].item())))
    echo2 = all(outs[k][1] == forced[outs[k][0]-lg.max_delay] for k in range(len(outs)) if outs[k][0]-lg.max_delay >= 0)
    out_txt2 = "".join(piece(t) for _, t in outs)
    log(f"[T2] 步数={seg.shape[-1]+12} 输出帧={len(outs)} 强制回显(逐帧与输入一致)={echo2}")
    log(f"[T2] 输出文本流: {out_txt2[:200]}")
    # 释放 12 步（moshi 同时停止强制）
    free = []
    for i in range(12):
        o = lg.step(input_tokens=sine_f)
        if o is not None: free.append(piece(int(o[0,0,0].item())))
    log(f"[T2] 释放后自由生成: {''.join(free)}")
    print("TEXT_FORCE_PROBE_DONE")
