#!/usr/bin/env python3
"""显存时间线诊断：定位 harness 中 case 之间显存增长的位置"""
import gc, os, subprocess, numpy as np, torch, sentencepiece
from huggingface_hub import hf_hub_download
from moshi.models import loaders, LMGen
from moshi.models.lm import load_audio as lm_load_audio, _iterate_audio, encode_from_sphn, SILENCE_TOKENS

DEV = "cuda"
def m(tag):
    gc.collect(); torch.cuda.empty_cache()
    a = torch.cuda.memory_allocated()/2**30
    r = torch.cuda.memory_reserved()/2**30
    mx = torch.cuda.max_memory_allocated()/2**30
    print(f"[mem] {tag}: allocated={a:.2f}G reserved={r:.2f}G peak={mx:.2f}G", flush=True)

OUT = os.environ.get("PP_OUT_DIR", "outputs/pp_smoke")
mimi_w  = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
moshi_w = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
m("start")
mimi   = loaders.get_mimi(mimi_w, DEV); m("mimi1 loaded")
mimi_b = loaders.get_mimi(mimi_w, DEV); m("mimi2 loaded")
omimi  = loaders.get_mimi(mimi_w, DEV); m("mimi3 loaded")
lm     = loaders.get_moshi_lm(moshi_w, device=DEV); lm.eval()
nbytes = sum(p.numel()*p.element_size() for p in lm.parameters())
print(f"[model] lm params={sum(p.numel() for p in lm.parameters())/1e9:.2f}B, bytes={nbytes/2**30:.2f}GiB, dtype={next(lm.parameters()).dtype}")
print(f"[model] cfg context={getattr(lm, 'context', '?')} num_codebooks={lm.num_codebooks} dep_q={lm.dep_q} delays={lm.delays}")
m("lm loaded")

frame_size = int(mimi.sample_rate / mimi.frame_rate)
lm_gen = LMGen(lm, audio_silence_frame_cnt=int(0.5*mimi.frame_rate), sample_rate=mimi.sample_rate,
               device=DEV, frame_rate=mimi.frame_rate, use_sampling=True)
for x in (mimi, mimi_b, omimi): x.streaming_forever(1)
lm_gen.streaming_forever(1)

# warmup（同 harness）
for _ in range(4):
    z = torch.zeros(1, 1, frame_size, device=DEV)
    c = mimi.encode(z); mimi_b.encode(z); omimi.encode(z)
    for k in range(c.shape[-1]):
        t = lm_gen.step(c[:, :, k:k+1])
        if t is not None: omimi.decode(t[:, 1:9])
torch.cuda.synchronize(); m("warmup done")

a_samp = lm_load_audio(f"{OUT}/case_pause/clip_a.wav", mimi.sample_rate)
b_samp = lm_load_audio(f"{OUT}/case_pause/clip_b.wav", mimi.sample_rate)
zero_f = lm_gen._encode_zero_frame(); sine_f = lm_gen._encode_sine_frame()

def reset_all():
    mimi.reset_streaming(); mimi_b.reset_streaming(); omimi.reset_streaming(); lm_gen.reset_streaming()

# ---- 第 1 轮：编码 + 回放 + 生成，模仿 case 1 ----
reset_all(); m("reset#0")
a_codes = torch.cat([x for x in encode_from_sphn(mimi, _iterate_audio(a_samp, frame_size, pad=True), 1)], dim=-1)
m(f"encode A done (frames={a_codes.shape[-1]})")
b_codes = torch.cat([x for x in encode_from_sphn(mimi_b, _iterate_audio(b_samp, frame_size, pad=True), 1)], dim=-1)
m(f"encode B done (frames={b_codes.shape[-1]})")

n = a_codes.shape[-1]
for i in range(0, n, 100):
    hi = min(i+100, n)
    for j in range(i, hi):
        lm_gen.step(input_tokens=a_codes[:, :, j:j+1], moshi_tokens=(b_codes[:, :, j:j+1] if j % 2 == 0 else zero_f), text_token=3)
    m(f"replay steps {hi}/{n}")

outs = []
for j in range(n):
    o = lm_gen.step(input_tokens=a_codes[:, :, j:j+1])
    if o is not None:
        outs.append(o)
        omimi.decode(o[:, 1:9])
m(f"free-run done (outs={len(outs)})")
del outs, a_codes, b_codes

# ---- 第 2 轮：reset 后重编码（harness 中 case2 的崩溃点）----
reset_all(); m("reset#1")
a_codes = torch.cat([x for x in encode_from_sphn(mimi, _iterate_audio(a_samp, frame_size, pad=True), 1)], dim=-1)
m(f"2nd encode A done (frames={a_codes.shape[-1]})")
print("DIAG OK")
