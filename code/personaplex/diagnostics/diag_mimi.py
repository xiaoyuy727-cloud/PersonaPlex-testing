#!/usr/bin/env python3
"""定位 mimi 编码每次 +3.1GB 的来源：遍历所有 streaming 状态的张量"""
import gc, os, dataclasses, torch, numpy as np
from moshi.models import loaders
from moshi.models.lm import load_audio, _iterate_audio, encode_from_sphn

DEV = "cuda"
def m(tag):
    gc.collect(); torch.cuda.empty_cache()
    print(f"[mem] {tag}: allocated={torch.cuda.memory_allocated()/2**30:.2f}G", flush=True)

def tensor_sizes(obj, prefix=""):
    out = []
    if isinstance(obj, torch.Tensor):
        out.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items(): out += tensor_sizes(v, f"{prefix}.{k}" if prefix else k)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj): out += tensor_sizes(getattr(obj, f.name), f"{prefix}.{f.name}" if prefix else f.name)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj): out += tensor_sizes(v, f"{prefix}[{i}]")
    return out

def dump_state(mod, prefix=""):
    tot = 0
    st = getattr(mod, "_streaming_state", None)
    if st is not None:
        for name, t in tensor_sizes(st):
            mb = t.numel() * t.element_size() / 2**20
            tot += mb
            if mb > 0.5:
                print(f"    {prefix}|{name}: {tuple(t.shape)} {t.dtype} {mb:.1f}MB")
    for cname, child in mod.named_children():
        tot += dump_state(child, f"{prefix}.{cname}" if prefix else cname)
    return tot

mimi_w = loaders.DEFAULT_REPO
from huggingface_hub import hf_hub_download
w = hf_hub_download(mimi_w, loaders.MIMI_NAME)
mimi = loaders.get_mimi(w, DEV)
print("mimi cfg:", getattr(mimi, "context", "?"), "| frame_rate:", mimi.frame_rate, "| sr:", mimi.sample_rate)
m("mimi loaded")
mimi.streaming_forever(1)

def encode(path, tag):
    samp = load_audio(path, mimi.sample_rate)
    codes = torch.cat([x for x in encode_from_sphn(mimi, _iterate_audio(samp, 1920, pad=True), 1)], dim=-1)
    m(f"{tag}: encoded {codes.shape[-1]} frames")
    return codes

# 1) 长 clip 编码
OUT = os.environ.get("PP_OUT_DIR", "outputs/pp_smoke")
codes = encode(os.path.join(OUT, "case_pause1/clip_a.wav"), "run1(765帧)")
tot = dump_state(mimi, "mimi")
print(f"[state] 状态张量总计 {tot:.1f} MB（仅 >0.5MB 的已列出）")

# 2) reset 后看显存是否回收
mimi.reset_streaming(); m("reset 后")
tot = dump_state(mimi, "mimi")
print(f"[state] reset 后状态总计 {tot:.1f} MB")

# 3) 短 clip 编码（100 帧）看增长是否线性
import subprocess
subprocess.run(["ffmpeg","-y","-loglevel","error","-t","8","-i",os.path.join(OUT, "case_pause1/clip_a.wav"),"-ar","24000","/tmp/short.wav"], check=True)
codes2 = encode("/tmp/short.wav", "run2(短)")
m("短编码后")
