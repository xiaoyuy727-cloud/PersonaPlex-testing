# 环境与版本记录（2026-10-11 实验环境）

## 硬件 / 系统

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090（32GB，sm_120） |
| 驱动 / CUDA | 595.71.05 / 12.8 运行时（torch cu128 wheel） |
| 系统 | Ubuntu 22.04 容器（root，无 Docker 权限） |

## 软件

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.12.3（miniconda base） | venv `--system-site-packages` 复用 base torch |
| PyTorch | **2.8.0+cu128** | 关键：moshi 依赖声明 `torch<2.5` 会降级导致 Blackwell 不可用，安装时用 `--no-deps` 规避 |
| moshi-personaplex | 0.1.0（本地源码安装） | 源自 NVIDIA/personaplex commit `3428dfd95309a7f3c84fd93259ded0f810d1ff91` |
| sphn | 0.1.12 | |
| sentencepiece | 0.2.0 | |
| huggingface-hub | 0.24.7 | 模型/数据下载 |
| numpy | 2.1.3（venv 内） | moshi 要求 <2.2 |
| einops / safetensors / sounddevice / aiohttp / soundfile | 0.7.0 / 0.4.5 / 0.5.0 / 3.10.11 / 0.14.0 | 完整清单见 `requirements-frozen.txt` |
| 系统包 | libopus-dev、ffmpeg、p7zip-full | ffmpeg 用于音频裁剪/重采样；7z 用于 tar/ISO |

## 模型与数据版本（复现锚点）

| 资源 | 版本锚点 |
|---|---|
| PersonaPlex 权重 | `nvidia/personaplex-7b-v1` @ `fdaf4090a61cb315c138a1faee287ffd6c716309`（bf16，8.37B 参数） |
| PersonaPlex 源码 | commit `3428dfd95309a7f3c84fd93259ded0f810d1ff91` |
| Open Yap 1K 样本 | `TheAgenticDataCompany/open-yap-1k`，2026-10-11 下载；文件校验和见 `dataset_files.sha256` |

## 随机性与确定性

- 全局种子 `42424242`（torch/cuda/numpy/random 同步设置，与官方 `seed_all` 一致）。
- 已验证：同种子两次独立运行的 onset 与文本输出逐字一致。
- 注意：CUDA Graph 捕获与批次无关的浮点非确定性理论上仍可能存在；跨硬件复现不保证逐位一致。
