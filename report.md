# LLAISYS 项目报告

## 一、已完成工作


|作业 #0-#4（基础） | 张量、算子、模型推理、多平台适配全部完成 |


## 二、作业内容

**作业 #1：张量** — 实现了张量的核心操作：`load`、`isContiguous`、`view`、`permute`、`slice`。所有测试通过。

**作业 #2：算子** — 实现了 9 个 CPU 算子：`add`、`argmax`、`embedding`、`linear`、`rearrange`、`rms_norm`、`rope`、`self_attention`、`swiglu`。支持 Float32/Float16/BFloat16 数据类型，全部测试通过。

**作业 #3：大语言模型推理** — 实现了 DeepSeek-R1-Distill-Qwen-1.5B 模型的完整推理链路：C++ Decoder 实现（Transformer 前向传播 + KV Cache）、C API 导出 + Python ctypes 封装、端到端推理输出与 PyTorch 完全一致。

**作业 #4：多平台 CUDA 适配** — 实现了在 Nvidia GPU 和天数 Iluvatar CoreX GPU 两个平台CUDA 加速推理。

## 三、测试结果
作业1-3 通过了CI自动测试
作业四在算力平台运行成功通过 
平台配置：英伟达：RTX 4090 D - 24 GB * 1
         天数：：智铠100 - 32 GB * 1
