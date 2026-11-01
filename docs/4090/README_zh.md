# Pika H3 Flex：RTX 4090 社区部署与近等价优化

本项目是**非官方社区集成**，与 Pika、MiniMax、Comfy Org 无隶属或背书关系。名称仅用于说明兼容对象，不表示商标授权。目标是在独立 ComfyUI 中，保留原生量化模型、LoRA、采样轨迹和完整音视频输出，通过减少中间分配、控制 FFN 峰值改善 24 GB 显卡的运行路径。

- [英文安装指南（含 Windows、可移植命令）](<INSTALL.md>)
- [英文技术说明、基线来源与验收边界](<OPTIMIZATIONS.md>)
- [Ref2VA 参考生视频：模型下载、工作流、官方示例与实测](<REF2VA.md>)

**Ref2VA 已测通**：官方 Record-shop dance 的两张图＋含音轨参考视频，在 1344×768、243帧/24fps（10.125秒）下完整节点执行 **397.957秒（约6分38秒）**；480p同规格190.218秒。另有独立音频接口480p派生测试190.815秒。均是单次测试，15秒和最大参考数量未验证；本机结果没有保住官方角色的动物面部与尾巴，音频未主观试听。完整测量条件和[官方效果演示](https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/0ddf55ad7067ca1e76812253bd7b4600df87f759/assets/showcases/motion-transfer.mp4)见 Ref2VA 指南。

**不要把旧基线的 376/378 秒当作本次新扩展的实测。** 新低峰值 LoRA、可选 Triton SwiGLU 与局部编译，必须由各自的独立实测来证明收益。本文不承诺任意提示词 7 分钟、不声称全局最快或质量无损。

## 一、先区分两条运行路径

仓库保留了 Pika-Labs 的原版推理源码及授权；其 standalone 路径的代表性测试设备是 H100 80 GB，与此处 4090 部署不是同一个环境。

4090 推荐路线是：**独立 ComfyUI + Comfy Org 预量化组件 + 本仓库 custom node**。原 H100 依赖另存为 `requirements-standalone.txt`；根 `requirements.txt` 仅保留注释，避免 custom-node manager 自动安装 H100 依赖。扩展在兼容 ComfyUI 环境上无额外必需 pip 依赖。不要在 4090 环境安装 standalone requirements，不要运行旧 standalone 下载器，也不要自动下载约 210 GB BF16 基座快照。保留的 `pyproject.toml` 与 `src` 属于 standalone 路径，**不要 `pip install -e .` 来安装本扩展**；Docker/Conda 原 H100 配方也改用独立 requirements。这里只需按需取得对应的单个量化文件；仓库不附带模型权重。

历史验证平台为 Windows、RTX 4090 24 GB、64 GB 系统内存、Python 3.12.14。64 GB 是已测机器配置，不代表严格最低内存。模型、依赖、下载暂存和输出需另留 SSD 空间；GPU 驱动需兼容 torch 的 CUDA 13.0 运行时。

| 项目 | 历史基线版本 |
| --- | --- |
| ComfyUI | 0.43.0，commit `0dbfd493b4cc604db743c699f0f7ce01348e54a4` |
| torch / torchvision / torchaudio | 2.9.1+cu130 / 0.24.1 / 2.9.1 |
| comfy-kitchen / comfy-aimdo | 0.2.40 / 0.6.0 |
| frontend / workflow templates | 1.57.6 / 0.11.100 |

完整已测 Windows 包版本以 [environment.json](<../../benchmarks/results/environment.json>) 为准，包含 CUDA 后缀、frontend、PyAV 与可选 Triton 分发版本；[4090-windows.txt](<../../constraints/4090-windows.txt>) 用于对 ComfyUI requirements 执行 `pip -c` 约束。constraints 限制依赖解析，不会主动安装所有列出的包，也不是完整通用 lockfile。

Linux 命令仅作为可移植部署说明，不是 Linux 测速认证；其他 GPU、其他 ComfyUI 版本、CPU-only、macOS、ROCm、DirectML、最大数量多模态参考输入均未在此完成资格验证。

## 二、Windows 独立部署

在全新工作目录执行，避免覆盖现有 ComfyUI。下面安装的是 **ComfyUI 自己的 requirements**，不是本仓库的 standalone requirements，也不是本仓库只含注释的根 requirements。

```powershell
git clone https://github.com/Comfy-Org/ComfyUI.git ComfyUI-4090
Set-Location ComfyUI-4090
git checkout 0dbfd493b4cc604db743c699f0f7ce01348e54a4
py -3.12 -m venv .venv
$Python = Join-Path $PWD '.venv\Scripts\python.exe'
git clone https://github.com/ling5717/Pika-H3-Flex-4090.git custom_nodes/Pika-H3-Flex-4090
$Constraints = 'custom_nodes/Pika-H3-Flex-4090/constraints/4090-windows.txt'
& $Python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
& $Python -m pip install -r requirements.txt -c $Constraints
& $Python -m pip install comfy-kitchen==0.2.40 comfy-aimdo==0.6.0 -c $Constraints
& $Python -m pip check
& $Python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

安装前应查看依赖解析结果；后续升级可能改变传递依赖。正式测速需记录本扩展实际 checkout commit，不可只写会变化的 `main`。

Windows portable 版从其根目录使用以下解释器/入口（前提是 embedded Python 与目标 wheel 兼容）：

```powershell
$Python = Join-Path $PWD 'python_embeded\python.exe'
git clone https://github.com/ling5717/Pika-H3-Flex-4090.git ComfyUI/custom_nodes/Pika-H3-Flex-4090
& $Python -s -c "import sys, torch; print(sys.version); print(torch.__version__)"
& $Python -s ComfyUI/main.py --listen 127.0.0.1 --port 8189 --disable-auto-launch --offline --use-ck-attention --reserve-vram 3.5
```

不要盲目替换 portable 自带 torch；版本不一致时需自行验证或改用独立 Python 配方。完整仓库放在 `custom_nodes` 下，由根入口加载 `comfy_extension`；不要只拷贝包目录，也不要为加载节点而安装 upstream editable 包。POSIX shell 完整命令见[安装指南](<INSTALL.md>)。

## 三、只下载需要的组件

先阅读模型授权，在 [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main) 按单文件下载并放入 ComfyUI 对应模型目录。以下是公开上游文件标识，不是仓库内附件。

| 目录 | 文件 |
| --- | --- |
| `models/diffusion_models` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| `models/text_encoders` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| `models/vae` | `minimax_h3_video_vae_int8_convrot.safetensors` |
| `models/vae` | `minimax_h3_audio_vae_fp32.safetensors` |
| `models/loras` | `pika_h3_flex_fl2va_v1_bf16.safetensors` |

这是 static pruned INT8 扩散模型、Qwen NVFP4 文本编码器、INT8 视频 VAE、FP32 音频 VAE 与 Comfy Org Pika LoRA，不等于全 BF16/FP32 模型。文件名不足以证明权重身份，复现需记录来源 revision 与 SHA-256。也可用 ComfyUI 的 `--extra-model-paths-config` 引用已有模型盘，不必复制；含个人路径的配置不要直接公开。

## 四、启动、接图与核验

标准 Windows 独立环境内执行：

```powershell
& $Python main.py --listen 127.0.0.1 --port 8189 --disable-auto-launch --offline --use-ck-attention --reserve-vram 3.5
```

打开 `http://127.0.0.1:8189`。`--offline` 用于模型已手动准备后的启动，不负责安装或下载；默认仅本机监听，不要无鉴权对公网暴露。

**必须读后端日志，不能仅看启动参数。** 历史成功环境显示 CK CUDA backend available、`Using Comfy Kitchen attention`、双流 async weight offloading 与 DynamicVRAM。当时 CK 的 Triton backend 为 disabled，CUDA backend 正常；这不代表新扩展的可选 Triton 激活已经通过验证。不要为了编译而随意关掉动态显存或叠加 `--highvram` 等选项。

从 ComfyUI 官方 Pika H3 Flex FL2VA/文生视频模板开始，保留原生条件构建、音视频 VAE 与保存路径。先以短片验证环境，再尝试长片。

历史完整长片规格：

- 1344 × 768，362 帧，24 fps，实际约 15.083 秒，seed 42；
- `BasicGuider` + `KSamplerSelect(lcm)` + `SamplerCustomAdvanced`；
- `ManualSigmas` 为 `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0`，完整 6 NFE；
- 原生 model-only bypass LoRA，strength 1.0；历史 FFN 分块 16384。

通用“六步 Euler”不是该模型发布轨迹的等价替代。改尺寸、少帧、少步必须另列工作负载，不能作为同规格提速。

使用本扩展时，`PikaH3LowPeakLoRA` **替换而非叠加**原生 bypass LoRA loader；其 MODEL 输出经 `PikaH3Optimize` 再接 guider。两个节点均返回单个 MODEL。`PikaH3LowPeakLoRA` 输入为 `model/lora_name/strength/mode`，起步 `strength=1.0`、`mode=low_peak`；另有实验性 `addmm`。`PikaH3Optimize` 输入为 `model/chunk_size/activation` 及 optional `fused_modulation`，起步 FFN `16384`、activation=`native`（原始 MLP 激活，compile 关闭）、`fused_modulation=false`。分块 combo 字符串支持 `8192/16384/32768`，激活模式为互斥的 `native/triton/compiled`。要复现旧基线，应使用旧原生 bypass 路径，不能把新节点成绩与旧结果混标。

## 五、优化具体做什么

### FFN token 分块

仅将逐 token 独立的 MLP 分块执行，限制 gate/up/激活中间张量峰值，不切 attention 序列，不丢 token，不换权重，不减少采样次数。16384 是默认起点，不保证所有机器最优。块过小增加 launch 开销，过大可能增加卸载或 OOM。

实数运算下逐 token MLP 可以分块再拼接；浮点 GEMM 内核可能随 shape 改变，**不能承诺逐位一致**。历史本地 CPU 128 项比较通过，不自动等于新封装实现已通过 GPU 及端到端验收。

### 低峰值普通线性 LoRA

保留 `base + linear(linear(x, down), up) * scale` 计算，尽量复用新分配的 LoRA delta 缓冲；不改写输入或 base 输出。单位 scale 可跳过额外乘法，autograd/dtype 不适合原位时采用 out-of-place 路径。这不是新的融合 INT8 GEMM。

以上指默认 `mode=low_peak`。可选 `mode=addmm` 在 dtype 适合时将第二次 LoRA GEMM、scale 和 residual 合并为 `torch.addmm`，改变原生分步 FP16/BF16 舍入边界，属于单独实验候选，不是舍入保持版；需独立误差和端到端验证，本文不提供其新速度成绩。

使用原生 key 映射和可撤销 patcher injection；未知键、漏映射、重复目标等应报错，不得静默丢弃权重。支持范围为普通二维 linear LoRA，不支持卷积 LoRA、DoRA、mid/reshape 变体及 bypass 叠加。模型切换、卸载与 ejection 正确性仍需验收。

### 可选 SwiGLU

eager 默认沿用 gate-first SwiGLU 语义并减少可安全复用的中间分配。Triton 候选只融合激活，不宣称投影双输出或 INT8 线性层融合；显式保留 SiLU 后 FP16/BF16 舍入，再乘 up。即使保留舍入边界，指数函数实现仍可产生误差，不承诺 bitwise 相同。

helper 有 eager/auto/triton：严格 triton 对不支持的输入或后端失败报错；auto 可回退 eager。须记录实际 `last_backend`，不能将 auto 的 eager 回退测成 Triton 成绩。Windows Triton/toolchain 与驱动组合需单独验证，缺少时用 eager，不自动安装或更换系统 CUDA。

### 可选局部 compile

默认关闭。node 的 `compiled` 模式仅编译 eager SwiGLU 激活 tensor 函数，不编译整个 FFN 或 INT8 投影；采用无模型闭包的 free tensor 子图与 `torch.compile(fullgraph=True, dynamic=True)`。不编译加载、卸载、CPU/GPU 传输和整个队列，不禁用动态显存。wrapper 的 metadata/compiled callable 引用 LRU 有界，不保留输入/输出/权重 tensor，但不承诺 PyTorch 自身 compiler cache 同样有界；清理和可撤销注入仍由集成负责。

首轮编译、shape 变化、重编译、graph breaks 都可抵消收益。Windows Inductor 的可用性与运行时绑定。CPU 测试通过不代表 Inductor/Triton 的真实 GPU 收益；需分别报告首轮与 warm 时间。

### 可选 RMSNorm/AdaLN 融合

`fused_modulation` 为 optional bool，默认 **false**，与激活模式独立。通过原生 cast/offload 生命周期调用共享 Comfy Kitchen `rms_adaln`，将 RMS affine weight 折入 AdaLN scale，保留原 epsilon 和 scalar/per-token tensor segment 行选择语义，不构造全 token 展开的调制表。

affine folding 改变低精度舍入。已有局部 scalar/tensor segment 算子检查不等于完整视频和生命周期全面验收；多个 segment 还需要输出及临时分段分配，不能自动推断省峰值或提速。与其他 block/norm 包装冲突时应报错，不能覆盖；训练/autograd 与 dtype promotion 等适用情况仍保留 native 路径。首次验证保持关闭。

### VAE 预取：本版本未实现

异步 VAE prefetch 为 **NO-GO**，不是隐藏开关：所检查 native runtime 缺合适的 per-queue cleanup hook，且 VAE OOM 会在节点内立即改用 tiling 重试。没有正确取消/清理所有权的预取可能跨重试保留 tensor，因此本扩展不加入私有/全局 cleanup hook，不替换原生 OOM-to-tiling 行为，不宣称该项收益。

## 六、历史实测与证据边界

来源是维护者发布前保存的本地 `PIKA-FAST-README.md`、两次后端日志及请求/缓存记录；本文转录可公开的事实，不包含私有目录，也不暗示读者已获得这些本地原件。新扩展完整性能/验证矩阵统一由 `docs/BENCHMARKS.md` 发布维护，本说明不重复新时长数字，避免多个位置漂移；未发布前不提供不存在的结果链接。

相同模型、提示词和 seed，**原生 bypass + CK + 历史 FFN16384** 两次独立重启后首次请求：

| 历史测试 | Comfy 后端完整执行 | 节点结果缓存 |
| --- | ---: | --- |
| 第一次 | 376.127 秒 | `cached_nodes=[]` |
| 独立新进程复测 | 378.028 秒 | `cached_nodes=[]` |
| 原生 bypass + CK、无 FFN 分块，仅一次 | 561.938 秒 | 部分 loader 已缓存 |

提示词为：“A small paper airplane glides gently through a bright classroom. Soft classroom ambience and a gentle paper flutter, no speech or music.”

计时含该请求实际模型加载、文本编码、6 次采样、音视频解码和 MP4 保存；不含安装与服务启动。OS 文件/驱动缓存未清空，因此 fresh process **不等于严格冷缓存**。无 FFN 那一次缓存条件不同且只有一个样本，不能据此宣称严格 A/B 的百分比提速。

两段历史输出均完整解码 362 帧，H.264 + 32 kHz 双声道 AAC，音频非静音；只目视抽查首/中/尾画面，**没有主观试听音轨**，没有全面的动作、音画同步或跨提示词质量评测。整卡离散显存采样不是逐进程 allocator 峰值，也没有 Linux `gpu_watch`/`resources_clear` 认证。

历史约 1.822 GiB 为原生 208 组 LoRA 的额外驻留量，不是新 loader 的节省量。**新 low-peak LoRA、Triton 激活和局部 compile 的端到端速度、误差与显存收益，不能从上表推导，应查看中央 benchmark 报告。**

已公开 [RTX 4090 算子 probe](<../../benchmarks/results/operator-probe-4090.json>)，含完整 shape、指标定义及重复测量：所测 low-peak LoRA 的 BF16/FP16 与 native 输出相同；所测 Triton SwiGLU 的 BF16 相同、FP16 有小误差，compiled 则有更明显舍入差异。这些只覆盖记录的输入，不是普遍逐位相同或通用误差阈值。单个 seed 的完整解码 RGB 一致也不等于 latent/音轨相同或跨提示词质量结论；微小单次计时差不能宣传为显著提速。

SOL/SLA/VSA 稀疏、跨步预测不在推荐版范围；历史 SOL 尝试主动中止，无成功视频，不计入成功测速。缓存结果的亚秒保存时间也不是生成速度。

## 七、怎样做可信复测

1. 固定模型与 LoRA 哈希、ComfyUI/扩展 commits、运行时版本、实际 kernel backend、参数、提示词与 seed；
2. 对每个候选先做算子误差测试，覆盖 dtype、shape、alpha/scale、布局与 fallback；报告非有限值及绝对/相对误差，不能事后放宽阈值掩盖问题；
3. CUDA 微基准同步并分离 warm-up/编译开销；微内核快不等于完整请求快；
4. 一次只改变一个因素，多轮完整 fresh-process A/B，统一缓存条件并检查 `cached_nodes`；
5. 验证所有帧、时长、分辨率、音轨格式与非静音；主观试听、同步和时序画面质量需独立检查；
6. 测试切换工作流/模型后 hooks 撤销、原始行为恢复、无持续显存增长；未证实清理就不要写“资源完全释放”；
7. 公开日志前清除个人路径、提示词隐私、用户名及密钥，不提交权重或私人媒体。

未建立通用画质/音频误差阈值。一次 CPU 测试、导入成功或一段视频成功，不足以证明所有输入近等价。

## 八、常见问题

- **缺节点：**检查完整仓库位置、重启 ComfyUI 并读 import 日志；不要靠安装旧 standalone requirements 修复。
- **LoRA 映射失败：**核对适配器、模型、普通 linear 支持范围及是否重复加载；不要吞错或忽略未映射键。
- **OOM：**关闭其他 GPU 任务、核对原生动态卸载与预留显存、先跑小规格；换规格要另标结果。
- **Triton/compile 不可用：**维持 eager + compile off；不影响按该路径测试 FFN 和普通 LoRA。
- **耗时异常短：**检查缓存与实际 NFE，确认不是复用最终 latent/视频。
- **能否保证任意平台/提示词相同速度与质量：**不能；未测范围与误差边界见上文。

## 九、授权和合规（互不替代）

- 新增 ComfyUI 集成组件标注 **GPL-3.0-only**；ComfyUI 自身遵循其 GPL 授权。不要把全仓库说成单一 Apache 或单一 GPL。
- 保留的 upstream 推理/加载代码继续遵循 [Apache-2.0](<../../LICENSE-CODE>)，原声明不被覆盖。
- 基座、模型衍生物与适配器继续受 [MiniMax H3 Community License](<../../licenses/MiniMax-H3-LICENSE.txt>) 约束，代码开源不等于权重获得 GPL/Apache 授权。该协议有地域限制（排除欧盟、英国、韩国、美国）、用途限制及商业条件；适用情形需另行取得授权。以完整协议为准，这里不是法律意见。
- 不能因本项目非官方或公开源代码而忽略下游使用、安全、归因和分发义务；不授予商标权，不暗示官方支持。
