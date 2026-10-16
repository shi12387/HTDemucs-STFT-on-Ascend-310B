# HTDemucs STFT on Ascend 310B

基于 Ascend C 的固定参数 STFT 工程，面向 Orange Pi AI Pro 8T 上的 Ascend 310B4。工程将已完成 padding 的双声道 FP32 波形转换为 HTDemucs 频谱分支使用的实部、虚部四平面张量，并提供工程生成、编译、ACL 调用和 CPU 参考数值验证脚本。

本工程已在 Ascend 310B4、CANN 8.0.0 环境中完成板端验证，六组测试均输出 `mismatch_count=0`。`PASS_STANDALONE_STFT_ALIGNMENT` 由 `validate.py` 在全部输出数值、执行记录和输入恢复检查通过后生成；仅完成编译或执行不会产生该标志。测试流程与判据见 [测试说明](tests/README.md)，板端实测数据见“验证结果与输出文件”。

## 1. 输入与输出

| 项目 | 配置 |
|---|---|
| 输入波形 | `[2,351232]`，连续 FP32，声道顺序为左、右 |
| FFT 长度 | `n_fft=4096` |
| 帧移 | `hop_length=1024` |
| 窗 | 长度 4096 的周期 Hann 窗，`periodic=True` |
| STFT 归一化 | `normalized=True`，即乘以 `1/sqrt(4096)=1/64` |
| 分帧结果 | `[2,340,4096]` |
| 保留频点 | `0..2047`，去掉 Nyquist 频点 2048 |
| 保留帧 | `2..337`，Python 切片为 `[2:338]` |
| 最终输出 | `[1,4,2048,336]`，连续 FP32 |
| 四平面顺序 | `left_real, left_imag, right_real, right_imag` |

输入必须已经包含调用方需要的 padding。CPU 参考使用 `center=False`，不再对输入执行中心补齐。工程不包含音频解码、重采样、波形 padding、频谱 mean/std 统计与模型归一化、模型推理或 iSTFT。

输出包含 2,752,512 个 FP32 元素，共 11,010,048 字节。最后一维为时间帧，因此每个频点对应的 336 个时间值连续存储。输出是带符号的实部与虚部，不是幅度谱、功率谱或对数谱。

## 2. 实现原理

STFT 先将波形切分为重叠帧并乘窗，再计算每一帧的离散傅里叶变换。本工程使用预生成的 `cos`、`-sin` 系数矩阵，将 DFT 转换为 Cube 矩阵乘法。

输出对应的数学目标为：

```text
w[n] = 0.5 - 0.5*cos(2*pi*n/4096)

Z[c,k,t] = sum(n=0..4095) {
    x[c,(t+2)*1024+n] * w[n] * exp(-i*2*pi*k*n/4096)
} / 64

c = 0..1, k = 0..2047, t = 0..335
```

计算由四个算子完成：

| 算子 | 源文件名 | 功能 |
|---|---|---|
| `HtStftFrameWindow` | `ht_stft_frame_window.cpp` | 按固定帧长和帧移分帧，乘以 Hann/64 窗 |
| `HtStftSplit` | `ht_stft_split.cpp` | 将加窗数据乘回 64，拆成 FP16 高位值和残差低位值，并补齐矩阵行数 |
| `HtDftMatmul` | `ht_dft_matmul.cpp` | 对分块数据与 DFT 系数执行 FP16 输入、FP32 输出的 Cube 矩阵乘法 |
| `HtStftMergePack` | `ht_stft_merge_pack.cpp` | 合并四项结果、除以 64，裁剪频点和帧，排列为四平面输出 |

输入矩阵与 DFT 系数均使用 high/low 拆分。对矩阵 `A` 和 `B`，以两组 FP16 值近似原数据，再计算：

```text
A*B ≈ A_high*B_high + A_high*B_low
    + A_low*B_high + A_low*B_low
```

四项结果按内核规定的 FP32 顺序合并，以降低直接转换为单组 FP16 带来的误差。输入、输出为 FP32，内部包含 FP16 运算；该方法仍存在浮点近似误差。

分帧、拆分、DFT 和合并打包均在 NPU 执行。CPU 负责初始化、生成并缓存 DFT 系数、资源管理、ACL 调度和测试验证。窗及 DFT 系数在初始化时上传一次，设备计算链中不回读中间张量。

## 3. 目录与文件

| 路径 | 用途 |
|---|---|
| `op_host/` | 四个算子的接口注册、类型与形状推导、tiling 计算及 tiling 数据结构 |
| `op_kernel/` | 四个 Ascend C 内核的实现 |
| `runtime/` | C++ ACL 测试程序、设备缓冲区管理和 STFT 流水线 |
| `runtime/main.cpp` | 六组测试的执行入口，保存 NPU 输出和执行报告 |
| `runtime/stft_pipeline.h` | 算子资源预加载、张量视图、常量缓存、调用与同步 |
| `runtime/dft_plan.h` | 生成余弦和负正弦 DFT 系数 |
| `tests/` | CPU 参考计算的解析测试，不需要 NPU |
| `tests/README.md` | CPU 与 NPU 测试流程、参考结果计算、通过判据和日志解读 |
| `01_prepare.sh` | 生成测试数据与 CPU 参考，通过 msopgen 创建四算子工程 |
| `02_build_and_test.sh` | 编译算子工程，随后编译并运行 ACL 测试与数值验证 |
| `03_test_only.sh` | 使用已构建的算子库重新编译测试程序并运行验证 |
| `anchor_dft_matmul.json` | msopgen 创建工程时使用的算子描述 |
| `configure_project.py` | 配置多算子源码、内核编译条目及资源预加载生成步骤 |
| `resource_preload_codegen.py` | 为生成的 ACLNN 包装代码加入共享资源注册入口 |
| `cann_common.sh` | 设置 CANN、头文件、库路径及工程根目录 |
| `prepare_tests.py` | 生成固定测试信号、Hann 窗、CPU 参考和输入清单 |
| `baseline_sources.json` | 12 个算子源码文件的 SHA256 校验值 |
| `check_project.py` | 检查算子源码校验值、生成工程源码及构建产物 |
| `begin_run.py` | 检查测试输入，记录本次使用的算子库并清除上次通过标志 |
| `support.py` | 固定形状、文件校验、数据读取等公共函数 |
| `validate.py` | 对全部输出元素进行数值比较，生成验证报告与通过标志 |
| `THIRD_PARTY_NOTES.md` | 第三方依赖、工具和参考资料说明 |

运行时会生成以下目录，无需作为源码提交：

| 目录 | 内容 |
|---|---|
| `StftStandalone/` | msopgen 生成的 CANN 工程、编译中间文件、自定义算子库和 ACLNN 头文件 |
| `bin/` | 编译得到的 `stft_test` 可执行程序 |
| `test_data/` | 测试波形、窗、CPU 参考频谱与清单 |
| `results/` | NPU 输出、数值验证结果和计时报告 |

## 4. 快速开始

### 环境要求

- Orange Pi AI Pro 8T，Ascend 310B4，驱动可正常访问设备。
- CANN Toolkit 8.0.0，包含 `msopgen`、Ascend C 编译工具和 ACL 开发库。
- Bash、CMake、Make、支持 C++11 的 g++。板端构建日志中的 GCC 版本为 11.4.0。
- Python 3.8 或更高版本、NumPy；默认 CPU 参考还需要 PyTorch。NumPy 参考路径使用 `sliding_window_view`，需要 NumPy 1.20 或更高版本。

将源码放在 `~/ascend_work/htdemucs_stft_standalone`。使用 ASCII、无空格路径。为复现已通过的构建路径，工程所在路径各级目录名避免包含芯片标识字符串 `ascend310b`；该路径约束来自板端对照测试，不代表已确认工具内部的路径解析机制。

### 首次生成、编译与测试

```bash
cd "$HOME/ascend_work/htdemucs_stft_standalone"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
chmod go-w .
bash 01_prepare.sh
bash 02_build_and_test.sh
```

`01_prepare.sh` 默认使用 CPU PyTorch 生成参考数据。选择 NumPy 参考时，将准备命令替换为：

```bash
bash 01_prepare.sh --reference numpy
```

两种参考均使用本次生成并传给 NPU 的实际 FP32 Hann 窗。参考后端与版本记录在 `test_data/manifest.json`。

准备脚本要求 `StftStandalone/` 尚不存在。工程已生成时，重新构建使用：

```bash
bash 02_build_and_test.sh
```

已完成构建且源码、测试数据均未改变时，仅重新测试使用：

```bash
bash 03_test_only.sh
```

修改算子或运行脚本后，需要重新生成测试数据清单，并按需要重新编译；文档修改不影响源码校验。使用 PyTorch 参考的示例：

```bash
python3 prepare_tests.py
bash 02_build_and_test.sh
```

默认设备编号为 `0`。例如指定设备运行：

```bash
HTSTFT_DEVICE_ID=0 bash 03_test_only.sh
```

CANN Toolkit 默认路径为 `/usr/local/Ascend/ascend-toolkit/latest`，可通过 `HTSTFT_CANN_ROOT` 设置。构建脚本使用本地生成的 `StftStandalone/build_out/op_api/lib/libcust_opapi.so`，无需将算子库安装到系统 OPP 目录。

CPU 参考解析测试可单独执行：

```bash
python3 tests/test_reference.py
```

## 5. 验证结果与输出文件

完整测试包含三个步骤：`prepare_tests.py` 生成波形和 CPU 参考，`runtime/main.cpp` 通过 ACL 执行 NPU STFT 并保存输出，最后由 `validate.py` 比较全部元素并判定是否通过。脚本职责、参考计算、输出覆盖检查和失败判据详见 [tests/README.md](tests/README.md)。

测试依次运行 `tones_warmup`、`impulse`、`tones`、`noise`、`zero`、`tones_restore`。它们分别覆盖首次执行、脉冲响应、双声道多频正弦、随机信号、全零输入，以及复用同一设备缓冲区后的输入恢复。

每个用例检查全部 2,752,512 个元素，要求：

- `abs(NPU-reference) <= 2e-5 + 2e-5*abs(reference)`。
- 相对 L2 误差不超过 `2e-5`。
- 输出全部为有限值，全零输入的输出数值为零。
- `tones_restore` 与 `tones` 的 NPU 输出逐位相同。

2026-10-16 板端验证记录如下。数据来自 `run.log` 和 `validate.log`，表示所列测试信号的结果。

| 用例 | 最大绝对误差 | 相对 L2 误差 | 超出容差元素数 |
|---|---:|---:|---:|
| `tones_warmup` | 2.8610e-6 | 4.4099e-7 | 0 |
| `impulse` | 2.7940e-9 | 8.9000e-8 | 0 |
| `tones` | 2.8610e-6 | 4.4099e-7 | 0 |
| `noise` | 1.7881e-7 | 4.8976e-7 | 0 |
| `zero` | 0 | 0 | 0 |
| `tones_restore` | 2.8610e-6 | 4.4099e-7 | 0 |

成功时，终端和 `validate.log` 最后输出：

```text
PASS_STANDALONE_STFT_ALIGNMENT
```

`bitwise_difference_count` 记录与 CPU 参考的位表示差异；数值通过条件以误差门限为准。浮点计算及正负零的位表示差异可能使该计数非零。

### 耗时记录

| 项目 | 本次板端记录 |
|---|---:|
| 常量与设备缓冲区准备 `preparation_wall_ms` | 3402.79 ms |
| 首次设备流水线 `pipeline_wall_ms` | 1068.57 ms |
| 其余五次设备流水线 `pipeline_wall_ms` | 468.62–470.70 ms |
| 其余五次输入传输、流水线及输出回读 `wall_ms` | 485.29–488.22 ms |
| 其余五次 DFT 阶段 | 428.56–430.54 ms |

每次流水线包含 7,571 次 kernel 启动和 46 次同步，其中 `HtDftMatmul` 调用 7,568 次。设备计算链中记录的应用层中间拷贝和回读次数均为 0；输入传输与最终输出回读在该链外执行。

计时使用主机 `steady_clock`，包含 ACL 查询、调用、调度及同步，不是设备 Event 测得的纯内核耗时。`wall_ms` 不包含初始化、输出哨兵重置、磁盘读写和 CPU 验证。以上是六用例验证程序的观测结果，不是多轮统计性能基准，也不是与 CPU 的速度对比。

### 生成的结果

| 文件 | 内容 |
|---|---|
| `results/*_npu_f32.bin` | 各用例的裸 FP32 输出，无文件头，形状 `[1,4,2048,336]` |
| `results/run_report.json` | 执行顺序、计时、常量缓存和调用计数 |
| `results/validation.json` | 各用例的数值误差与验证状态 |
| `results/stft_pass.json` | 本次通过标志及输入、结果和算子库校验信息 |
| `results/timing_runs.csv` | 各用例的分项时间及预热标记 |
| `prepare.log`、`build.log` | 工程生成和算子编译日志 |
| `app_build.log`、`run.log`、`validate.log` | 测试程序编译、运行与数值验证日志 |

读取一个输出的示例：

```python
import numpy as np

y = np.fromfile("results/tones_npu_f32.bin", dtype="<f4").reshape(1, 4, 2048, 336)
left_real = y[0, 0]
left_imag = y[0, 1]
right_real = y[0, 2]
right_imag = y[0, 3]
```

运行在编译、资源加载或 ACL 调用阶段失败时，脚本会停止，可能尚未生成 `validate.log`。此时应先检查对应步骤的日志；缺少验证日志本身不表示 STFT 数值不一致。

## 6. 第三方说明

第三方依赖与技术参考见 [THIRD_PARTY_NOTES.md](THIRD_PARTY_NOTES.md)。该文件用于记录来源，不代替仓库许可证。
