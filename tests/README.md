# STFT 测试说明

本文件说明测试输入如何生成、CPU 参考如何计算、NPU 输出如何比较，以及 `PASS_STANDALONE_STFT_ALIGNMENT` 的产生条件。工程接口、快速开始和实测误差及耗时见 [工程 README](../README.md)。

## 1. 两类测试与运行命令

| 测试 | 入口 | 是否需要 NPU | 检查内容 | 成功输出 |
|---|---|---|---|---|
| CPU 解析测试 | `tests/test_reference.py` | 否 | 参考计算的归一化、实虚部顺序、符号与帧裁剪 | Python unittest 的 `OK` |
| 板端数值测试 | `03_test_only.sh`，或构建后由 `02_build_and_test.sh` 调用 | 是 | 四算子完整执行、全部输出的 CPU 对照、全零及恢复检查 | `PASS_STANDALONE_STFT_ALIGNMENT` |

CPU 解析测试不会运行 NPU，也不会生成板端通过标志。

以下命令在工程根目录执行。只运行 CPU 解析测试：

```bash
python3 tests/test_reference.py
```

首次生成工程、编译并完成板端验证：

```bash
bash 01_prepare.sh
bash 02_build_and_test.sh
```

算子已构建且源码、测试数据未改变时，重新运行板端测试：

```bash
bash 03_test_only.sh
```

对已完成的执行记录重新进行数值判定：

```bash
python3 validate.py
```

最后一个命令只读取保存的输入、参考、执行报告与 NPU 输出，不会重新运行 NPU；它要求相关文件与算子库仍满足本次测试记录的校验条件。

## 2. 文件职责与测试流程

| 文件 | 职责 |
|---|---|
| `prepare_tests.py` | 生成波形、Hann 窗和 CPU 参考，并记录文件及源码的 SHA256 |
| `check_project.py` | 检查 12 个算子源码文件、生成工程源码、ACLNN 头文件和自定义库 |
| `begin_run.py` | 核对输入清单，记录本次算子库，并清除通过标志与执行报告 |
| `runtime/main.cpp` | 逐个执行六组输入，检查输出覆盖，保存结果和执行报告 |
| `runtime/stft_pipeline.h` | 分配并复用设备缓冲区、缓存系数、执行四算子链、统计调用与拷贝 |
| `validate.py` | 校验执行记录，逐元素比较 CPU 与 NPU 输出，检查全零与恢复结果 |
| `tests/test_reference.py` | 使用解析预期值检查 CPU 参考中的关键约定 |

`tests/` 存放 CPU 解析测试和测试说明；板端测试入口和数值判定脚本位于上表所列位置。

完整流程为：生成输入与参考，构建算子和 ACL 程序，运行六组 NPU 测试，最后由 Python 判定全部结果。运行程序的完成提示与最终数值通过标志分别由不同阶段产生。

## 3. CPU 参考如何计算

### 固定接口

| 项目 | 约定 |
|---|---|
| 输入 | `[2,351232]` 连续 FP32，已经包含调用方需要的 padding |
| 窗 | 长度 4096 的 FP32 周期 Hann 窗 |
| STFT | `n_fft=4096`，`hop_length=1024`，`normalized=True` |
| 中心补齐 | `center=False`，不再额外添加 padding |
| 频点裁剪 | `[:2048]`，去掉 Nyquist 频点 |
| 时间裁剪 | `[2:338]`，保留原始帧 2 至 337 |
| 输出 | `[1,4,2048,336]`，顺序为左实部、左虚部、右实部、右虚部 |

### 默认 PyTorch 路径

`prepare_tests.py` 默认用 CPU PyTorch 生成参考：

```python
z = torch.stft(
    torch.from_numpy(x),
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    window=torch.from_numpy(window),
    center=False,
    normalized=True,
    onesided=True,
    return_complex=True,
)
z = z[:, :2048, 2:338]
y = torch.view_as_real(z).permute(0, 3, 1, 2).contiguous()
reference = y.reshape(1, 4, 2048, 336).numpy()
```

其中 `window` 为 `torch.hann_window(4096, periodic=True, dtype=torch.float32)` 生成的未缩放 Hann 窗。

NPU 的 FrameWindow 输入使用同一窗除以 64。Split 在拆分前乘回 64，MergePack 最后除以 64，因此整个 NPU 链的归一化目标与 `normalized=True` 对应。CPU 参考不应再额外除以 64。

### NumPy 路径

准备时选择 `--reference numpy`，则使用：

1. `sliding_window_view` 与步长 1024，生成 `[2,340,4096]` 帧。
2. 各帧与本次的 FP32 Hann 窗相乘。
3. 将加窗值转换为 FP64，执行 `numpy.fft.rfft(..., norm="ortho")`。
4. 执行同样的频点与帧裁剪，排列四平面，结果转换为连续 FP32。

参考结果不读取 NPU 输出，也不调用 NPU 的 DFT MatMul 链。参考后端及版本、随机种子、窗文件和参数记录在 `test_data/manifest.json`。

PyTorch 与 NumPy 模式分别生成本次的实际 FP32 Hann 窗；不同模式的窗末位可能不同。每次 NPU 输入窗与该次参考窗保持一致，不能混用不同准备任务的文件。

## 4. 六组板端测试

| 用例 | 输入特点 | 检查目的 |
|---|---|---|
| `tones_warmup` | 左右声道含不同频率、振幅和正弦/余弦组合 | 首次执行；预热用例仍参与数值验证 |
| `impulse` | 两声道在不同位置具有不同符号和幅度的脉冲 | 检查分帧位置、相位、窗口和裁剪 |
| `tones` | 与预热相同的多频信号，包含高频分量 | 非首次执行的频谱和实虚部顺序 |
| `noise` | 固定种子的随机信号，右声道额外缩放 | 检查密集频谱和声道区分 |
| `zero` | 波形全部为零 | 检查设备缓冲区是否残留非零结果 |
| `tones_restore` | 复用同一组设备地址重新写入 tones | 检查输入变化后能否恢复逐位一致的 NPU 输出 |

随机种子默认为 `2026`。六组测试均使用 `[2,351232]` FP32 输入，每组验证完整的 2,752,512 个输出元素。

窗和 DFT 系数在初始化时上传一次，各用例复用常量和设备缓冲区。NPU 计算链为：

```text
HtStftFrameWindow → HtStftSplit → HtDftMatmul → HtStftMergePack
```

CPU 在链前上传波形，在链后回读最终 packed 频谱，不在链内重建或修正中间张量。

## 5. PASS 的完整条件

### NPU 执行与输出覆盖

每个用例执行前，C++ 程序将整个最终输出有效区填入 NaN 哨兵。运行四算子链并同步后，回读输出，检查所有值均为有限数。未被覆盖的 NaN、计算产生的 NaN 或无穷大都会导致运行失败。

程序同时检查本次调用与传输计数：

- 7,568 次 MatMul 与其余三个算子各一次，共 7,571 次 kernel 启动。
- 46 次 stream 同步。
- 设备计算链内没有应用层中间上传或回读。
- 输入为 2,809,856 字节，最终输出为 11,010,048 字节。

本测试的输出覆盖检查使用有效区 NaN 哨兵，没有在输出有效区两侧额外设置 guard 标记；该检查不代表已验证所有设备内存访问均无越界。

只有六组用例均执行完成，C++ 程序才写出 `results/run_report.json`，状态为 `RUN_COMPLETE_REQUIRES_NUMERICAL_VALIDATION`。终端的 `COMPLETE: six cases...` 只表示执行完成，后续仍需数值验证。

### 文件与执行记录校验

`validate.py` 还要求：

- 输入与 CPU 参考文件的大小及 SHA256 和清单相符，清单记录的源码未改变。
- 输入清单未在运行中改变，本次算子库与运行记录的大小及 SHA256 相符。
- 六组用例的顺序、预热标记、调用与同步次数、中间拷贝计数及输入输出大小正确。
- 常量上传 3 次，共 67,649,536 字节；共享工作区为 2,097,152 字节。
- 记录的计时为非负有限数；耗时长短不作为数值通过门限。

文件校验用于关联本次使用的输入、参考、源码、库与执行记录，不替代数值比较。

### 全元素误差判定

设 `a` 为 NPU 输出，`b` 为 CPU 参考，要求：

```text
对全部元素：
    abs(a[i] - b[i]) <= 2e-5 + 2e-5 * abs(b[i])

relative_l2_error = sqrt(sum((a-b)^2) / sum(b^2)) <= 2e-5
```

NPU 输出和保存的参考均为 FP32；误差指标在 CPU 上以 FP64 计算。若参考能量为零，只有误差也为零才将相对 L2 误差记为 0，否则为无穷大并失败。

还必须通过两个严格检查：

- `zero` 用例的全部输出数值为零，正零与负零均视为数值零。
- `tones_restore` 与 `tones` 的 NPU 输出逐位相同。

### 生成通过标志

上述条件全部通过后，`validate.py` 才会生成：

- `results/validation.json`：误差指标与验证状态。
- `results/stft_pass.json`：通过状态、输入与输出校验信息和算子库信息。
- `results/timing_runs.csv`：六组用例的分项耗时及预热标记。

最后打印：

```text
PASS_STANDALONE_STFT_ALIGNMENT
```

标准流程在运行前清除通过标志，验证程序也会在开始判定时清除通过标志；任何检查失败都不会生成新的有效 PASS。日志中的单条 `[CHECK]` 只对应一个用例，必须看到最终 PASS 才表示整体通过。

## 6. 如何解读日志

| 字段或提示 | 含义 |
|---|---|
| `COMPLETE: six cases...` | 执行完成，仍需运行数值验证 |
| `checked_elements` | 每组比较的元素数，应为 2,752,512 |
| `mismatch_count` | 超出逐元素容差的数量，必须为 0 |
| `bitwise_difference_count` | 与 CPU 参考的位表示差异数，不要求普遍为 0 |
| `max_abs_error` | 最大绝对误差，记录指标，不单独代替逐元素判据 |
| `rmse` | 均方根误差，记录指标，不另设通过门限 |
| `relative_l2_error` | 相对 L2 误差，必须不超过 `2e-5` |
| `intermediate_host_copies` / `intermediate_readbacks` | 设备计算链内的中间主机拷贝与回读，必须为 0 |
| `PASS_STANDALONE_STFT_ALIGNMENT` | 所有执行与数值判据均通过 |

浮点计算顺序及正负零的位表示差异可能使 `bitwise_difference_count` 非零。例如全零输出可以全部为数值零，但部分元素的符号位与 CPU 不同。这不等同于 `mismatch_count` 非零。

编译或 ACL 运行失败时，Shell 脚本会停止，可能还未生成 `validate.log`。若已有验证日志但缺少 PASS，应检查异常及对应用例；不能将已打印的部分结果视为整体通过。

## 7. CPU 解析测试和适用范围

`tests/test_reference.py` 包含五项检查：

1. `test_baseline_sources`：检查 12 个算子源码文件的 SHA256。
2. `test_zero`：零波形的 CPU 参考输出为数值零。
3. `test_cos_sin_planes_and_normalization`：使用单频余弦与正弦，检查输出形状、四平面顺序、正负号及归一化后的预期幅度。
4. `test_shifted_impulse_and_frame_crop`：用已知位置脉冲检查保留帧的索引、窗权重、相位和左右声道区分。
5. `test_restore_cases_are_identical`：检查 tones、预热与恢复用例生成的波形相同。

板端测试验证的是所列固定形状合成波形的最终 packed STFT。它不逐项回读全部中间帧或 MatMul 结果，也不验证音频解码、外部 padding、mean/std 模型归一化、Encoder/Decoder、iSTFT、任意输入幅度或其他 FFT 参数。

PASS 证明本次测试满足上述数值与执行条件，不表示达到特定速度或优于 CPU。计时为包含 ACL 查询、调用、调度与同步的主机墙钟时间；具体耗时与计时边界见工程 README。
