# 第三方依赖与参考说明

本文件说明 HTDemucs STFT Ascend C 工程使用的开发工具、运行库、数值参考工具及技术参考资料。

## 1. Huawei Ascend / CANN

本工程在 CANN Toolkit 8.0.0 环境中构建，使用以下组件：

| 组件 | 在本工程中的用途 |
|---|---|
| Ascend C | 编写分帧加窗、high/low 拆分、DFT 矩阵乘法及合并打包内核 |
| Ascend C MatMul API | 调用 Cube 矩阵乘法 |
| ACL / ACLNN | 创建张量描述、查询工作区、调用算子、管理设备内存与同步 |
| 算子注册与 tiling API | 声明接口、推导形状和计算运行参数 |
| `msopgen` 与工程模板 | 在使用者安装的 CANN 环境中生成构建工程和 ACLNN 包装代码 |
| CANN 编译及打包工具 | 编译内核、打包算子资源并生成本地自定义库 |

`op_host/`、`op_kernel/` 和 `runtime/` 中的 CANN 接口由使用者安装的 Toolkit 提供。`configure_project.py` 配置生成工程，`resource_preload_codegen.py` 处理生成的 ACLNN 包装代码，以提供算子资源预加载入口。

开发中使用 CANN STFT Ascend C 源码中的分帧与 DFT 矩阵乘法思路作为技术参考。相关文件名包括 `stft.cpp`、`stft.h`、`stft_generalized.h`、`stft_generalized_complex.h` 和 `stft_plan_mul.h`。本仓库的四算子链路采用固定输入形状、high/low 拆分及面向 HTDemucs 的裁剪和打包接口。

SDK 头文件、运行库、编译器及上述官方 STFT 源码文件不作为本工程源码交付。由 CANN 模板生成的文件应保留模板所带的版权与许可说明；第三方组件的使用以其随附条款为准。

官方资料：

- [CANN 文档入口](https://www.hiascend.com/cann/document)
- [CANN 8.0.0：Ascend C 自定义算子开发实践](https://www.hiascend.com/document/detail/zh/canncommercial/800/devaids/opdev/optool/atlasopdev_16_0027.html)

## 2. PyTorch

`prepare_tests.py` 默认使用 CPU PyTorch 生成周期 Hann 窗和 STFT 数值参考，涉及：

- `torch.hann_window`：生成 `periodic=True` 的 FP32 Hann 窗。
- `torch.stft`：以 `center=False`、`normalized=True`、`onesided=True` 计算 CPU 参考。
- `torch.view_as_real`：将复数频谱转换并排列为实部、虚部四平面。

PyTorch 用于测试数据与参考结果生成，不参与 NPU STFT 内核执行。选择 `--reference numpy` 时不需要 PyTorch。实际参考后端及版本保存在 `test_data/manifest.json`。

官方资料：

- [PyTorch 项目](https://github.com/pytorch/pytorch)
- [torch.stft](https://docs.pytorch.org/docs/stable/generated/torch.stft.html)
- [torch.hann_window](https://docs.pytorch.org/docs/stable/generated/torch.hann_window.html)

## 3. NumPy

NumPy 用于测试信号生成、二进制张量读取、误差计算和结果校验。NumPy 参考模式使用 `numpy.fft.rfft` 及 `norm="ortho"` 生成频谱参考，并使用 `sliding_window_view` 分帧。

该参考模式需要 NumPy 1.20 或更高版本。NumPy 不参与 NPU 内核计算。

官方资料：

- [NumPy 项目](https://github.com/numpy/numpy)
- [numpy.fft.rfft](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html)

## 4. Demucs / HTDemucs

本工程的应用接口面向 HTDemucs：双声道、4096 点 STFT、1024 点帧移、周期 Hann 窗、归一化频谱，以及实部和虚部四平面排列。频点与帧的裁剪范围在本工程 README 中明确给出。

本工程不包含 Demucs 模型权重、OM 模型或完整音频分离推理程序。独立测试通过程序生成信号，无需下载模型或音频数据。

参考资料：

- [Demucs 官方仓库](https://github.com/facebookresearch/demucs)
- [HTDemucs 模型实现](https://github.com/facebookresearch/demucs/blob/main/demucs/htdemucs.py)
- [频谱处理实现](https://github.com/facebookresearch/demucs/blob/main/demucs/spec.py)

## 5. 构建工具

工程使用 Bash、Python、CMake、Make 和 g++ 组织生成、编译与测试流程。这些工具由使用者的系统环境提供，不随本工程分发。

## 6. 许可范围

本文是第三方依赖与参考资料说明，不是本仓库的 `LICENSE`，也不为第三方代码、SDK、库或模型授予许可。

本仓库自身代码的授权应由单独的 `LICENSE` 明确。若文件包含第三方版权或许可声明，应保留相应声明；第三方依赖的授权不因被本工程调用而发生改变。
