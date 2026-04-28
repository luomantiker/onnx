# CUDA 自定义算子 ONNX 到 torch 化替换的完整示例

该示例工程用于复现并验证：

1. 含 CUDA 自定义算子的 ONNX 可以通过 ONNXRuntime 推理。
2. 同一模型导出为标准 ONNX 算子后（torch 化），可被地平线工具链编译。
3. 两个 ONNX 推理结果在浮点误差范围内一致。

---

## 1. 背景说明

在 PyTorch 中，CUDA Extension 自定义算子可用于高性能实现；但导出到 ONNX 后通常表现为自定义域节点（例如 `custom_domain::CustomAdd`）。

- ONNXRuntime 可以通过 ONNX Function 展开或自定义注册机制执行此类节点。
- 地平线工具链（`hb_compile`）通常仅支持其白名单内的标准 ONNX 算子，不支持未知自定义算子域。

因此，模型落地地平线平台前，需将自定义算子进行 **torch 化替换**（替换成标准 PyTorch/ONNX 算子）。

---

## 2. 环境准备

> 以下命令均使用 `python3`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch onnx onnxruntime numpy setuptools
```

构建 CUDA Extension 还需要：

- NVIDIA GPU 驱动
- CUDA Toolkit（`nvcc` 可用）
- 与当前 PyTorch 对应的 CUDA 版本

地平线验证还需要：

- 已安装并可调用 `hb_compile`
- 可用的地平线编译环境和许可

---

## 3. 工程目录

```text
samples/horizon_cuda_torchify/
├── cuda_op/
│   ├── __init__.py
│   ├── build.sh
│   ├── custom_add.cpp
│   ├── custom_add_kernel.cu
│   └── setup.py
├── data/
│   └── generate_input.py
├── export/
│   ├── export_cuda_onnx.py
│   └── export_torch_onnx.py
├── horizon_test/
│   ├── compile_cuda_onnx.sh
│   ├── compile_torch_onnx.sh
│   └── config.yaml
├── infer/
│   ├── compare_outputs.py
│   └── infer_onnxruntime.py
├── model/
│   ├── model_torch_impl.py
│   └── model_with_cuda_op.py
└── README.md
```

---

## 4. 步骤说明（一步一步）

### Step 1. 编译 CUDA 自定义算子

```bash
cd samples/horizon_cuda_torchify/cuda_op
./build.sh
```

算子功能：

```text
output = input1 + input2
```

其中：

- CUDA 内核在 `custom_add_kernel.cu`
- Python 可调用入口为 `custom_add(x, y)`

---

### Step 2. 导出带自定义算子的 ONNX

```bash
cd /workspace/onnx/samples/horizon_cuda_torchify
python3 export/export_cuda_onnx.py
```

输出：

- `export/model_cuda.onnx`

该模型包含：

- 自定义节点 `custom_domain::CustomAdd`
- 同时附带 ONNX Function 定义（函数体是标准 `Add`），便于 ONNXRuntime 执行

---

### Step 3. 导出 torch 化 ONNX

```bash
cd /workspace/onnx/samples/horizon_cuda_torchify
python3 export/export_torch_onnx.py
```

输出：

- `export/model_torch.onnx`

该模型只包含标准 ONNX 算子（`Add`）。

---

### Step 4. 生成输入数据

```bash
cd /workspace/onnx/samples/horizon_cuda_torchify
python3 data/generate_input.py --seed 2026
```

输出：

- `data/input.npy`（运行时生成，不入库，避免 PR 二进制文件问题）

---

### Step 5. ONNXRuntime 推理验证

```bash
cd /workspace/onnx/samples/horizon_cuda_torchify
python3 infer/infer_onnxruntime.py
python3 infer/compare_outputs.py
```

输出文件：

- `infer/output_cuda.npy`
- `infer/output_torch.npy`

比较指标：

- MSE
- Cosine Similarity

期望：

- `MSE` 接近 0
- `Cosine Similarity` 接近 1

---

### Step 6. 地平线工具链编译验证

#### 6.1 编译 CUDA 自定义算子 ONNX（预期失败）

```bash
cd /workspace/onnx/samples/horizon_cuda_torchify/horizon_test
./compile_cuda_onnx.sh
```

预期：

- 报错自定义算子不支持（expected failure）

#### 6.2 编译 torch 化 ONNX（预期成功）

```bash
cd /workspace/onnx/samples/horizon_cuda_torchify/horizon_test
./compile_torch_onnx.sh
```

预期：

- 编译通过（expected success）

---

## 5. 预期结果矩阵

| 模型 | ONNXRuntime | Horizon hb_compile |
|---|---|---|
| `model_cuda.onnx`（含自定义算子） | OK | FAIL（预期） |
| `model_torch.onnx`（标准算子） | OK | OK（预期） |

---

## 6. 关键结论（必须）

**torch 化是将自定义算子替换为标准 ONNX 算子的必要步骤，是模型在地平线平台落地的前提条件。**

