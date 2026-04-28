# CUDA 自定义算子 torch 化适配地平线平台指导工程

本工程定位为**客户改造指导模板**：帮助你把现网 PyTorch/CUDA 自定义算子模型，改造成可在地平线工具链 (`hb_compile`) 上编译部署的 ONNX 模型。

核心思路：

1. 先保留原始 CUDA 自定义算子导出 ONNX（用于对照验证）。
2. 再将自定义算子 torch 化为标准算子导出 ONNX（用于平台落地）。
3. 验证两者数值一致性，并确认地平线编译行为差异。

---

## 1. 工程目录（工程根目录即 `horizon_cuda_torchify`）

```text
horizon_cuda_torchify/
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
│   ├── config_cuda.yaml
│   └── config_torch.yaml
├── infer/
│   ├── compare_outputs.py
│   └── infer_onnxruntime.py
├── model/
│   ├── model_torch_impl.py
│   └── model_with_cuda_op.py
└── README.md
```

---

## 2. 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch onnx onnxruntime numpy setuptools
```

额外依赖：

- CUDA Toolkit（用于构建 extension）
- 地平线工具链（`hb_compile`）

---

## 3. 客户改造步骤

### Step 1. 编译 CUDA 自定义算子

```bash
cd cuda_op
./build.sh
cd ..
```

---

### Step 2. 导出 CUDA 自定义算子 ONNX（对照组）

```bash
python3 export/export_cuda_onnx.py
```

输出：`export/model_cuda.onnx`

---

### Step 3. 导出 torch 化 ONNX（落地组）

```bash
python3 export/export_torch_onnx.py
```

输出：`export/model_torch.onnx`

---

### Step 4. 生成输入与标定数据（双输入 x/y）

```bash
python3 data/generate_input.py --seed 2026 --num-calib 5
```

数据组织：

- 推理输入：
  - `data/inference/x.npy`
  - `data/inference/y.npy`
- 标定输入（按输入名分目录）：
  - `data/calibration/x/0.npy ...`
  - `data/calibration/y/0.npy ...`

---

### Step 5. ONNXRuntime 数值一致性验证

```bash
python3 infer/infer_onnxruntime.py
python3 infer/compare_outputs.py
```

检查指标：

- `MSE` 趋近 0
- `Cosine Similarity` 趋近 1

---

### Step 6. 地平线工具链编译验证

#### 6.1 CUDA 自定义算子 ONNX（预期失败）

```bash
cd horizon_test
./compile_cuda_onnx.sh
```

说明：命令只使用 `hb_compile --config ...`，模型路径来自 `config_cuda.yaml` 的 `onnx_model`。

#### 6.2 torch 化 ONNX（预期成功）

```bash
./compile_torch_onnx.sh
cd ..
```

说明：命令只使用 `hb_compile --config ...`，模型路径来自 `config_torch.yaml` 的 `onnx_model`。

---

## 4. 关键配置说明

- `march` 已配置为 `nash-p`（地平线目标架构）。
- CUDA 与 torch 版本分别使用独立 config：
  - `horizon_test/config_cuda.yaml`
  - `horizon_test/config_torch.yaml`

---

## 5. 交付结论（面向客户）

当客户现有模型包含 CUDA 自定义算子时，需先完成 **torch 化替换为标准 ONNX 算子**，再进入地平线编译流程；这是模型在地平线平台可编译、可部署的前提条件。
