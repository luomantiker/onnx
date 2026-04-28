#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# expected failure: custom_domain::CustomAdd is not supported by Horizon toolchain
hb_compile --config config.yaml --model ../export/model_cuda.onnx
