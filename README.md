# Overview
This repository provides working examples that demonstrate how to use OpenVINO as an execution backend in ONNX Runtime for both Python and C++ applications.

There are mainly two parts:
 - Python example
 - C++ example

## OpenVINO + ONNX Runtime in Python
Follow the steps below to install and run the Python examples.

  - Installation:
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install onnxruntime-openvino
    ```
  - Usage:
    ```
    # Legacy ONNX-Runtime API
    python3 test_onnx_only.py

    # ONNX-Runtime with OpenVINO
    python3 test_openvino_only.py

    # Compare ONNX-Runtime with/without OpenVINO
    python3 test_both_onnx_openvino.py

    # Run based on OpenVINO API without ONNX-Runtime
    python3 test_openvino_api.py
    ```


## OpenVINO + ONNX Runtime in C++

TBD...

- Reference: https://github.com/intel/onnxruntime
- Reference: https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-archive-linux.html
- Reference: https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html
