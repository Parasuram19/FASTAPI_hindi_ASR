#!/usr/bin/env python3

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    onnx_model = onnx.load("model.onnx")
    quantize_dynamic(
        model_input="model.onnx",
        model_output="model.int8.onnx",
        per_channel=True,
        weight_type=QuantType.QUInt8,
    )


if __name__ == "__main__":
    main()
