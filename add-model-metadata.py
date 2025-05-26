#!/usr/bin/env python3

# Copyright (c)  2023  Xiaomi Corporation
# Author: Fangjun Kuang

from typing import Dict

import numpy as np
import onnx


def get_vocab_size():
    with open("tokens.txt") as f:
        return len(f.readlines())


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)
    print(f"Updated {filename}")


def main():
    vocab_size = get_vocab_size()
    # 8 for citrinet
    # 4 for conformer ctc
    subsampling_factor = 4

    meta_data = {
        "vocab_size": str(vocab_size),
        "normalize_type": "per_feature",
        "subsampling_factor": str(subsampling_factor),
        "model_type": "EncDecCTCModelBPE",
        "version": "1",
        "model_author": "nemo",
        "comment": "https://registry.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_small",
    }
    add_meta_data("model.onnx", meta_data)


if __name__ == "__main__":
    main()
