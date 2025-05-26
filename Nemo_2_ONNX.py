import nemo.collections.asr as nemo_asr
import os

# ----------------------
# Step 1: Load the .nemo model
# ----------------------
print("Loading the model...")
nemo_model_path = "models/stt_hi_conformer_ctc_medium.nemo"
model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=nemo_model_path)
print("Model loaded successfully.")

# ----------------------
# Step 2: Export to ONNX
# ----------------------
onnx_export_path = "model.onnx"
print(f"Exporting model to ONNX at {onnx_export_path} ...")
model.export(onnx_export_path)
print("ONNX model exported successfully.")

# ----------------------
# Step 3: Create tokens.txt
# ----------------------
print("Creating tokens.txt file...")
vocab = model.decoder.vocabulary
with open("tokens.txt", "w", encoding="utf-8") as f:
    for i, token in enumerate(vocab):
        f.write(f"{token} {i}\n")
    f.write(f"<blk> {i+1}\n")
print("tokens.txt created successfully.")

# ----------------------
# Step 4 (Optional): Add Sherpa metadata
# ----------------------
# Optional: Add metadata using sherpa's script
# Requires: wget + internet or script manually placed in the directory

ADD_METADATA_SCRIPT = "add-model-metadata.py"

if not os.path.exists(ADD_METADATA_SCRIPT):
    print("Downloading Sherpa metadata script...")
    os.system(f"wget https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-small/resolve/main/{ADD_METADATA_SCRIPT}")

print("Adding model metadata for Sherpa...")
os.system("python3 add-model-metadata.py")
print("Metadata added to ONNX model.")

# ----------------------
# Step 5 (Optional): Quantize ONNX model
# ----------------------
QUANT_SCRIPT = "quantize-model.py"

if not os.path.exists(QUANT_SCRIPT):
    print("Downloading quantization script...")
    os.system(f"wget https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-small/resolve/main/{QUANT_SCRIPT}")

print("Quantizing the ONNX model...")
os.system("python3 quantize-model.py")
print("Quantization complete. Output: model.int8.onnx")
