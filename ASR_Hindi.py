import nemo.collections.asr as nemo_asr
from pyctcdecode import build_ctcdecoder
import torchaudio
import numpy as np
import torch
# Paths to your files
NEMO_MODEL_PATH = "models/stt_hi_conformer_ctc_medium.nemo"
KENLM_MODEL_PATH = "models/stt_hi_conformer_ctc_medium_kenlm.bin"
AUDIO_PATH = "suits-la-pilot-scene-leak-nbc_TxYi5gsM.mp3"  # Must be 16kHz mono WAV

# Step 1: Load ASR model
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(NEMO_MODEL_PATH)

# Step 2: Extract vocabulary
vocab = asr_model.decoder.vocabulary

# Step 3: Build beam search decoder with KenLM
decoder = build_ctcdecoder(
    labels=vocab,
    kenlm_model_path=KENLM_MODEL_PATH
)

# Step 4: Load audio
waveform, sample_rate = torchaudio.load(AUDIO_PATH)

# Resample to 16000 Hz if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Convert to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Step 5: Get log probabilities from the ASR model
asr_model.eval()
with torch.no_grad():
    log_probs = asr_model.forward(input_signal=waveform, input_signal_length=torch.tensor([waveform.shape[1]]))
    log_probs = log_probs[0][0].cpu().numpy()  # shape: [time, vocab_size]

# Step 6: Decode using KenLM beam search
transcription = decoder.decode(log_probs)
print("Transcription:", transcription)
