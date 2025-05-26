import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio

def load_vocab(path="tokens.txt"):
    vocab = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            token, _ = line.strip().split()
            vocab.append(token)
    return vocab

def greedy_decode(logits, vocab):
    pred_ids = np.argmax(logits, axis=-1)
    blank_id = vocab.index("<blk>")
    prev_token = None
    decoded = []
    for idx in pred_ids:
        if idx != prev_token and idx != blank_id:
            decoded.append(vocab[idx])
        prev_token = idx
    return "".join(decoded)

def load_audio(audio_path, target_sample_rate=16000):
    audio, sr = sf.read(audio_path)
    if sr != target_sample_rate:
        audio = torchaudio.functional.resample(torch.tensor(audio), sr, target_sample_rate).numpy()
    return audio.astype(np.float32)

def preprocess(audio, sample_rate=16000):
    waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
    # Parameters similar to original model
    n_fft = 512
    win_length = 400  # 25ms window
    hop_length = 160  # 10ms stride
    n_mels = 64
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_spec = mel_spectrogram(waveform)  # (1, n_mels, time)
    # Convert power spectrogram (amplitude squared) to log scale (dB)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    # Transpose to (time, features)
    features = mel_spec_db.squeeze(0).transpose(0, 1).numpy()
    return features

def infer_onnx(audio_path, onnx_model_path="model.onnx", tokens_path="tokens.txt"):
    vocab = load_vocab(tokens_path)
    audio = load_audio(audio_path)
    features = preprocess(audio)
    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: features[np.newaxis, :, :]}  # Add batch dim
    logits = ort_session.run(None, ort_inputs)[0]
    logits = logits[0]
    transcription = greedy_decode(logits, vocab)
    return transcription

if __name__ == "__main__":
    audio_file = "suits-la-pilot-scene-leak-nbc_TxYi5gsM.mp3"
    onnx_path = "model.onnx"
    tokens_file = "tokens.txt"
    print("Running inference...")
    print(infer_onnx(audio_file, onnx_path, tokens_file))
