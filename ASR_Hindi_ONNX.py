"""
Simple Hindi ASR using NVIDIA NeMo with robust fallback
Prioritizes reliability over optimization
"""

import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import nemo.collections.asr as nemo_asr
from pyctcdecode import build_ctcdecoder


class SimpleHindiASR:
    """Simple and reliable Hindi ASR with fallback mechanisms"""
    
    def __init__(self, 
                 nemo_model_path: str,
                 kenlm_model_path: Optional[str] = None):
        """
        Initialize the ASR system
        
        Args:
            nemo_model_path: Path to .nemo model file
            kenlm_model_path: Path to KenLM language model (optional)
        """
        self.nemo_model_path = nemo_model_path
        self.kenlm_model_path = kenlm_model_path
        
        self.asr_model = None
        self.decoder = None
        self.vocab = None
        
        self._load_model()
        self._setup_decoder()
    
    def _load_model(self):
        """Load the NeMo ASR model"""
        print("Loading NeMo ASR model...")
        try:
            self.asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
                self.nemo_model_path
            )
            self.asr_model.eval()
            
            # Extract vocabulary
            self.vocab = self.asr_model.decoder.vocabulary
            print(f"Model loaded successfully with vocabulary size: {len(self.vocab)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load NeMo model: {e}")
    
    def _setup_decoder(self):
        """Setup the CTC decoder with optional KenLM"""
        print("Setting up CTC decoder...")
        
        try:
            if self.kenlm_model_path and os.path.exists(self.kenlm_model_path):
                # Use beam search with language model
                self.decoder = build_ctcdecoder(
                    labels=self.vocab,
                    kenlm_model_path=self.kenlm_model_path
                )
                print("KenLM language model loaded for beam search")
            else:
                # Use simple CTC decoder without language model
                self.decoder = build_ctcdecoder(labels=self.vocab)
                print("Using CTC decoder without language model")
                
        except Exception as e:
            print(f"Warning: Failed to setup advanced decoder: {e}")
            # Fallback to basic greedy decoding
            self.decoder = None
            print("Using basic greedy decoding as fallback")
    
    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, float]:
        """
        Preprocess audio file to required format
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (waveform, duration_seconds)
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=16000
                )
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Calculate duration
            duration = waveform.shape[1] / 16000
            
            # Validate duration (5-10 seconds recommended)
            if duration < 1:
                print(f"Warning: Audio duration {duration:.2f}s is very short")
            elif duration > 30:
                print(f"Warning: Audio duration {duration:.2f}s is quite long")
            
            print(f"Audio preprocessed: {duration:.2f}s, shape: {waveform.shape}")
            return waveform, duration
            
        except Exception as e:
            raise RuntimeError(f"Failed to preprocess audio: {e}")
    
    def greedy_decode(self, log_probs: np.ndarray) -> str:
        """
        Simple greedy CTC decoding as fallback
        
        Args:
            log_probs: Log probabilities from the model
            
        Returns:
            Decoded text
        """
        # Get the most likely character at each time step
        predictions = np.argmax(log_probs, axis=-1)
        
        # Remove consecutive duplicates and blank tokens (assuming blank is index 0)
        decoded_chars = []
        prev_char = None
        
        for char_idx in predictions:
            if char_idx != 0 and char_idx != prev_char:  # 0 is typically blank token
                if char_idx < len(self.vocab):
                    decoded_chars.append(self.vocab[char_idx])
            prev_char = char_idx
        
        # Join characters and clean up
        text = ''.join(decoded_chars)
        # Basic cleanup - remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Transcribing: {audio_path}")
        
        # Preprocess audio
        waveform, duration = self.preprocess_audio(audio_path)
        
        # Get predictions from model
        print("Running ASR inference...")
        try:
            with torch.no_grad():
                # Use keyword arguments as required by NeMo
                log_probs = self.asr_model.forward(
                    input_signal=waveform,
                    input_signal_length=torch.tensor([waveform.shape[1]])
                )
                log_probs = log_probs[0][0].cpu().numpy()
            
            print(f"Model output shape: {log_probs.shape}")
            
        except Exception as e:
            raise RuntimeError(f"ASR model inference failed: {e}")
        
        # Decode the predictions
        print("Decoding predictions...")
        try:
            if self.decoder is not None:
                # Use advanced CTC decoder
                transcription = self.decoder.decode(log_probs)
            else:
                # Use simple greedy decoding
                transcription = self.greedy_decode(log_probs)
            
            return transcription.strip()
            
        except Exception as e:
            print(f"Advanced decoding failed: {e}")
            print("Falling back to greedy decoding...")
            return self.greedy_decode(log_probs)


def main():
    """Example usage"""
    
    # Configuration
    NEMO_MODEL_PATH = "stt_hi_conformer_ctc_medium.nemo"
    KENLM_MODEL_PATH = "stt_hi_conformer_ctc_medium_kenlm.bin"
    AUDIO_PATH = "Suits LA Pilot Scene Leak NBC.mp3"
    
    print("="*60)
    print("SIMPLE HINDI ASR TRANSCRIPTION")
    print("="*60)
    
    try:
        # Initialize ASR system
        print("Initializing Hindi ASR system...")
        asr = SimpleHindiASR(
            nemo_model_path=NEMO_MODEL_PATH,
            kenlm_model_path=KENLM_MODEL_PATH if os.path.exists(KENLM_MODEL_PATH) else None
        )
        
        # Transcribe audio
        print(f"\nTranscribing audio file: {AUDIO_PATH}")
        transcription = asr.transcribe(AUDIO_PATH)
        
        print("\n" + "="*50)
        print("TRANSCRIPTION RESULT:")
        print("="*50)
        print(f'"{transcription}"')
        print("="*50)
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check your file paths.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your model files and audio format.")


if __name__ == "__main__":
    main()