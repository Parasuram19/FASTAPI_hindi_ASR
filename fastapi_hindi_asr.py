"""
FastAPI Server for Hindi ASR with NVIDIA NeMo
Optimized for production use with async support and validation
"""

import os
import asyncio
import tempfile
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import torchaudio
import numpy as np

# Conditional imports with error handling
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("Warning: NeMo not available. Install with: pip install nemo_toolkit[all]")

try:
    from pyctcdecode import build_ctcdecoder
    PYCTCDECODE_AVAILABLE = True
except ImportError:
    PYCTCDECODE_AVAILABLE = False
    print("Warning: pyctcdecode not available. Install with: pip install pyctcdecode")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Response models
class TranscriptionResponse(BaseModel):
    """Response model for transcription"""
    transcription: str = Field(..., description="Transcribed text from audio")
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    processing_time_seconds: float = Field(..., description="Time taken to process")
    model_type: str = Field(..., description="Model type used (pytorch/onnx)")
    confidence_score: Optional[float] = Field(None, description="Confidence score if available")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether ASR model is loaded")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    dependencies_available: Dict[str, bool] = Field(..., description="Status of required dependencies")

class ServerInfoResponse(BaseModel):
    """Server information response"""
    message: str
    version: str
    docs: str
    health: str
    endpoints: Dict[str, str]
    supported_formats: List[str]
    max_file_size_mb: int
    max_duration_seconds: int

# Global ASR model instance
asr_model_instance = None
start_time = time.time()

class AsyncHindiASR:
    """Async-compatible Hindi ASR with thread pool execution"""
    
    def __init__(self, 
                 nemo_model_path: str,
                 kenlm_model_path: Optional[str] = None,
                 max_workers: int = 2):
        """
        Initialize the async ASR system
        
        Args:
            nemo_model_path: Path to .nemo model file
            kenlm_model_path: Path to KenLM language model (optional)
            max_workers: Maximum number of worker threads for inference
        """
        if not NEMO_AVAILABLE:
            raise RuntimeError("NeMo is not available. Please install nemo_toolkit[all]")
        
        self.nemo_model_path = nemo_model_path
        self.kenlm_model_path = kenlm_model_path
        self.max_workers = max_workers
        
        # Thread pool for CPU-bound inference tasks
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self.asr_model = None
        self.decoder = None
        self.vocab = None
        self.is_loaded = False
        
    async def initialize(self):
        """Initialize the model asynchronously"""
        if self.is_loaded:
            return
            
        logger.info("Initializing ASR model...")
        
        # Check if model file exists
        if not os.path.exists(self.nemo_model_path):
            raise FileNotFoundError(f"NeMo model not found at {self.nemo_model_path}")
        
        loop = asyncio.get_event_loop()
        
        try:
            # Load model in thread pool to avoid blocking
            await loop.run_in_executor(self.executor, self._load_model)
            await loop.run_in_executor(self.executor, self._setup_decoder)
            
            self.is_loaded = True
            logger.info("ASR model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ASR model: {e}")
            raise
    
    def _load_model(self):
        """Load the NeMo ASR model (blocking operation)"""
        try:
            self.asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
                self.nemo_model_path
            )
            self.asr_model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.asr_model = self.asr_model.cuda()
                logger.info("Model moved to GPU")
            else:
                logger.info("Using CPU for inference")
            
            self.vocab = self.asr_model.decoder.vocabulary
            logger.info(f"Model loaded with vocabulary size: {len(self.vocab)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load NeMo model: {e}")
    
    def _setup_decoder(self):
        """Setup the CTC decoder (blocking operation)"""
        try:
            if not PYCTCDECODE_AVAILABLE:
                logger.warning("pyctcdecode not available, using simple CTC decoding")
                self.decoder = None
                return
            
            if self.kenlm_model_path and os.path.exists(self.kenlm_model_path):
                self.decoder = build_ctcdecoder(
                    labels=self.vocab,
                    kenlm_model_path=self.kenlm_model_path
                )
                logger.info("KenLM language model loaded")
            else:
                self.decoder = build_ctcdecoder(labels=self.vocab)
                logger.info("Using CTC decoder without language model")
                
        except Exception as e:
            logger.warning(f"Decoder setup failed: {e}, using simple decoding")
            self.decoder = None
    
    def _preprocess_audio(self, audio_path: str) -> tuple:
        """Preprocess audio file (blocking operation)"""
        try:
            # Check file size to prevent memory issues
            file_size = os.path.getsize(audio_path)
            if file_size > 100 * 1024 * 1024:  # 100MB
                raise ValueError("Audio file too large for processing")
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=16000
                )
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Normalize audio
            waveform = waveform / torch.max(torch.abs(waveform))
            
            # Move to same device as model
            if torch.cuda.is_available() and next(self.asr_model.parameters()).is_cuda:
                waveform = waveform.cuda()
            
            duration = waveform.shape[1] / sample_rate
            return waveform, duration
            
        except Exception as e:
            raise ValueError(f"Audio preprocessing failed: {e}")
    
    def _run_inference(self, waveform: torch.Tensor) -> np.ndarray:
        """Run model inference (blocking operation)"""
        try:
            with torch.no_grad():
                log_probs = self.asr_model.forward(
                    input_signal=waveform,
                    input_signal_length=torch.tensor(
                        [waveform.shape[1]], 
                        device=waveform.device
                    )
                )
                return log_probs[0][0].cpu().numpy()
                
        except Exception as e:
            raise RuntimeError(f"Model inference failed: {e}")
    
    def _decode_predictions(self, log_probs: np.ndarray) -> str:
        """Decode predictions (blocking operation)"""
        try:
            if self.decoder and PYCTCDECODE_AVAILABLE:
                return self.decoder.decode(log_probs).strip()
            else:
                # Fallback to greedy decoding
                return self._greedy_decode(log_probs)
        except Exception as e:
            logger.warning(f"Decoder failed: {e}, using greedy decoding")
            return self._greedy_decode(log_probs)
    
    def _greedy_decode(self, log_probs: np.ndarray) -> str:
        """Simple greedy CTC decoding"""
        predictions = np.argmax(log_probs, axis=-1)
        decoded_chars = []
        prev_char = None
        
        for char_idx in predictions:
            if char_idx != 0 and char_idx != prev_char:  # Skip blank and repeated tokens
                if char_idx < len(self.vocab):
                    decoded_chars.append(self.vocab[char_idx])
            prev_char = char_idx
        
        return ''.join(decoded_chars).strip()
    
    async def transcribe_async(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file asynchronously
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        start_time = time.time()
        loop = asyncio.get_event_loop()
        
        try:
            # Run blocking operations in thread pool
            waveform, duration = await loop.run_in_executor(
                self.executor, self._preprocess_audio, audio_path
            )
            
            log_probs = await loop.run_in_executor(
                self.executor, self._run_inference, waveform
            )
            
            transcription = await loop.run_in_executor(
                self.executor, self._decode_predictions, log_probs
            )
            
            processing_time = time.time() - start_time
            
            return {
                "transcription": transcription,
                "duration_seconds": float(duration),
                "processing_time_seconds": float(processing_time),
                "model_type": "pytorch",
                "confidence_score": None  # Could be implemented if needed
            }
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)

# FastAPI app initialization
app = FastAPI(
    title="Hindi ASR API",
    description="Hindi Automatic Speech Recognition using NVIDIA NeMo",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CONFIG = {
    "NEMO_MODEL_PATH": os.getenv("NEMO_MODEL_PATH", "models/stt_hi_conformer_ctc_medium.nemo"),
    "KENLM_MODEL_PATH": os.getenv("KENLM_MODEL_PATH", "models/stt_hi_conformer_ctc_medium_kenlm.bin"),
    "MAX_FILE_SIZE": int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024)),  # 50MB
    "MAX_DURATION": float(os.getenv("MAX_DURATION", 60)),  # 60 seconds
    "MIN_DURATION": float(os.getenv("MIN_DURATION", 0.1)),  # 0.1 seconds
    "ALLOWED_FORMATS": {".wav", ".mp3", ".flac", ".m4a"},
    "MAX_WORKERS": int(os.getenv("MAX_WORKERS", 2))
}

def get_file_size(file: UploadFile) -> int:
    """Get file size by reading content"""
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    return size

def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file"""
    
    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in CONFIG["ALLOWED_FORMATS"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed: {', '.join(CONFIG['ALLOWED_FORMATS'])}"
        )
    
    # Check file size
    file_size = get_file_size(file)
    if file_size > CONFIG["MAX_FILE_SIZE"]:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size / (1024*1024):.1f}MB). Maximum size: {CONFIG['MAX_FILE_SIZE'] // (1024*1024)}MB"
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded"
        )

async def validate_audio_duration(temp_path: str) -> float:
    """Validate audio duration"""
    try:
        # Load just the metadata to check duration
        info = torchaudio.info(temp_path)
        duration = info.num_frames / info.sample_rate
        
        if duration < CONFIG["MIN_DURATION"]:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short ({duration:.2f}s). Minimum duration: {CONFIG['MIN_DURATION']}s"
            )
        
        if duration > CONFIG["MAX_DURATION"]:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long ({duration:.2f}s). Maximum duration: {CONFIG['MAX_DURATION']}s"
            )
        
        return duration
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio file: {str(e)}"
        )

async def cleanup_temp_file(file_path: str):
    """Background task to cleanup temporary files"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup temp file {file_path}: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the ASR model on startup"""
    global asr_model_instance
    
    logger.info("Starting Hindi ASR API server...")
    
    # Check dependencies
    if not NEMO_AVAILABLE:
        logger.error("NeMo not available. Please install with: pip install nemo_toolkit[all]")
        return
    
    # Check if model files exist
    if not os.path.exists(CONFIG["NEMO_MODEL_PATH"]):
        logger.error(f"NeMo model not found at {CONFIG['NEMO_MODEL_PATH']}")
        return
    
    try:
        asr_model_instance = AsyncHindiASR(
            nemo_model_path=CONFIG["NEMO_MODEL_PATH"],
            kenlm_model_path=CONFIG["KENLM_MODEL_PATH"] if os.path.exists(CONFIG["KENLM_MODEL_PATH"]) else None,
            max_workers=CONFIG["MAX_WORKERS"]
        )
        
        await asr_model_instance.initialize()
        logger.info("ASR model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}")
        asr_model_instance = None

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global asr_model_instance
    if asr_model_instance:
        asr_model_instance.cleanup()
        logger.info("ASR model cleanup completed")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if asr_model_instance and asr_model_instance.is_loaded else "unhealthy",
        model_loaded=asr_model_instance is not None and asr_model_instance.is_loaded,
        uptime_seconds=uptime,
        dependencies_available={
            "nemo": NEMO_AVAILABLE,
            "pyctcdecode": PYCTCDECODE_AVAILABLE,
            "torch": True,
            "torchaudio": True
        }
    )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file to transcribe (.wav, .mp3, .flac, .m4a)")
):
    """
    Transcribe audio file to text
    
    - **audio_file**: Audio file in supported format
    - Returns transcribed text with metadata
    """
    
    # Check if model is loaded
    if not asr_model_instance or not asr_model_instance.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="ASR model not loaded. Please check server logs and ensure model files are available."
        )
    
    # Validate file
    validate_audio_file(audio_file)
    
    # Create temporary file
    temp_path = None
    try:
        # Save uploaded file to temporary location
        suffix = Path(audio_file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Validate audio duration
        await validate_audio_duration(temp_path)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_path)
        
        # Transcribe audio
        try:
            result = await asr_model_instance.transcribe_async(temp_path)
            
            logger.info(f"Successfully transcribed audio: {len(result['transcription'])} characters, "
                       f"{result['duration_seconds']:.2f}s audio, "
                       f"{result['processing_time_seconds']:.2f}s processing")
            
            return TranscriptionResponse(
                transcription=result["transcription"],
                duration_seconds=result["duration_seconds"],
                processing_time_seconds=result["processing_time_seconds"],
                model_type=result["model_type"],
                confidence_score=result["confidence_score"]
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {str(e)}"
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/", response_model=ServerInfoResponse)
async def root():
    """Root endpoint with API information"""
    return ServerInfoResponse(
        message="Hindi ASR API",
        version="1.0.0",
        docs="/docs",
        health="/health",
        endpoints={
            "transcribe": "POST /transcribe - Upload audio file for transcription"
        },
        supported_formats=list(CONFIG["ALLOWED_FORMATS"]),
        max_file_size_mb=CONFIG["MAX_FILE_SIZE"] // (1024*1024),
        max_duration_seconds=int(CONFIG["MAX_DURATION"])
    )

if __name__ == "__main__":
    # Configuration for running directly
    uvicorn.run(
        "fastapi_hindi_asr:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        workers=1,     # Single worker due to model loading
        log_level="info"
    )