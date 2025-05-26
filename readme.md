# Hindi ASR API

A production-ready FastAPI server for Hindi Automatic Speech Recognition using NVIDIA NeMo. This API provides real-time transcription of Hindi audio files with optimized performance for small audio files.

## Features

- **Async Processing**: Non-blocking audio transcription using thread pools
- **Multiple Audio Formats**: Support for WAV, MP3, FLAC, and M4A files
- **GPU Acceleration**: Automatic GPU utilization when available
- **Language Model Support**: Optional KenLM integration for improved accuracy
- **Production Ready**: Comprehensive error handling, validation, and logging
- **RESTful API**: Well-documented endpoints with OpenAPI/Swagger
- **Docker Support**: Containerized deployment ready

## Requirements

### System Dependencies
- Python 3.8+
- CUDA (optional, for GPU acceleration)
- FFmpeg (for audio processing)

### Python Dependencies
- FastAPI
- NVIDIA NeMo Toolkit
- PyTorch
- Torchaudio
- pyctcdecode (optional, for language model support)

## Quick Start

### 1. Using Docker (Recommended)

```bash
# Build the Docker image
docker build -t hindi-asr-api .

# Run the container
docker run -p 8000:8000 \
  -v /path/to/your/models:/app/models \
  -e NEMO_MODEL_PATH=/app/models/stt_hi_conformer_ctc_medium.nemo \
  hindi-asr-api
```

### 2. Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd hindi-asr-api

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export NEMO_MODEL_PATH="models/stt_hi_conformer_ctc_medium.nemo"
export KENLM_MODEL_PATH="models/stt_hi_conformer_ctc_medium_kenlm.bin"  # Optional

# Run the server
python fastapi_hindi_asr.py
```

## Model Setup

### Download NeMo Model

```bash
# Create models directory
mkdir -p models

# Download the Hindi NeMo model (example)
wget -O models/stt_hi_conformer_ctc_medium.nemo \
  "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_hi_conformer_ctc_medium/versions/1.10.0/files/stt_hi_conformer_ctc_medium.nemo"

# Optional: Download KenLM language model for improved accuracy
# wget -O models/stt_hi_conformer_ctc_medium_kenlm.bin \
#   "<language-model-url>"
```

## Configuration

Configure the API using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEMO_MODEL_PATH` | `models/stt_hi_conformer_ctc_medium.nemo` | Path to NeMo model file |
| `KENLM_MODEL_PATH` | `models/stt_hi_conformer_ctc_medium_kenlm.bin` | Path to KenLM language model (optional) |
| `MAX_FILE_SIZE` | `52428800` (50MB) | Maximum audio file size in bytes |
| `MAX_DURATION` | `60` | Maximum audio duration in seconds |
| `MIN_DURATION` | `0.1` | Minimum audio duration in seconds |
| `MAX_WORKERS` | `2` | Number of worker threads for inference |

## API Usage

### Health Check

```bash
curl -X GET http://localhost:8000/health
```

### Transcribe Audio

```bash
# Basic transcription
curl -X POST \
  -F "audio_file=@sample_hindi_audio.wav" \
  http://localhost:8000/transcribe

# Using Postman/Thunder Client
# POST http://localhost:8000/transcribe
# Body: form-data
# Key: audio_file, Type: File, Value: [select your audio file]
```

### Sample Response

```json
{
  "transcription": "नमस्ते, आप कैसे हैं?",
  "duration_seconds": 3.5,
  "processing_time_seconds": 0.8,
  "model_type": "pytorch",
  "confidence_score": null
}
```

## API Endpoints

### GET `/`
Root endpoint with API information and supported formats.

### GET `/health`
Health check endpoint showing model status and dependencies.

### POST `/transcribe`
Main transcription endpoint.

**Parameters:**
- `audio_file` (required): Audio file in supported format

**Supported Formats:**
- WAV (recommended)
- MP3
- FLAC  
- M4A

**Constraints:**
- Maximum file size: 50MB
- Maximum duration: 60 seconds
- Minimum duration: 0.1 seconds

### GET `/docs`
Interactive API documentation (Swagger UI).

## Performance Optimization

### For Small Audio Files (<10s)
- Optimized preprocessing pipeline
- Thread pool execution for non-blocking operations
- Efficient memory management

### For Production Deployment
```bash
# Using Gunicorn with multiple workers
gunicorn -w 1 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  fastapi_hindi_asr:app
```

**Note:** Use single worker (`-w 1`) due to model loading requirements.

## Docker Deployment

### Basic Deployment

```bash
# Build image
docker build -t hindi-asr-api .

# Run with volume mount for models
docker run -d \
  --name hindi-asr \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  hindi-asr-api
```

### GPU Support

```bash
# Run with GPU support
docker run -d \
  --name hindi-asr-gpu \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  hindi-asr-api
```

### Environment File (.env)

```env
NEMO_MODEL_PATH=/app/models/stt_hi_conformer_ctc_medium.nemo
KENLM_MODEL_PATH=/app/models/stt_hi_conformer_ctc_medium_kenlm.bin
MAX_FILE_SIZE=52428800
MAX_DURATION=60
MAX_WORKERS=2
```

## Testing

### Basic Functionality Test

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test transcription with sample file
curl -X POST \
  -F "audio_file=@test_audio.wav" \
  http://localhost:8000/transcribe \
  | jq '.'
```

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Simple load test (adjust concurrency based on your needs)
ab -n 10 -c 2 -T 'multipart/form-data' \
  -p test_audio.wav \
  http://localhost:8000/transcribe
```

## Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   Error: NeMo model not found at [path]
   ```
   - Ensure the model file exists at the specified path
   - Check file permissions
   - Verify the NEMO_MODEL_PATH environment variable

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   - Reduce MAX_WORKERS
   - Use CPU-only mode by setting CUDA_VISIBLE_DEVICES=""
   - Process smaller audio files

3. **Audio Format Issues**
   ```
   Invalid audio file: [format error]
   ```
   - Ensure audio file is not corrupted
   - Convert to WAV format using FFmpeg
   - Check audio file duration and size limits

### Logging

Logs are output to stdout/stderr. For production:

```bash
# Redirect logs to file
python fastapi_hindi_asr.py > app.log 2>&1 &

# Or use Docker logging
docker logs hindi-asr-api
```

## Development

### Local Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
uvicorn fastapi_hindi_asr:app --reload --host 0.0.0.0 --port 8000
```

### Code Structure

```
├── fastapi_hindi_asr.py    # Main application file
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── models/               # Model files directory
│   ├── stt_hi_conformer_ctc_medium.nemo
│   └── stt_hi_conformer_ctc_medium_kenlm.bin
└── README.md            # This file
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions:
- Check the troubleshooting section above
- Review server logs for detailed error messages
- Ensure all dependencies are properly installed