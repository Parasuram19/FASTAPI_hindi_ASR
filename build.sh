#!/bin/bash

# Hindi ASR Docker Build Script
set -e

echo "🚀 Building Hindi ASR Docker Image..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if required files exist
echo "📋 Checking required files..."

if [ ! -f "fastapi_hindi_asr.py" ]; then
    echo -e "${RED}❌ fastapi_hindi_asr.py not found!${NC}"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt not found!${NC}"
    exit 1
fi

if [ ! -d "models" ]; then
    echo -e "${RED}❌ models directory not found!${NC}"
    exit 1
fi

# Check for model files
MODEL_FILES=(
    "models/stt_hi_conformer_ctc_medium.nemo"
    "models/stt_hi_conformer_ctc_medium_kenlm.bin"
)

for file in "${MODEL_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${YELLOW}⚠️  Warning: $file not found${NC}"
    else
        echo -e "${GREEN}✅ Found: $file${NC}"
    fi
done

# Build options
IMAGE_NAME="hindi-asr-api"
TAG="latest"
BUILD_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag|-t)
            TAG="$2"
            shift 2
            ;;
        --name|-n)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --no-cache)
            BUILD_ARGS="--no-cache"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --tag TAG     Set image tag (default: latest)"
            echo "  -n, --name NAME   Set image name (default: hindi-asr-api)"
            echo "  --no-cache        Build without cache"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo -e "${GREEN}🏗️  Building image: ${FULL_IMAGE_NAME}${NC}"

# Build the Docker image
docker build ${BUILD_ARGS} -t "${FULL_IMAGE_NAME}" .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo ""
    echo "📊 Image Information:"
    docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo ""
    echo "🚀 To run the container:"
    echo "   docker run -p 8000:8000 ${FULL_IMAGE_NAME}"
    echo ""
    echo "🐳 Or use docker-compose:"
    echo "   docker-compose up -d"
    echo ""
    echo "🔗 API will be available at: http://localhost:8000"
    echo "📚 API Documentation: http://localhost:8000/docs"
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi