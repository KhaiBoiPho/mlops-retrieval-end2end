#!/bin/bash
# scripts/deploy_runpod.sh
# Build and deploy models to RunPod serverless

set -e  # Exit on error

# Configuration
DOCKER_REGISTRY="your-dockerhub-username"  # Change this!
BI_ENCODER_IMAGE="${DOCKER_REGISTRY}/legal-bi-encoder"
CROSS_ENCODER_IMAGE="${DOCKER_REGISTRY}/legal-cross-encoder"
VERSION="v1.0.0"

echo "========================================"
echo "RunPod Deployment Script"
echo "========================================"
echo ""

# Function to build and push image
build_and_push() {
    local service=$1
    local image=$2
    local dockerfile=$3
    
    echo "----------------------------------------"
    echo "Building $service..."
    echo "----------------------------------------"
    
    docker build \
        -f "$dockerfile" \
        -t "${image}:latest" \
        -t "${image}:${VERSION}" \
        .
    
    echo ""
    echo "Pushing $service to registry..."
    docker push "${image}:latest"
    docker push "${image}:${VERSION}"
    
    echo "✓ $service pushed successfully"
    echo ""
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    exit 1
fi

# Login to Docker Hub
echo "Logging in to Docker Hub..."
docker login

# Build and push Bi-Encoder
build_and_push \
    "Bi-Encoder" \
    "$BI_ENCODER_IMAGE" \
    "docker/bi-encoder/Dockerfile"

# Build and push Cross-Encoder
build_and_push \
    "Cross-Encoder" \
    "$CROSS_ENCODER_IMAGE" \
    "docker/cross-encoder/Dockerfile"

echo "========================================"
echo "✓ All images pushed successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Go to https://www.runpod.io/console/serverless"
echo "2. Create new template with these images:"
echo "   - Bi-Encoder: ${BI_ENCODER_IMAGE}:${VERSION}"
echo "   - Cross-Encoder: ${CROSS_ENCODER_IMAGE}:${VERSION}"
echo "3. Create endpoints and note the endpoint IDs"
echo "4. Update your .env file with endpoint URLs"
echo ""