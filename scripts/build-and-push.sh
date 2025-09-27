#!/bin/bash

# Build and push script for Hamravesh deployment
# Usage: ./scripts/build-and-push.sh [tag]

set -e

# Configuration
REGISTRY="registry.hamravesh.com"
USERNAME="${HAMRAVESH_USERNAME:-your-username}"
PROJECT_NAME="nlp-toolkit"
TAG="${1:-latest}"

IMAGE_NAME="${REGISTRY}/${USERNAME}/${PROJECT_NAME}:${TAG}"

echo "Building Docker image: ${IMAGE_NAME}"

# Build the Docker image
docker build -t "${IMAGE_NAME}" .

echo "Built image: ${IMAGE_NAME}"

# Login to Hamravesh registry
echo "Logging in to Hamravesh registry..."
docker login "${REGISTRY}"

# Push the image
echo "Pushing image to registry..."
docker push "${IMAGE_NAME}"

echo "Successfully pushed ${IMAGE_NAME}"

# Update deployment with new image
if [ -f "k8s/deployment.yaml" ]; then
    echo "Updating deployment with new image..."
    sed -i.bak "s|registry.hamravesh.com/your-username/nlp-toolkit:latest|${IMAGE_NAME}|g" k8s/deployment.yaml
    echo "Deployment updated. Apply with: kubectl apply -f k8s/deployment.yaml"
fi
