#!/bin/bash

# Deployment script for Hamravesh Kubernetes cluster
# Usage: ./scripts/deploy.sh [namespace]

set -e

NAMESPACE="${1:-default}"
PROJECT_NAME="nlp-toolkit"

echo "Deploying ${PROJECT_NAME} to namespace: ${NAMESPACE}"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if we can connect to the cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "Error: Cannot connect to Kubernetes cluster"
    echo "Please ensure your kubeconfig is properly configured"
    exit 1
fi

# Create namespace if it doesn't exist
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests
echo "Applying ConfigMap..."
kubectl apply -f k8s/configmap.yaml -n "${NAMESPACE}"

echo "Applying Deployment..."
kubectl apply -f k8s/deployment.yaml -n "${NAMESPACE}"

echo "Applying Service..."
kubectl apply -f k8s/service.yaml -n "${NAMESPACE}"

echo "Applying Ingress..."
kubectl apply -f k8s/ingress.yaml -n "${NAMESPACE}"

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/${PROJECT_NAME} -n "${NAMESPACE}" --timeout=300s

# Get service information
echo ""
echo "Deployment completed successfully!"
echo ""
echo "Service information:"
kubectl get service ${PROJECT_NAME}-service -n "${NAMESPACE}"

echo ""
echo "Ingress information:"
kubectl get ingress ${PROJECT_NAME}-ingress -n "${NAMESPACE}"

echo ""
echo "To check logs:"
echo "kubectl logs -f deployment/${PROJECT_NAME} -n ${NAMESPACE}"

echo ""
echo "To check pod status:"
echo "kubectl get pods -l app=${PROJECT_NAME} -n ${NAMESPACE}"
