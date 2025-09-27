# Hamravesh Deployment Guide

This guide provides step-by-step instructions for deploying your NLP toolkit on Hamravesh infrastructure.

## Prerequisites

1. **Hamravesh Account**: Access to Hamravesh platform with container registry and Kubernetes cluster access
2. **Docker**: Installed and configured on your local machine
3. **kubectl**: Kubernetes command-line tool
4. **Git**: For version control and CI/CD integration

## Step 1: Prepare Your Project

### 1.1 Update Configuration Files

1. **Update the Dockerfile** (already created):
   - Located at `Dockerfile` in project root
   - Optimized for production with proper caching and security

2. **Update Kubernetes manifests**:
   - Replace `your-username` with your actual Hamravesh username
   - Replace `your-domain.com` with your desired domain
   - Update image references in `k8s/deployment.yaml`

### 1.2 Configure Environment Variables

Create a `.env` file for local development:
```bash
HAMRAVESH_USERNAME=your-actual-username
HAMRAVESH_TOKEN=your-registry-token
KUBECONFIG_BASE64=your-kubeconfig-base64-encoded
```

## Step 2: Build and Push Docker Image

### 2.1 Manual Build and Push

```bash
# Set your Hamravesh credentials
export HAMRAVESH_USERNAME="your-username"
export HAMRAVESH_TOKEN="your-registry-token"

# Build and push using the provided script
./scripts/build-and-push.sh v1.0.0
```

### 2.2 Manual Docker Commands

```bash
# Build the image
docker build -t registry.hamravesh.com/your-username/nlp-toolkit:latest .

# Login to Hamravesh registry
docker login registry.hamravesh.com

# Push the image
docker push registry.hamravesh.com/your-username/nlp-toolkit:latest
```

## Step 3: Deploy to Kubernetes

### 3.1 Configure kubectl

1. **Get your kubeconfig from Hamravesh panel**
2. **Set up kubectl**:
   ```bash
   # Save your kubeconfig
   mkdir -p ~/.kube
   cp your-kubeconfig-file ~/.kube/config
   
   # Test connection
   kubectl cluster-info
   ```

### 3.2 Deploy Using Scripts

```bash
# Deploy to default namespace
./scripts/deploy.sh

# Deploy to specific namespace
./scripts/deploy.sh nlp-production
```

### 3.3 Manual Deployment

```bash
# Apply all manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check deployment status
kubectl rollout status deployment/nlp-toolkit
```

## Step 4: Configure Public Access

### 4.1 Update Ingress Configuration

1. **Edit `k8s/ingress.yaml`**:
   - Replace `nlp-toolkit.your-domain.com` with your desired domain
   - Ensure SSL certificate configuration is correct

2. **Apply the ingress**:
   ```bash
   kubectl apply -f k8s/ingress.yaml
   ```

### 4.2 DNS Configuration

1. **Get the external IP**:
   ```bash
   kubectl get ingress nlp-toolkit-ingress
   ```

2. **Configure DNS**:
   - Point your domain to the external IP
   - Wait for DNS propagation (up to 24 hours)

## Step 5: Set Up CI/CD Pipeline

### 5.1 GitHub Actions Setup

1. **Add secrets to your GitHub repository**:
   - `HAMRAVESH_USERNAME`: Your Hamravesh username
   - `HAMRAVESH_TOKEN`: Your registry token
   - `KUBECONFIG`: Base64 encoded kubeconfig

2. **The CI/CD pipeline will automatically**:
   - Build and push images on code changes
   - Deploy to Kubernetes on main branch pushes

### 5.2 Manual CI/CD Setup

If not using GitHub Actions, set up your preferred CI/CD system:

1. **Configure build triggers** on code changes
2. **Set up registry authentication**
3. **Configure Kubernetes deployment**
4. **Set up monitoring and notifications**

## Step 6: Monitoring and Maintenance

### 6.1 Check Deployment Status

```bash
# Check pods
kubectl get pods -l app=nlp-toolkit

# Check services
kubectl get services

# Check ingress
kubectl get ingress

# View logs
kubectl logs -f deployment/nlp-toolkit
```

### 6.2 Scaling

```bash
# Scale up replicas
kubectl scale deployment nlp-toolkit --replicas=3

# Check resource usage
kubectl top pods -l app=nlp-toolkit
```

### 6.3 Updates

```bash
# Update image
kubectl set image deployment/nlp-toolkit nlp-toolkit=registry.hamravesh.com/your-username/nlp-toolkit:v1.1.0

# Rollback if needed
kubectl rollout undo deployment/nlp-toolkit
```

## Step 7: Security Considerations

### 7.1 Container Security

- Images are built with non-root user
- Minimal base image (python:3.11-slim)
- No unnecessary packages installed
- Health checks configured

### 7.2 Network Security

- Service uses ClusterIP (internal access only)
- Ingress provides HTTPS termination
- Resource limits configured

### 7.3 Secrets Management

Store sensitive data in Kubernetes secrets:
```bash
# Create secret for API keys
kubectl create secret generic nlp-secrets \
  --from-literal=api-key=your-api-key

# Reference in deployment
# Add to deployment.yaml:
# env:
# - name: API_KEY
#   valueFrom:
#     secretKeyRef:
#       name: nlp-secrets
#       key: api-key
```

## Troubleshooting

### Common Issues

1. **Image pull errors**:
   - Check registry credentials
   - Verify image exists in registry

2. **Pod startup failures**:
   - Check resource limits
   - Verify environment variables
   - Check logs: `kubectl logs deployment/nlp-toolkit`

3. **Ingress not working**:
   - Check DNS configuration
   - Verify ingress controller is running
   - Check SSL certificate status

4. **Performance issues**:
   - Monitor resource usage
   - Consider scaling up replicas
   - Check application logs for bottlenecks

### Useful Commands

```bash
# Get detailed pod information
kubectl describe pod <pod-name>

# Execute commands in pod
kubectl exec -it <pod-name> -- /bin/bash

# Port forward for local testing
kubectl port-forward service/nlp-toolkit-service 8080:80

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

## API Endpoints

Once deployed, your application will be available at:

- **Web Interface**: `https://your-domain.com/`
- **Health Check**: `https://your-domain.com/health`
- **API Endpoints**:
  - `POST /api/regex` - Text processing
  - `POST /api/tokenizers` - Tokenizer comparison
  - `POST /api/seq2seq` - Sequence-to-sequence processing
  - `POST /api/pos-ner` - POS tagging and NER

## Support

For issues specific to Hamravesh infrastructure:
1. Check Hamravesh documentation
2. Contact Hamravesh support
3. Review Kubernetes logs and events

For application-specific issues:
1. Check application logs
2. Verify configuration files
3. Test locally with Docker
