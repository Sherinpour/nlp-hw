#!/bin/bash

echo "=== تنظیمات اولیه برای Hamravesh ==="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker نصب نیست. لطفاً Docker را نصب کنید."
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl نصب نیست. لطفاً kubectl را نصب کنید."
    exit 1
fi

echo "✅ Docker و kubectl نصب هستند"
echo ""

# Get user input
read -p "نام کاربری Hamravesh خود را وارد کنید: " HAMRAVESH_USERNAME
read -p "نام پروژه (پیش‌فرض: nlp-toolkit): " PROJECT_NAME
PROJECT_NAME=${PROJECT_NAME:-nlp-toolkit}

echo ""
echo "=== تنظیمات شما ==="
echo "نام کاربری: $HAMRAVESH_USERNAME"
echo "نام پروژه: $PROJECT_NAME"
echo ""

# Update deployment files
echo "🔄 به‌روزرسانی فایل‌های deployment..."

# Update deployment.yaml
sed -i "s|registry.hamravesh.com/your-username/nlp-toolkit:latest|registry.hamravesh.com/$HAMRAVESH_USERNAME/$PROJECT_NAME:latest|g" k8s/deployment.yaml

# Update build script
sed -i "s|your-username|$HAMRAVESH_USERNAME|g" scripts/build-and-push.sh
sed -i "s|nlp-toolkit|$PROJECT_NAME|g" scripts/build-and-push.sh

echo "✅ فایل‌ها به‌روزرسانی شدند"
echo ""

echo "=== مراحل بعدی ==="
echo "1. وارد پنل Hamravesh شوید"
echo "2. Token مربوط به Container Registry را دریافت کنید"
echo "3. کانفیگ Kubernetes cluster را دانلود کنید"
echo "4. دستورات زیر را اجرا کنید:"
echo ""
echo "export HAMRAVESH_USERNAME='$HAMRAVESH_USERNAME'"
echo "export HAMRAVESH_TOKEN='your-registry-token'"
echo "./scripts/build-and-push.sh"
echo "./scripts/deploy.sh"
echo ""
echo "برای راهنمایی کامل، فایل DEPLOYMENT.md را مطالعه کنید."
