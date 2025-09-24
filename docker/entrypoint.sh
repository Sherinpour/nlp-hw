#!/usr/bin/env bash
set -e
# Optional: activate venv if you create one; here we use system pip inside image
exec "$@"
