#!/usr/bin/env bash
set -e
python - <<'PY'
import nltk, stanza, os
# NLTK
nltk.download('punkt', download_dir=os.environ.get('NLTK_DATA'))
nltk.download('averaged_perceptron_tagger', download_dir=os.environ.get('NLTK_DATA'))
# Stanza (fa)
stanza.download('fa', model_dir=os.environ.get('STANZA_RESOURCES_DIR'))
PY
