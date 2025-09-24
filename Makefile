SHELL := /bin/bash

cpu-build:
\tdocker compose build app

cpu-shell:
\tdocker compose run --rm app bash

gpu-build:
\tdocker compose --profile gpu build app-gpu

gpu-shell:
\tdocker compose --profile gpu run --rm app-gpu bash

notebook:
\tdocker compose --profile notebooks up jupyter

down:
\tdocker compose down -v

warm-caches:
\tdocker compose run --rm app bash -lc "python docker/downloads.sh"

test:
\tdocker compose run --rm app bash -lc "pytest -q"

train:
\tdocker compose run --rm app bash -lc "python -m src.train.train_seq2seq --config configs/model_seq2seq.yml"
