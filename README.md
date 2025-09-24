Run each file in cli directory with these command. find corresponding.
```shell
python -m src.cli.run_regex --mode dates --text "$(cat data/raw/sample_texts/fa_examples.txt)"
python -m src.cli.run_regex --mode abbr  --text "$(cat data/raw/sample_texts/fa_examples.txt)"
python -m src.cli.run_regex --mode html  --text "$(cat data/raw/sample_texts/html_sample.html)"
python -m src.cli.run_regex --mode json  --text "$(cat data/raw/sample_texts/json_sample.json)"
python -m src.cli.run_tokenizers --cfg configs/tokenization.yml --compare
python -m src.cli.run_seq2seq \
    --prep_cfg configs/preprocess.yml \
    --model_cfg configs/model_seq2seq.yml

```
