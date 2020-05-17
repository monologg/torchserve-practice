#!/bin/bash

mkdir -p model_store

torch-model-archiver --model-name "nsmc_small" --version 0.0.1 --serialized-file ./models/koelectra-small-finetuned-nsmc/pytorch_model.bin \
        --extra-files "./models/koelectra-small-finetuned-nsmc/config.json,./models/koelectra-small-finetuned-nsmc/tokenizer_config.json,./models/koelectra-small-finetuned-nsmc/vocab.txt,./nsmc_torchserve_handler.py,./model.py" \
        --handler "./nsmc_torchserve_handler.py" \
        --export-path ./model_store \
        -f

torch-model-archiver --model-name "nsmc_base" --version 0.0.1 --serialized-file ./models/koelectra-base-finetuned-nsmc/pytorch_model.bin \
        --extra-files "./models/koelectra-base-finetuned-nsmc/config.json,./models/koelectra-base-finetuned-nsmc/tokenizer_config.json,./models/koelectra-base-finetuned-nsmc/vocab.txt,./nsmc_torchserve_handler.py,./model.py" \
        --handler "./nsmc_torchserve_handler.py" \
        --export-path ./model_store \
        -f