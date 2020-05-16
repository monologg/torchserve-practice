#!/bin/bash

torch-model-archiver --model-name nsmc --version 1.2 --serialized-file ./koelectra-small-finetuned-nsmc/pytorch_model.bin \
        --extra-files "./koelectra-small-finetuned-nsmc/config.json,./koelectra-small-finetuned-nsmc/tokenizer_config.json,./koelectra-small-finetuned-nsmc/vocab.txt,./nsmc_torchserve_handler.py" \
        --handler "./nsmc_torchserve_handler.py" \
        --export-path ./model_store \
        -f