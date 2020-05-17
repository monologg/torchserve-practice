# torchserve-practice

Torchserve with multiple nlp models

## Installation

- See [this guide](https://github.com/pytorch/serve#install-torchserve)

## How to run

```bash
$ curl -v -X PUT "http://localhost:8081/models/nsmc?min_worker=1"
```

```bash
$ curl -X POST http://127.0.0.1:8080/predictions/nsmc_small -T input_text.txt
```

## Reference

- [Torchserve Documentation](https://pytorch.org/serve/index.html)
- [Deploying huggingface‘s BERT to production with pytorch/serve](https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18)
