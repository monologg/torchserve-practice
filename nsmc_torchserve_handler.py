from abc import ABC
import json
import logging
import os

import torch
import numpy as np
from transformers import ElectraTokenizer
from model import ElectraForSequenceClassification

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class NSMCClassifierHandler(BaseHandler, ABC):
    def __init__(self):
        super(NSMCClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        logger.info("device: {}".format(self.device))

        self.model = ElectraForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = ElectraTokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.info("Model from path {} loaded successfully".format(model_dir))

        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        logger.info("Received text: '{}'".format(text))

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt"
        )
        return inputs

    def inference(self, data):
        with torch.no_grad():
            prediction = self.model(
                input_ids=data["input_ids"].to(self.device),
                attention_mask=data["attention_mask"].to(self.device),
                token_type_ids=data["token_type_ids"].to(self.device)
            )[0].cpu().numpy()

        score = np.exp(prediction) / np.exp(prediction).sum(-1, keepdims=True)

        result = {
            "label": self.model.config.id2label[score.argmax()],
            "score": score.max().item()
        }

        logger.info("Model predicted: '%s'", result)
        return [result]

    def postprocess(self, data):
        return data

    def handle(self, data, context):
        model_input = self.preprocess(data)
        logger.info(model_input)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)


_service = NSMCClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        return _service.handle(data, context)
    except Exception as e:
        raise e
