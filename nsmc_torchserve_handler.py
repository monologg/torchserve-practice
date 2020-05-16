from abc import ABC
import json
import logging
import os

import torch
from transformers import ElectraTokenizer
from src import ElectraForSequenceClassification

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

        self.model = ElectraForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = ElectraTokenizer.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        logger.info("Model from path {} loaded successfully".format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode("utf-8")
        logger.info("Received text: '{}'".format(sentences))

        inputs = self.tokenizer.encode_plus(
            sentences,
            add_special_tokens=True,
            return_tensors="pt"
        )
        return inputs

    def inference(self, data):
        batch = tuple(t.to(self.device) for t in data)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2]
        }
        prediction = self.model(**inputs)[0].cpu().argmax().item()

        logger.info("Model predicted: '%s'", prediction)

        if self.mapping:
            prediction = self.mapping[str(prediction)]

        return [prediction]

    def postprocess(self, data):
        return data

    def handle(self, data, context):
        model_input = self.preprocess(data)
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
