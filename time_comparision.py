import time

import torch
from transformers import ElectraTokenizer
from src import ElectraForSequenceClassification

tokenizer = ElectraTokenizer.from_pretrained("koelectra-small-finetuned-sentiment")


# Load data
texts = [
    "이 영화는 미쳤다. 넷플릭스가 일상화된 시대에 극장이 존재해야하는 이유를 증명해준다.",
]

encoded_outputs = tokenizer.encode_plus(
    texts,
    return_tensors="pt"
)


# 2. Load torchscript model
jit_model = torch.jit.load("traced_model.pt")
jit_model.eval()

cur_time = time.time()
for i in range(20):
    outputs = jit_model(
        encoded_outputs["input_ids"],
        encoded_outputs["attention_mask"],
        encoded_outputs["token_type_ids"]
    )

print("torchscript model time: {:.2f}".format(time.time()-cur_time))

# 1. Load torch model
model = ElectraForSequenceClassification.from_pretrained("koelectra-small-finetuned-sentiment")
model.eval()

cur_time = time.time()
for i in range(20):
    outputs = model(
        encoded_outputs["input_ids"],
        encoded_outputs["attention_mask"],
        encoded_outputs["token_type_ids"]
    )

print("Simple model time: {:.2f}".format(time.time()-cur_time))
