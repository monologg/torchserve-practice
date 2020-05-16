import torch
import numpy as np
from transformers import ElectraTokenizer, ElectraConfig
from src import ElectraForSequenceClassification


tokenizer = ElectraTokenizer.from_pretrained("koelectra-small-finetuned-sentiment")

# Tokenize input text
text = "이 영화는 정말 최고야"
output = tokenizer.encode_plus(
    text,
    return_tensors="pt"
)
print(output)

model = ElectraForSequenceClassification.from_pretrained("koelectra-small-finetuned-sentiment", torchscript=True)
model.eval()

traced_model = torch.jit.trace(
    model,
    [output["input_ids"], output["attention_mask"], output["token_type_ids"]]
)
torch.jit.save(traced_model, "traced_model.pt")


# Load model
loaded_model = torch.jit.load("traced_model.pt")
loaded_model.eval()
with torch.no_grad():
    outputs = loaded_model(output["input_ids"], output["attention_mask"], output["token_type_ids"])
print(outputs)
outputs = outputs[0].cpu()
print(outputs.size())

scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)
print(scores)
