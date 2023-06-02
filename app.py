import json
from nltk.tokenize import word_tokenize
from pytorch_pretrained_bert import cached_path

url = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

# Download dan load data JSON
personachat_file = cached_path(url)
with open(personachat_file, "r", encoding="utf-8") as f:
    dataset = json.loads(f.read())

# Tokenize NLTK tokenizer
def tokenize(obj):
    if isinstance(obj, str):
        return word_tokenize(obj)
    if isinstance(obj, dict):
        return {n: tokenize(o) for n, o in obj.items()}
    return [tokenize(o) for o in obj]

dataset = tokenize(dataset)


for conversation in dataset:
    for message in conversation["messages"]:
        role = message["role"]
        content = message["content"]
        print(f"Role: {role}")
        print(f"Content: {content}")
        print("-----")
