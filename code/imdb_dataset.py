from datasets import load_dataset

# Load the dataset
dataset = load_dataset("imdb")

# Create label mappings
labels = dataset["train"].features["label"].names # ["neg", "pos"]
num_labels = len(labels)
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

print(f"Labels: {labels}")
print(f"label2id mapping: {label2id}")
print(f"id2label mapping: {id2label}")