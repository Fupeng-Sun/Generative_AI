# Prepare the Dataset
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("imdb")

# Create label mappings
labels = dataset["train"].features["label"].names # ["neg", "pos"]
num_labels = len(labels)
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# Load Model and Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model ID from Hugging Face Hub
model_id = "google/gemma-2b-it"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Gemma doesn't have a default pad token, so we set it to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Load model with a classification head
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
# The model's pad token id must match the tokenizer's
model.config.pad_token_id = tokenizer.pad_token_id