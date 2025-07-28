from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model ID from Hugging Face Hub
model_id = "google/gemma-3-1b-it"

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