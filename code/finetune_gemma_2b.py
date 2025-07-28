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

# Preprocess and Tokenize Data
def preprocess_function(examples):
    # Tokenize the text. `truncation=True` ensures that long texts are cut to the model's max length.
    return tokenizer(examples["text"], truncation=True)

# Apply the function to the whole dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Fine-tuning with the Trainer API 
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Get the class with the highest probability
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

# A data collator will dynamically pad the texts to the length of the longest one in a batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
save_and_eval_steps = 500 # Example: run every 500 steps

training_args = TrainingArguments(
    output_dir="./gemma-imdb-classifier",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    
    # --- Ensure evaluation is enabled and matches saving frequency ---
    do_eval=True,                       # Crucial: This enables evaluation
    save_steps=save_and_eval_steps,     # Set saving frequency
    eval_steps=save_and_eval_steps,     # Set evaluation frequency to match saving
    # ----------------------------------------------------------------
    
    load_best_model_at_end=True,        # This now works because evaluation is on
    push_to_hub=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start fine-tuning!
trainer.train()

# To save the final model
trainer.save_model("./final_model")