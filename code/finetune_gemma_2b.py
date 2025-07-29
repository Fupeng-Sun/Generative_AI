import numpy as np
import os
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# Optional: Disable wandb logging if you have network issues
os.environ["WANDB_DISABLED"] = "true"

# 1. Prepare the Dataset
# ========================
print("Loading and preparing dataset...")
dataset = load_dataset("imdb")

# Create label mappings required by the model
labels = dataset["train"].features["label"].names  # ["neg", "pos"]
num_labels = len(labels)
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# 2. Load Model and Tokenizer
# ===========================
print("Loading model and tokenizer...")
model_id = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Set a padding token if one doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
# Ensure the model's pad token ID is configured
model.config.pad_token_id = tokenizer.pad_token_id

# 3. Preprocess and Tokenize Data
# ===============================
print("Tokenizing dataset...")

def preprocess_function(examples):
    """Tokenize the text, truncating to a maximum length."""
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. Define Metrics and Training Configuration
# ============================================
def compute_metrics(eval_pred):
    """Computes accuracy metric for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Dynamically pads batches to the longest sequence
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./gemma-imdb-classifier",
    num_train_epochs=2,
    
    # 🚀 Memory and Speed Optimizations
    learning_rate=2e-5,
    per_device_train_batch_size=4,      # Reduced batch size to prevent OOM
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,      # Compensate for smaller per-device batch size
    fp16=True,                          # Enable mixed-precision for speed and memory savings
    
    # 📈 Logging and Saving Strategy
    evaluation_strategy="epoch",        # Evaluate at the end of each epoch
    save_strategy="epoch",              # Save at the end of each epoch
    load_best_model_at_end=True,        # Load the best model when training is complete
    
    # Other Parameters
    weight_decay=0.01,
    push_to_hub=False,
    report_to="none",                   # Explicitly disable all reporting integrations
)

# 5. Initialize and Run Trainer
# =============================
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
print("Starting training...")
trainer.train()

# Save the final model
print("Saving final model...")
trainer.save_model("./final_model")

print("Training complete!")