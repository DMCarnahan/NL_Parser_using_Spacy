import os
import requests
import pandas as pd
from transformers import Trainer, TrainingArguments, pipeline, BertForMaskedLM, BertTokenizerFast, BertForTokenClassification, DataCollatorForTokenClassification
from datasets import load_dataset, load_metric, load_from_disk
import tensorflow as tf
from torch.utils.data import DataLoader

# Limit TensorFlow's memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the MatBERT model from the local directory
matbert_model_path = r"C:\Users\dmich\Desktop\ammonia-text-miner\MatBERT-synthesis-classifier\results"  # Adjust the path if necessary

# Load the tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained(matbert_model_path, do_lower_case=False)
model = BertForTokenClassification.from_pretrained(matbert_model_path, num_labels=7)  # Adjust num_labels as needed

# Load the dataset
dataset = load_dataset("conll2003")

# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding='max_length', max_length=128, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Save the tokenized dataset to disk
tokenized_datasets.save_to_disk("tokenized_datasets")

# Load the tokenized dataset from disk
tokenized_datasets = load_from_disk("tokenized_datasets")

# Set format for PyTorch
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define the data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Debugging: Check the shape of the batch
loader = DataLoader(
    tokenized_datasets["train"],
    batch_size=16,
    shuffle=True
)

batch = next(iter(loader))
print(batch['input_ids'].shape)  # Should be [16, max_length]
print(batch['labels'].shape)     # Should match [16, max_length]

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(matbert_model_path)

# Create the NER pipeline
ner_model = pipeline("ner", model=model, tokenizer=tokenizer)

# Function to extract specific parameters from the article abstract
def extract_parameters(abstract):
    ner_results = ner_model(abstract)
    print("NER Results:", ner_results)  # Debugging information
    catalyst_identity = "Unknown"
    overpotential = "Unknown"
    # Process the NER results to extract specific parameters
    for entity in ner_results:
        print("Entity:", entity)  # Debugging information
        if entity['entity_group'] == 'CATALYST':
            catalyst_identity = entity['word']
        elif entity['entity_group'] == 'OVERPOTENTIAL':
            overpotential = entity['word']
    return catalyst_identity, overpotential

# Clear unused variables to free up memory
del model, tokenizer, ner_model