from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.data import DataLoader
import torch

from multi_label_class import MultiTaskBertModel  # Import the model class
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn

from datasets import load_dataset, Dataset
import pandas as pd
import json

dataset_path = './complete_data.json'
train_dataset = load_dataset('json', data_files=dataset_path)

converted_data = []
for feedback in train_dataset['train']:
  new_feedback = {}
  new_feedback['text'] = feedback['text']
  new_feedback['evidence'] = int(feedback['evidence'])
  new_feedback['suggestion'] = int(feedback['suggestion'])
  new_feedback['connection'] = int(feedback['connection'])
  converted_data.append(new_feedback)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels_evidence = torch.tensor([item['evidence'] for item in batch])
    labels_suggestion = torch.tensor([item['suggestion'] for item in batch])
    labels_connection = torch.tensor([item['connection'] for item in batch])

    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'labels_evidence': labels_evidence.to(device),
        'labels_suggestion': labels_suggestion.to(device),
        'labels_connection': labels_connection.to(device),
    }

# Tokenize the dataset
train_dataset['train'] = Dataset.from_pandas(pd.DataFrame(data=converted_data))
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_datasets = train_dataset['train'].map(tokenize_function, batched=True)
train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=16, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = MultiTaskBertModel.from_pretrained("bert-base-cased")
model.to(device)  # Send model to device (CPU or GPU)

# Prepare optimizer and schedule (linear warm-up and decay)
optimizer = AdamW(model.parameters(), lr=5e-5)

NUM_EPOCHS = 3

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        # Zero gradients at the start
        optimizer.zero_grad()
        # Unpack the batch and send to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_evidence = batch['labels_evidence'].to(device)
        labels_suggestion = batch['labels_suggestion'].to(device)
        labels_connection = batch['labels_connection'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits_evidence, logits_suggestion, logits_connection = outputs

        # Calculate loss for each task
        loss_fct = nn.CrossEntropyLoss()
        loss_evidence = loss_fct(logits_evidence.view(-1, 4), labels_evidence.view(-1))
        loss_suggestion = loss_fct(logits_suggestion.view(-1, 2), labels_suggestion.view(-1))
        loss_connection = loss_fct(logits_connection.view(-1, 2), labels_connection.view(-1))

        # Combine losses by summing the tensors directly
        total_loss = loss_evidence + loss_suggestion + loss_connection

        total_loss.backward()  # Correctly applies backward on a tensor
        optimizer.step()


    # Convert total_loss to a Python number AFTER the backward pass
    total_loss_num = total_loss.item()
    avg_loss = total_loss_num / len(train_dataloader)
    print(f"Epoch {epoch}: Average total loss {avg_loss}")
    print(f"Epoch {epoch}: Total loss {total_loss_num}")

# After completing the training process, save the model's state dictionary
model_save_path = './multi_label_qual_score_model.pth'  # Specify your save path here
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")