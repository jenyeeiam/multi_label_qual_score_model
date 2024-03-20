from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from torch.utils.data import DataLoader, random_split
import torch

from multi_label_class import MultiTaskBertModel  # Import the model class
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn

from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

dataset_path = './data/complete_data_with_sentiment_word_count.json'
raw_dataset = load_dataset('json', data_files=dataset_path)['train']
data = [dict(text=entry['text'], evidence=int(entry['evidence']), suggestion=int(entry['suggestion']),
             connection=int(entry['connection']), sentiment=float(entry['sentiment']),
             word_count=int(entry['word_count'])) for entry in raw_dataset]

# Convert to a Pandas DataFrame and then to a Hugging Face Dataset
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Apply tokenization to the entire dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Now, convert the entire tokenized dataset to PyTorch tensors
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'evidence', 'suggestion', 'connection', 'sentiment', 'word_count'])
# Then split the dataset
torch.manual_seed(42)  # For reproducibility
train_size = int(0.7 * len(tokenized_dataset))
val_size = int(0.15 * len(tokenized_dataset))
test_size = len(tokenized_dataset) - train_size - val_size
# Use the split indices to create subsets
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(tokenized_dataset, [train_size, val_size, test_size])

def collate_fn(batch):
    # Collate the batch
    batch_dict = {key: [] for key in ['input_ids', 'attention_mask', 'evidence', 'suggestion', 'connection', 'sentiment', 'word_count']}
    for item in batch:
        for key in batch_dict.keys():
            batch_dict[key].append(item[key])
    
    # Convert lists to tensors (if not already tensors)
    for key in batch_dict:
        batch_dict[key] = torch.stack(batch_dict[key])

    # Move the entire batch to the device in one go (just before returning)
    return {key: value.to(device) for key, value in batch_dict.items()}


# Create data loaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = MultiTaskBertModel.from_pretrained("bert-base-cased")
model.to(device)  # Send model to device (CPU or GPU)

# Prepare optimizer and schedule (linear warm-up and decay)
optimizer = AdamW(model.parameters(), lr=5e-5)

NUM_EPOCHS = 3

# Calculate the total number of training steps
total_steps = len(train_dataloader) * NUM_EPOCHS
# Initialize the scheduler, 10% of total steps for warm-up
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

# Define loss functions for each task
loss_functions = {
    'evidence': nn.CrossEntropyLoss(),
    'suggestion': nn.CrossEntropyLoss(),
    'connection': nn.CrossEntropyLoss(),
}
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch}")
    model.train()
    total_train_loss = 0  # Accumulate loss over all batches for the epoch
    for batch in train_dataloader:
        optimizer.zero_grad()  # Zero gradients at the start

        # Unpack the batch and send to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        evidence = batch['evidence'].to(device)
        suggestion = batch['suggestion'].to(device)
        connection = batch['connection'].to(device)
        sentiment = batch['sentiment'].to(device)
        word_count = batch['word_count'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, sentiment=sentiment, word_count=word_count)
        
        # Calculate and aggregate loss for each task
        total_loss = 0
        for task, logits_task in outputs.items():
            labels_task = batch[task].to(device)
            loss = loss_functions[task](logits_task.view(-1, logits_task.size(-1)), labels_task.view(-1))
            total_loss += loss

        total_loss.backward()  # Backpropagation
        optimizer.step()
        # Step the learning rate scheduler
        scheduler.step()
        total_train_loss += total_loss.item()

    # Start validation phase
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():  # No need to track gradients
        print("Validation phase")
        for batch in val_dataloader:
            # Unpack the batch and send to device
            # Similar to the training phase, but without backward() and step()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            evidence = batch['evidence'].to(device)
            suggestion = batch['suggestion'].to(device)
            connection = batch['connection'].to(device)
            sentiment = batch['sentiment'].to(device)
            word_count = batch['word_count'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, sentiment=sentiment, word_count=word_count)

            for task, logits_task in outputs.items():
                labels_task = batch[task].to(device)
                loss = loss_functions[task](logits_task.view(-1, logits_task.size(-1)), labels_task.view(-1))
                total_val_loss += loss

    

    # Calculate average losses for the epoch
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_val_loss = total_val_loss / len(val_dataloader)
    

    print(f"Epoch {epoch}: Average Training Loss = {avg_train_loss:.4f}, Average Validation Loss = {avg_val_loss:.4f}")

# Initialize variables to track test loss and optionally other metrics
total_test_loss = 0

# Initialize counters for correct predictions and total predictions for each task
correct_predictions_evidence, correct_predictions_suggestion, correct_predictions_connection = 0, 0, 0
total_predictions_evidence, total_predictions_suggestion, total_predictions_connection = 0, 0, 0
# Initialize lists for true labels and predictions
evidence_true, suggestion_true, connection_true = [], [], []
evidence_pred_list, suggestion_pred_list, connection_pred_list = [], [], []

model.eval()  # Set the model to evaluation mode
print("Starting testing...")
with torch.no_grad():
    for batch in test_dataloader:
        # Unpack the batch and move to the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        evidence = batch['evidence'].to(device)
        suggestion = batch['suggestion'].to(device)
        connection = batch['connection'].to(device)
        sentiment = batch['sentiment'].to(device)
        word_count = batch['word_count'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, sentiment=sentiment, word_count=word_count)

        # Calculate loss for each task
        for task, logits_task in outputs.items():
            labels_task = batch[task].to(device)
            loss = loss_functions[task](logits_task.view(-1, logits_task.size(-1)), labels_task.view(-1))
            total_test_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits_task, dim=1)
            correct_predictions = (preds == labels_task).sum().item()
            total_predictions = labels_task.size(0)

            if task == 'evidence':
                evidence_true.extend(labels_task.cpu().numpy())
                evidence_pred_list.extend(preds.cpu().numpy())
                # accuracy
                correct_predictions_evidence += correct_predictions
                total_predictions_evidence += total_predictions
            elif task == 'suggestion':
                suggestion_true.extend(labels_task.cpu().numpy())
                suggestion_pred_list.extend(preds.cpu().numpy())
                correct_predictions_suggestion += correct_predictions
                total_predictions_suggestion += total_predictions
            elif task == 'connection':
                connection_true.extend(labels_task.cpu().numpy())
                connection_pred_list.extend(preds.cpu().numpy())
                correct_predictions_connection += correct_predictions
                total_predictions_connection += total_predictions
# Now, calculate precision, recall, and F1 score for each task
tasks = [
    ("Evidence", evidence_true, evidence_pred_list),
    ("Suggestion", suggestion_true, suggestion_pred_list),
    ("Connection", connection_true, connection_pred_list),
]
for task_name, true_labels, pred_labels in tasks:
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')

    print(f"{task_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
accuracy_evidence = correct_predictions_evidence / total_predictions_evidence
accuracy_suggestion = correct_predictions_suggestion / total_predictions_suggestion
accuracy_connection = correct_predictions_connection / total_predictions_connection
avg_test_loss = total_test_loss / len(test_dataloader)
print(f"Average Test Loss: {avg_test_loss:.4f}")
print(f"Evidence Accuracy: {accuracy_evidence:.4f}")
print(f"Suggestion Accuracy: {accuracy_suggestion:.4f}")
print(f"Connection Accuracy: {accuracy_connection:.4f}")

# After completing the training process, save the model's state dictionary
model_save_path = './sentiment_qual_score_model2.pth'  # Specify your save path here
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")