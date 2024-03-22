# training_script.py
from multi_label_class import MultiTaskBertModel
from data_preparation import prepare_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
train_dataloader, val_dataloader, test_dataloader = prepare_dataset('./data/complete_data.json')

# Initialize the model
model = MultiTaskBertModel.from_pretrained("bert-base-cased")
model.to(device)  # Send model to device (CPU or GPU)

# Prepare optimizer and schedule (linear warm-up and decay)
optimizer = AdamW(model.parameters(), lr=5e-5)
NUM_EPOCHS = 3
total_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch}")
    model.train()
    total_train_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            evidence_labels=batch["evidence"],
            suggestion_labels=batch["suggestion"],
            connection_labels=batch["connection"],
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                evidence_labels=batch["evidence"],
                suggestion_labels=batch["suggestion"],
                connection_labels=batch["connection"],
            )
            total_val_loss += outputs["loss"].item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Epoch {epoch}: Average Training Loss = {avg_train_loss:.4f}, Average Validation Loss = {avg_val_loss:.4f}")

# Evaluate on the test set
model.eval()
total_test_loss = 0
metrics = {
    "evidence_acc": 0, "evidence_f1": 0, "evidence_auc": 0,
    "suggestion_acc": 0, "suggestion_f1": 0, "suggestion_auc": 0,
    "connection_acc": 0, "connection_f1": 0, "connection_auc": 0,
}
with torch.no_grad():
    for batch in test_dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            evidence_labels=batch["evidence"],
            suggestion_labels=batch["suggestion"],
            connection_labels=batch["connection"],
        )
        total_test_loss += outputs["loss"].item()
        test_metrics = model.compute_metrics(
            outputs, batch["evidence"], batch["suggestion"], batch["connection"]
        )
        for metric, value in test_metrics.items():
            metrics[metric] += value

for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value / len(test_dataloader):.4f}")

print(f"Average Test Loss: {total_test_loss / len(test_dataloader):.4f}")

# Save the model
model_save_path = './trained_models/multi_task_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
