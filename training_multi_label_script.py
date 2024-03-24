"""This script trains the refactored multi-label classification model on the dataset and evaluates it """

from multi_label_class import MultiTaskBertModel
from data_preparation import prepare_dataset
import torch
from transformers import AdamW, get_linear_schedule_with_warmup


def initialize_model(device):
    model = MultiTaskBertModel.from_pretrained("bert-base-cased")
    model.to(device)
    return model


def train_model(
    model, device, train_dataloader, val_dataloader, num_epochs, optimizer, scheduler
):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            # move the items to device
            for key, value in batch.items():
                batch[key] = value.to(device)

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
                for key, value in batch.items():
                    batch[key] = value.to(device)

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    evidence_labels=batch["evidence"],
                    suggestion_labels=batch["suggestion"],
                    connection_labels=batch["connection"],
                )
                total_val_loss += outputs["loss"].item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(
            f"Epoch {epoch}: Average Training Loss = {avg_train_loss:.4f}, Average Validation Loss = {avg_val_loss:.4f}"
        )

    return model


def evaluate_model(model, test_dataloader):
    model.eval()
    total_test_loss = 0
    metrics = {
        "evidence_acc": 0,
        "evidence_f1": 0,
        "evidence_auc": 0,
        "suggestion_acc": 0,
        "suggestion_f1": 0,
        "suggestion_auc": 0,
        "connection_acc": 0,
        "connection_f1": 0,
        "connection_auc": 0,
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
        metrics[metric] = value / len(test_dataloader)

    return total_test_loss / len(test_dataloader), metrics


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(device)
    # Load the dataset
    train_dataloader, val_dataloader, test_dataloader = prepare_dataset(
        "./data/test_data.json"
    )
    # Prepare optimizer and schedule (linear warm-up and decay)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    NUM_EPOCHS = 1
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps
    )
    trained_model = train_model(
        model,
        device,
        train_dataloader,
        val_dataloader,
        NUM_EPOCHS,
        optimizer,
        scheduler,
    )
    loss, metrics = evaluate_model(trained_model, test_dataloader)
    print(f"Test Loss: {loss}")
    print(f"Test Metrics: {metrics}")

    # Save the model
    model_save_path = "./trained_models/refactored_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
