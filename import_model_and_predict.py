import torch
from multi_label_class import MultiTaskBertModel  # Import the model class
from transformers import BertConfig
from data_preparation import prepare_dataset  # Import the function to prepare the dataset
from transformers import AutoTokenizer

def initialize_model(device, model_path):
    # Initialize the model
    model = MultiTaskBertModel(BertConfig())
    model.load_state_dict(torch.load(model_path))
    # Ensure you switch the model to evaluation mode
    model.eval()
    model.to(device)
    return model

def get_metrics_on_test_split(device, model, test_dataloader):
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
    # Assuming no batching needed for compute_metrics directly
    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to the appropriate device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Get model outputs
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids", None),  # Optional, depending on your model's requirements
                evidence_labels=batch["evidence"],
                suggestion_labels=batch["suggestion"],
                connection_labels=batch["connection"],
            )

            # Compute metrics for this batch and aggregate results as needed
            batch_metrics = model.compute_metrics(
                outputs, batch["evidence"], batch["suggestion"], batch["connection"]
            )
            
            for key in metrics.keys():
                metrics[key] += batch_metrics[key]

            return metrics

    # Average the metrics over all batches
    num_batches = len(test_dataloader)
    for key in metrics.keys():
        metrics[key] /= num_batches

if __name__ == "__main__":
    # This path should be to your dataset file
    dataset_path = "./data/complete_data.json"
    # Assuming prepare_dataset returns DataLoader objects
    _, _, test_dataloader = prepare_dataset(dataset_path)

    model_path = './trained_models/multi_task_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = initialize_model(device, model_path)

    test_metrics = get_metrics_on_test_split(device, loaded_model, test_dataloader)
    print(test_metrics)