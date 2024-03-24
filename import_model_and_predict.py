import torch
import pandas as pd
from multi_label_class import MultiTaskBertModel  # Import the model class
from transformers import AutoTokenizer


def load_model(model_path, device):
    model = MultiTaskBertModel.from_pretrained("bert-base-cased")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(text, model, tokenizer, device):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        evidence_logits, suggestion_logits, connection_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Convert logits to probabilities
        evidence_probs = torch.softmax(evidence_logits, dim=1)
        suggestion_probs = torch.softmax(suggestion_logits, dim=1)
        connection_probs = torch.softmax(connection_logits, dim=1)

        # Use argmax to get the most likely class index
        evidence_prediction = torch.argmax(evidence_probs, dim=1).item()
        suggestion_prediction = torch.argmax(suggestion_probs, dim=1).item()
        connection_prediction = torch.argmax(connection_probs, dim=1).item()

    return evidence_prediction, suggestion_prediction, connection_prediction


if __name__ == "__main__":
    print("creating device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./multi_label_qual_score_model.pth"
    print("making tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    print("loading model")
    model = load_model(model_path, device)

    # Load the CSV file
    input_csv_path = "unlabeled_data.csv"
    output_csv_path = "multi_label_predictions.csv"
    df = pd.read_csv(input_csv_path)

    # Initialize columns for predictions
    df["evidence_pred"] = 0
    df["suggestion_pred"] = 0
    df["connection_pred"] = 0

    # Iterate over the DataFrame and make predictions
    for index, row in df.iterrows():
        text = row["Text"]
        evidence_pred, suggestion_pred, connection_pred = predict(
            text, model, tokenizer, device
        )
        df.at[index, "evidence_pred"] = evidence_pred
        df.at[index, "suggestion_pred"] = suggestion_pred
        df.at[index, "connection_pred"] = connection_pred

    # Save the DataFrame with predictions to a new CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Output saved to {output_csv_path}")
