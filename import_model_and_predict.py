import torch
import pandas as pd
from multi_label_class import MultiTaskBertModel  # Import the model class
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer


def load_model(model_path, device):
    model = MultiTaskBertModel.from_pretrained("bert-base-cased")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(text, sentiment, word_count, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    sentiment_tensor = torch.tensor([sentiment], dtype=torch.float).unsqueeze(0).to(device)
    word_count_tensor = torch.tensor([word_count], dtype=torch.int).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # The model now returns a dictionary of logits
        logits = model(input_ids=input_ids, attention_mask=attention_mask, sentiment=sentiment_tensor, word_count=word_count_tensor)
        
        # Convert logits to probabilities for each task
        evidence_probs = torch.softmax(logits['evidence'], dim=1)
        suggestion_probs = torch.softmax(logits['suggestion'], dim=1)
        connection_probs = torch.softmax(logits['connection'], dim=1)
        
        # Use argmax to get the most likely class index for each task
        evidence_prediction = torch.argmax(evidence_probs, dim=1).item()
        suggestion_prediction = torch.argmax(suggestion_probs, dim=1).item()
        connection_prediction = torch.argmax(connection_probs, dim=1).item()
    
    return evidence_prediction, suggestion_prediction, connection_prediction


if __name__ == "__main__":
    print('creating device')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './sentiment_qual_score_model.pth'
    print('making tokenizer')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    print('loading model')
    model = load_model(model_path, device)

    # Load the CSV file
    input_csv_path = "./data/complete_data_with_sentiment_word_count.csv"  
    output_csv_path = "multi_label_predictions.csv"  
    df = pd.read_csv(input_csv_path)
    
    # Initialize columns for predictions
    df['evidence_pred'] = 0
    df['suggestion_pred'] = 0
    df['connection_pred'] = 0

    # Initialize lists for true labels and predictions
    evidence_true, suggestion_true, connection_true = [], [], []
    evidence_pred_list, suggestion_pred_list, connection_pred_list = [], [], []
    
    # Iterate over the DataFrame and make predictions
    for index, row in df.iterrows():
        text = row['text']
        sentiment = row['sentiment_score']
        word_count = row['word_count']

        evidence_pred, suggestion_pred, connection_pred = predict(text, sentiment, word_count, model, tokenizer, device)
         # Append predictions
        evidence_pred_list.append(evidence_pred)
        suggestion_pred_list.append(suggestion_pred)
        connection_pred_list.append(connection_pred)
        
        # Append true labels
        evidence_true.append(row['evidence_true'])
        suggestion_true.append(row['suggestion_true'])
        connection_true.append(row['connection_true'])
    
    # Save the DataFrame with predictions to a new CSV file
    # df.to_csv(output_csv_path, index=False)
    # print(f"Output saved to {output_csv_path}")
        
    # Calculate and print metrics for each task
    for task_name, true_labels, pred_labels in [
        ("Evidence", evidence_true, evidence_pred_list),
        ("Suggestion", suggestion_true, suggestion_pred_list),
        ("Connection", connection_true, connection_pred_list)
    ]:
        print(f"\nMetrics for {task_name}:")
        print(f"Precision: {precision_score(true_labels, pred_labels, average='macro'):.4f}")
        print(f"Recall: {recall_score(true_labels, pred_labels, average='macro'):.4f}")
        print(f"F1 Score: {f1_score(true_labels, pred_labels, average='macro'):.4f}")
    