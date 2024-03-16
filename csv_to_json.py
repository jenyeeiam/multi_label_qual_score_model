import csv
import json

# Define the input and output file names
input_csv_file = '/Users/personal/qual_score/data/complete_data_with_sentiment_word_count.csv'
output_json_file = '/Users/personal/qual_score/data/complete_data_with_sentiment_word_count.json'

# Function to preprocess text according to the specified rules
def preprocess_text(text):
    # Replace internal double quotes with two double quotes
    text = text.replace('"', '""')
    # Escape new lines
    text = text.replace('\n', '\\n')
    return text

# Initialize an empty list to hold the converted data
json_data = []

# Open the CSV file for reading with 'utf-8-sig' encoding to handle BOM
with open(input_csv_file, mode='r', encoding='utf-8-sig') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.DictReader(csv_file)

    # Iterate over the CSV rows
    for row in csv_reader:
        # Convert each field from string to integer
        row['evidence'] = int(row['evidence'])
        row['suggestion'] = int(row['suggestion'])
        row['connection'] = int(row['connection'])
        row['sentiment'] = float(row['sentiment_score'])
        row['word_count'] = int(row['word_count'])
        # Preprocess the 'text' field
        row['text'] = preprocess_text(row['text'])
        # Add the row to the list
        json_data.append({'text': f'"{row["text"]}"', 'evidence': row['evidence'], 'suggestion': row['suggestion'], 'connection': row['connection'], 'sentiment': row['sentiment'], 'word_count': row['word_count']})

# Open the JSON file for writing
with open(output_json_file, mode='w', encoding='utf-8') as json_file:
    # Write the data to the JSON file in the specified format, ensuring that
    # non-ASCII characters are output as-is and not as Unicode escape sequences.
    json.dump(json_data, json_file, ensure_ascii=False, indent=2)

print(f"CSV data has been converted to JSON and saved to {output_json_file}")