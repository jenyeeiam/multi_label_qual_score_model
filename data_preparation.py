# TODO import all the things

# data_preparation.py
def prepare_dataset(dataset_path):
    # Load the dataset
    raw_dataset = load_dataset('json', data_files=dataset_path)['train']
    data = [dict(text=entry['text'], evidence=int(entry['evidence']), suggestion=int(entry['suggestion']),
                 connection=int(entry['connection'])) for entry in raw_dataset]
    # Convert to a Pandas DataFrame and then to a Hugging Face Dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Convert the entire tokenized dataset to PyTorch tensors
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'evidence', 'suggestion', 'connection'])
    
    # Split the dataset
    torch.manual_seed(42)  # For reproducibility
    train_size = int(0.7 * len(tokenized_dataset))
    val_size = int(0.15 * len(tokenized_dataset))
    test_size = len(tokenized_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(tokenized_dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    def collate_fn(batch):
        # Collate the batch
        batch_dict = {key: [] for key in ['input_ids', 'attention_mask', 'evidence', 'suggestion', 'connection']}
        for item in batch:
            for key in batch_dict.keys():
                batch_dict[key].append(item[key])
        
        # Convert lists to tensors (if not already tensors)
        for key in batch_dict:
            batch_dict[key] = torch.stack(batch_dict[key])
        
        return {key: value.to(device) for key, value in batch_dict.items()}
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)
    
    return train_dataloader, val_dataloader, test_dataloader
