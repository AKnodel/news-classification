import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from huggingface_hub import login  # Import login function
from sklearn.preprocessing import LabelEncoder
import random

# Set your Hugging Face token here
HUGGINGFACE_TOKEN = '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'  # Replace with your actual token

# Set a seed for reproducibility
random.seed(42)

# Log in to Hugging Face
login(HUGGINGFACE_TOKEN)

# Load data from CSV
def load_data():
    train_df = pd.read_csv('models/train.csv')
    test_df = pd.read_csv('models/test.csv')
    
    # Rename columns for easier access
    train_df.columns = ['class_id', 'title', 'description']
    test_df.columns = ['class_id', 'title', 'description']
    
    # Convert class labels to start from 0
    train_df['class_id'] = train_df['class_id'] - 1
    test_df['class_id'] = test_df['class_id'] - 1

    # Randomly sample 1000 examples from train_df
    train_df = train_df.sample(n=1000, random_state=42)
    
    return train_df, test_df

# Custom Dataset class for PyTorch
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the input text
        inputs = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )

        inputs['labels'] = torch.tensor(label, dtype=torch.long)

        return {key: value.squeeze(0) for key, value in inputs.items()}

# Training function
def train_classifier(train_df):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)

    # Dataset and DataLoader
    train_dataset = NewsDataset(train_df['description'].values, train_df['class_id'].values, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=5e-5)

    model.train()

    # Training loop for epochs
    for epoch in range(1):
        print(f"Starting epoch {epoch+1}")
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Print progress for every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}")
        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss/len(train_loader)}")

    # Save the trained model
    model.save_pretrained('models/roberta_model')
    tokenizer.save_pretrained('models/roberta_model')

# Classification function to use the saved model
def classify_news(text):
    # Load the saved model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('models/roberta_model')
    model = RobertaForSequenceClassification.from_pretrained('models/roberta_model')
    model.eval()

    # Tokenize and encode the input text
    inputs = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

    # Perform the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=1).item()

    # Map the predicted class index back to a human-readable label
    label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    return label_map[predicted_class]

# Main function to load data and start training
def main():
    train_df, _ = load_data()
    train_classifier(train_df)

if __name__ == "__main__":
    main()
