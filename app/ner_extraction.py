# app/ner_extraction.py

import pandas as pd
import spacy

# Load spaCy's small English model for NER
nlp = spacy.load('en_core_web_sm')

def load_data():
    """
    Load data from CSV files for NER extraction.

    Returns:
    - train_df (DataFrame): DataFrame with training data, including columns 'class_id', 'title', 'description'.
    """
    train_df = pd.read_csv('models/train.csv')
    test_df = pd.read_csv('models/test.csv')

    train_df.columns = ['class_id', 'title', 'description']
    test_df.columns = ['class_id', 'title', 'description']

    train_df['class_id'] = train_df['class_id'] - 1
    test_df['class_id'] = test_df['class_id'] - 1

    train_df = train_df.sample(n=1000, random_state=42)
    
    return train_df

def extract_entities(text):
    """
    Extract named entities from a given text (e.g., people, organizations, locations).

    Parameters:
    - text (str): The input text for which to extract entities.

    Returns:
    - List[Tuple[str, str]]: A list of tuples containing each entity and its corresponding label.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def main():
    train_df = load_data()

    for index, row in train_df.iterrows():
        description = row['description']
        
        entities = extract_entities(description)
        
        print(f"News Description: {description}")
        if entities:
            print("Extracted Entities:")
            for entity, label in entities:
                print(f"  - {entity} ({label})")
        else:
            print("No named entities found.")
        print("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()
