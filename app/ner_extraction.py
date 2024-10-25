# app/ner_extraction.py

import spacy

# Load spaCy's small English model for NER
nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    """
    Extract named entities from a text (e.g., people, organizations, locations).
    
    Parameters:
    - text: The cleaned news article text
    
    Returns:
    - A list of entities with their labels (e.g., PERSON, ORG, GPE)
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities
