# app/utils.py

import re
import spacy

# Load spaCy's English language model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # Provide instructions if the model is not available
    print("Model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm' to install it.")

def preprocess_text(text):
    """
    Preprocess the input text by:
    - Lowercasing
    - Removing special characters
    - Tokenization
    - Removing stopwords
    - Lemmatization
    
    Parameters:
    - text: Raw text input
    
    Returns:
    - Preprocessed and cleaned text as a string
    """
    # Lowercase the text
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Process the text with spaCy if model is loaded
    if 'nlp' in globals():
        doc = nlp(text)
        cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        cleaned_text = ' '.join(cleaned_tokens)
    else:
        cleaned_text = text  # Fallback to simple lowercase text if spaCy model is missing

    return cleaned_text
