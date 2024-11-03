import re
import spacy
from transformers import pipeline
from heapq import nlargest
from collections import defaultdict

# Load spaCy and transformers models
try:
    nlp = spacy.load('en_core_web_sm')
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except OSError as e:
    print("Required models not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    print("pip install transformers")

def clean_text(text):
    """Basic text cleaning"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text

def extractive_summarize(text, num_sentences=3):
    """
    Perform extractive summarization using sentence importance scoring
    """
    doc = nlp(text)
    word_frequencies = defaultdict(int)
    
    # Calculate word frequencies excluding stop words
    for word in doc:
        if not word.is_stop and not word.is_punct:
            word_frequencies[word.text.lower()] += 1
    
    # Normalize frequencies
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= max_frequency
    
    # Score sentences based on word frequencies
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    
    # Get top N sentences
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join([str(sent) for sent in summary_sentences])
    
    return summary

def abstractive_summarize(text, max_length=200, min_length=30):
    """
    Perform abstractive summarization using BART
    """
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error in abstractive summarization: {e}")
        return text

def summarize_article(text, method='both', num_sentences=3):
    """
    Main function to summarize text using either or both methods
    
    Parameters:
    - text: Input text to summarize
    - method: 'extractive', 'abstractive', or 'both'
    - num_sentences: Number of sentences for extractive summary
    
    Returns:
    - Dictionary containing requested summaries
    """
    result = {}
    
    if method in ['extractive', 'both']:
        result['extractive'] = extractive_summarize(text, num_sentences)
    
    if method in ['abstractive', 'both']:
        result['abstractive'] = abstractive_summarize(text)
    
    return result
