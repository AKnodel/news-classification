# app/sentiment_analysis.py

from transformers import pipeline

# Initialize sentiment analysis pipeline
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
except Exception as e:
    print(f"Error loading sentiment model: {e}. Ensure the model 'cardiffnlp/twitter-roberta-base-sentiment' is accessible.")

def sentiment_analysis(text):
    """
    Analyzes the sentiment of the given text using the RoBERTa sentiment analysis model.

    :param text: The input text for sentiment analysis
    :return: A sentiment label (Positive, Neutral, Negative) with its confidence score
    """
    try:
        result = sentiment_pipeline(text)
        sentiment_label = result[0]['label']
        confidence_score = result[0]['score']
        
        # Convert label to a readable format
        if sentiment_label == "LABEL_0":
            sentiment = "Negative"
        elif sentiment_label == "LABEL_1":
            sentiment = "Neutral"
        else:
            sentiment = "Positive"
        
        return f"{sentiment} (Confidence: {confidence_score:.2f})"
    
    except Exception as e:
        return f"Error in sentiment analysis: {e}"
