# app/__init__.py

from .data_collection import fetch_news, preprocess_news_data
from .category_classification import classify_news
from .sentiment_analysis import sentiment_analysis
from .ner_extraction import extract_entities
from .utils import preprocess_text
