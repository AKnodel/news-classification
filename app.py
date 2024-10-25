import streamlit as st
from app.data_collection import fetch_news, preprocess_news_data
from app.category_classification import classify_news
from app.sentiment_analysis import sentiment_analysis
from app.ner_extraction import extract_entities
from app.utils import preprocess_text

# Streamlit App Setup
st.title("Real-time News Article Classification")

# Step 1: Topic Input
query = st.text_input("Enter the topic you want to search for", "technology")
fetch_button = st.button("Fetch News Articles")

if fetch_button:
    try:
        # Step 2: Fetch and Display News Articles
        news_df = fetch_news(query=query)
        st.write("Fetched News Articles")
        st.dataframe(news_df[['title', 'content']])
        
        # Step 3: Select an Article
        selected_article = st.selectbox("Select an article for analysis", news_df['title'])
        article_content = news_df[news_df['title'] == selected_article]['content'].values[0]
        st.write("Original Article Content:")
        st.write(article_content)
        
        # Step 4: Preprocess the Selected Article
        cleaned_article = preprocess_text(article_content)
        st.write("Cleaned Article Content:")
        st.write(cleaned_article)
        
        # Step 5: Perform Classification
        category = classify_news(cleaned_article)
        st.write(f"Predicted Category: **{category}**")
        
        # Step 6: Sentiment Analysis
        sentiment = sentiment_analysis(cleaned_article)
        st.write(f"Sentiment Analysis: **{sentiment}**")
        
        # Step 7: Named Entity Recognition (NER)
        entities = extract_entities(cleaned_article)
        st.write("Named Entities Extracted:")
        st.write(entities)
        
    except Exception as e:
        st.error(f"Error fetching or processing news: {str(e)}")
else:
    st.warning("Please enter a topic to fetch news articles.")
