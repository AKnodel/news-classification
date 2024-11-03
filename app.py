import streamlit as st
from app.data_collection import fetch_news, preprocess_news_data, fetch_article_content
from app.category_classification import classify_news
from app.sentiment_analysis import sentiment_analysis
from app.ner_extraction import extract_entities
from app.utils import preprocess_text

# Streamlit App Setup
st.title("Real-time News Article Classification")

# Step 1: Topic Input
query = st.text_input("Enter the topic you want to search for", "technology")
fetch_button = st.button("Fetch News Articles")

# Initialize session state for news articles if not already set
if 'news_df' not in st.session_state or fetch_button:
    st.session_state.news_df = fetch_news(query=query)

if 'selected_article_content' not in st.session_state:
    st.session_state.selected_article_content = ""

# Step 2: Fetch and Display News Articles only if the button is clicked
if fetch_button:
    try:
        st.write("Fetched News Articles")
        st.dataframe(st.session_state.news_df[['title', 'url', 'content']])
    except Exception as e:
        st.error(f"Error fetching news articles: {str(e)}")

# Step 3: Select an Article
if st.session_state.news_df.empty:
    st.warning("No articles found. Please try a different topic.")
else:
    selected_article_title = st.selectbox("Select an article for analysis", st.session_state.news_df['title'])

    # Get the selected article's details
    selected_article = st.session_state.news_df[st.session_state.news_df['title'] == selected_article_title].iloc[0]
    article_url = selected_article['url']  # Get the URL of the selected article

    # Fetch content using the selected article's URL if it is not already fetched
    if article_url != st.session_state.get('selected_article_url', None):
        st.session_state.selected_article_content = fetch_article_content(article_url)
        st.session_state.selected_article_url = article_url  # Store the selected URL

    # Display the content of the selected article
    st.write("Original Article Content:")
    st.write(st.session_state.selected_article_content)
    
    # Step 4: Preprocess the Selected Article
    cleaned_article = preprocess_text(st.session_state.selected_article_content)
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
