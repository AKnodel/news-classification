import pandas as pd
from .utils import preprocess_text
import http.client, urllib.parse
import json
import requests
from bs4 import BeautifulSoup

def fetch_article_content(url):
    """
    Fetches the full content of an article given its URL.
    
    Parameters:
    - url: The URL of the news article.
    
    Returns:
    - Full article content as a string or None if an error occurs.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP request errors
        soup = BeautifulSoup(response.content, 'html.parser')

        # Attempt to extract article content based on common patterns
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        return content.strip() if content else None

    except Exception as e:
        print(f"Error fetching article content from {url}: {str(e)}")
        return None
    
def fetch_news(query="technology", api_key="5f0SGwPZSqlYzbxClJVoVZ5cAi9q2ySZqwHOYwwm"):
    """
    Fetches news articles based on a topic using News API.
    
    Parameters:
    - query: The topic to search for news articles.
    - api_key: Your News API key.
    
    Returns:
    - A DataFrame containing news article details.
    """
    conn = http.client.HTTPSConnection('api.thenewsapi.com')
    params = urllib.parse.urlencode({
        'q': query,
        'api_token': api_key,
        'language': 'en',  # Fetch only English articles
        'sort': 'relevance'  # Sort by relevancy to the query
    })

    try:
        conn.request('GET', f'/v1/news/all?{params}')

        res = conn.getresponse()
        data = res.read()

        print(data.decode('utf-8'))
        articles_data = json.loads(data.decode('utf-8')).get('data', [])
        
        articles = [{
            'title': article['title'],
            'content': article['description'] or "",  # Use description for content if available
            'source': article['source'],
            'url': article['url']
        } for article in articles_data]

        news_df = pd.DataFrame(articles)
        return news_df

    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

# Preprocess the news articles
def preprocess_news_data(df):
    """
    Preprocesses a DataFrame containing news articles by cleaning text data.
    
    Parameters:
    - df: A DataFrame containing news articles fetched from the API.
    
    Returns:
    - A DataFrame with cleaned article text.
    """
    df['cleaned_text'] = df['content'].apply(lambda text: preprocess_text(text) if text else "")
    return df[['title','url', 'cleaned_text']]  # Return only relevant columns (title and cleaned text)
