# app/data_collection.py

import requests
import pandas as pd
from bs4 import BeautifulSoup
from .utils import preprocess_text

def fetch_news(query="technology"):
    """
    Fetches news articles from multiple global and Indian news outlets.
    
    Parameters:
    - query: The topic to search for news articles.
    
    Returns:
    - A DataFrame containing news article details.
    """
    sources = {
        "BBC": f"https://www.bbc.co.uk/search?q={query}",
        "CNN": f"https://www.cnn.com/search?q={query}",
        "Reuters": f"https://www.reuters.com/search/news?blob={query}",
        "Times of India": f"https://timesofindia.indiatimes.com/search/news/{query}",
        "The Hindu": f"https://www.thehindu.com/search/?q={query}",
        "NDTV": f"https://www.ndtv.com/search?searchtext={query}"
    }

    articles = []

    for outlet, url in sources.items():
        print(f"Fetching articles from {outlet}...")
        try:
            # Send request to fetch news
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP request errors
            soup = BeautifulSoup(response.content, 'html.parser')

            # Define selectors based on the outlet structure
            if outlet == "BBC":
                for item in soup.select('.SearchResult'):
                    title = item.select_one('h1').get_text(strip=True)
                    content = item.select_one('p').get_text(strip=True)
                    articles.append({'title': title, 'content': content, 'source': outlet})
            elif outlet == "CNN":
                for item in soup.select('.cnn-search__result-headline'):
                    title = item.get_text(strip=True)
                    link = item.find('a')['href']
                    if not link.startswith('http'):
                        link = 'https://www.cnn.com' + link  # Handle relative links
                    content = fetch_article_content(link)  # Fetch full content from the link
                    articles.append({'title': title, 'content': content, 'source': outlet})
            elif outlet == "Reuters":
                for item in soup.select('.search-result'):
                    title = item.select_one('h3').get_text(strip=True)
                    link = item.find('a')['href']
                    if not link.startswith('http'):
                        link = 'https://www.reuters.com' + link  # Handle relative links
                    content = fetch_article_content(link)  # Fetch full content from the link
                    articles.append({'title': title, 'content': content, 'source': outlet})
            elif outlet == "Times of India":
                for item in soup.select('.searchResult'):
                    title = item.select_one('.title').get_text(strip=True)
                    content = item.select_one('.description').get_text(strip=True)
                    articles.append({'title': title, 'content': content, 'source': outlet})
            elif outlet == "The Hindu":
                for item in soup.select('.story-card'):
                    title = item.select_one('.story-card-title').get_text(strip=True)
                    content = item.select_one('.story-card-content').get_text(strip=True)
                    articles.append({'title': title, 'content': content, 'source': outlet})
            elif outlet == "NDTV":
                for item in soup.select('.search-result'):
                    title = item.select_one('.article-title').get_text(strip=True)
                    content = item.select_one('.article-summary').get_text(strip=True)
                    articles.append({'title': title, 'content': content, 'source': outlet})

        except Exception as e:
            print(f"Error fetching from {outlet}: {str(e)}")

    # Convert the articles to a DataFrame
    news_df = pd.DataFrame(articles)
    return news_df

def fetch_article_content(link):
    """
    Fetches the full content of an article given its link.
    
    Parameters:
    - link: The URL of the news article.
    
    Returns:
    - Full article content as a string.
    """
    try:
        response = requests.get(link)
        response.raise_for_status()  # Check for HTTP request errors
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract content based on typical structures (you may need to adjust based on the website)
        paragraphs = soup.select('p')  # This can vary, adjust selector if necessary
        content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        return content
    except Exception as e:
        print(f"Error fetching article content: {str(e)}")
        return ""

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
    return df[['title', 'cleaned_text']]  # Return only relevant columns (title and cleaned text)
