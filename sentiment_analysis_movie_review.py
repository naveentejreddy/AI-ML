import os
import re
import requests
import urllib.parse
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('wordnet')

# Constants
API_KEY = "dc4db47c22b6564a0dc9c092768d49f7"
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# ML Pipeline
classifier = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=10000)),
    ('logreg', LogisticRegression())
])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [LEMMATIZER.lemmatize(w) for w in words if w not in STOP_WORDS]
    return " ".join(words)

def load_imdb_data(path):
    data = []
    for sentiment in ['pos', 'neg']:
        sentiment_path = os.path.join(path, sentiment)
        for file in os.listdir(sentiment_path):
            with open(os.path.join(sentiment_path, file), 'r', encoding='utf-8') as f:
                text = f.read()
                data.append((clean_text(text), sentiment))
    return data

def train_model():
    path = "./aclImdb/train"
    data = load_imdb_data(path)
    df = pd.DataFrame(data, columns=['review', 'sentiment'])
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

def predict_sentiment(review):
    cleaned = clean_text(review)
    return classifier.predict([cleaned])[0]

def get_movie_id(movie_name):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": API_KEY, "query": movie_name}
    response = requests.get(url, params=params)
    data = response.json()
    return data['results'][0]['id'] if data['results'] else None

def get_movie_reviews(movie_name):
    movie_id = get_movie_id(movie_name)
    if not movie_id:
        return []
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews"
    params = {"api_key": API_KEY}
    response = requests.get(url, params=params)
    data = response.json()
    return [r['content'] for r in data['results']]

def get_imdb_review_url(movie_name):
    query = urllib.parse.quote_plus(movie_name)
    search_url = f"https://www.imdb.com/find?q={query}&s=tt"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    first_movie = soup.find("a", class_="ipc-metadata-list-summary-item__t")
    if first_movie and "/title/tt" in first_movie["href"]:
        movie_id = first_movie["href"].split("/")[2]
        return f"https://www.imdb.com/title/{movie_id}/reviews"
    return None

def scrape_imdb_reviews(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("article")
    reviews = []
    for article in articles:
        review_div = article.find("div", class_="ipc-html-content-inner-div")
        if review_div:
            reviews.append(review_div.get_text(strip=True))
    return reviews

def combine_reviews(movie_name):
    tmdb_reviews = get_movie_reviews(movie_name)
    imdb_url = get_imdb_review_url(movie_name)
    imdb_reviews = scrape_imdb_reviews(imdb_url) if imdb_url else []
    return tmdb_reviews + imdb_reviews

def movie_review_prediction(movie_name):
    reviews = combine_reviews(movie_name)
    if not reviews:
        return f"No reviews found for '{movie_name}'."
    
    total_reviews = len(reviews)
    pos_count = sum(1 for review in reviews if predict_sentiment(review) == 'pos')
    score = (pos_count / total_reviews) * 100
    
    if score < 30:
        verdict = 'Flop'
    elif score < 50:
        verdict = 'Average'
    elif score < 80:
        verdict = 'Hit'
    else:
        verdict = 'Super Hit'

    return (f"Movie: {movie_name}\n"
            f"Total Reviews: {total_reviews}\n"
            f"Positive Reviews: {pos_count} ({score:.2f}%)\n"
            f"Verdict: {verdict}")

if __name__ == "__main__":
    train_model()
    movie_name = input("Enter the movie name: ")
    result = movie_review_prediction(movie_name)
    print(result)
