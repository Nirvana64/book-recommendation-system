import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class BookRecommender:
    def __init__(self, data_path='data/goodbooks-10k/books.csv', models_dir='models'):
        self.data_path = data_path
        self.models_dir = models_dir
        self.books = None
        self.tfidf_vectorizer = None
        self.cosine_sim = None
        
    def load_and_clean_data(self):
        """Load books.csv and clean missing descriptions"""
        logger.info(f"Loading dataset from {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
        self.books = pd.read_csv(self.data_path, on_bad_lines='skip')
        
        required_columns = ['title', 'authors', 'average_rating']
        missing_columns = [col for col in required_columns if col not in self.books.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.books['content'] = self.books['title'].fillna('') + ' ' + self.books['authors'].fillna('')
        
        logger.info(f"Loaded {len(self.books)} books")
        return self.books
    
    def build_model(self):
        """Build TF-IDF vectorizer and compute cosine similarity"""
        if self.books is None:
            raise ValueError("No data loaded. Call load_and_clean_data() first")
            
        logger.info("Building TF-IDF model")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.books['content'])
        
        logger.info("Computing cosine similarity matrix")
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        logger.info("Model built successfully")
        
    def save_artifacts(self):
        """Save model artifacts to disk"""
        if any(x is None for x in [self.tfidf_vectorizer, self.cosine_sim, self.books]):
            raise ValueError("Model not built. Call build_model() first")
            
        os.makedirs(self.models_dir, exist_ok=True)
        
        joblib.dump(self.tfidf_vectorizer, f'{self.models_dir}/tfidf_vectorizer.pkl')
        joblib.dump(self.cosine_sim, f'{self.models_dir}/cosine_sim.pkl')
        self.books.to_csv(f'{self.models_dir}/books_cleaned.csv', index=False)
        
        logger.info(f"Model artifacts saved to {self.models_dir}")
    
    def load_artifacts(self):
        """Load pre-trained artifacts"""
        required_files = [
            f'{self.models_dir}/tfidf_vectorizer.pkl',
            f'{self.models_dir}/cosine_sim.pkl',
            f'{self.models_dir}/books_cleaned.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {missing_files}")
            
        self.tfidf_vectorizer = joblib.load(f'{self.models_dir}/tfidf_vectorizer.pkl')
        self.cosine_sim = joblib.load(f'{self.models_dir}/cosine_sim.pkl')
        self.books = pd.read_csv(f'{self.models_dir}/books_cleaned.csv')
        
        logger.info("Model artifacts loaded successfully")
    
    def recommend(self, title, n=10):
        """Get top N similar books based on title"""
        if self.books is None or self.cosine_sim is None:
            raise ValueError("Model not loaded. Call load_artifacts() or build_model() first")
            
        if not isinstance(title, str) or not title.strip():
            return {"error": "Invalid title provided"}
            
        if n < 1 or n > 50:
            return {"error": "Number of recommendations must be between 1 and 50"}
        
        idx = self.books[self.books['title'].str.lower() == title.lower().strip()].index
        
        if len(idx) == 0:
            return {"error": f"Book '{title}' not found in dataset"}
        
        idx = idx[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        
        book_indices = [i[0] for i in sim_scores]
        recommendations = self.books.iloc[book_indices][['title', 'authors', 'average_rating']].to_dict('records')
        
        return {
            "query": title,
            "recommendations": recommendations
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        recommender = BookRecommender()
        recommender.load_and_clean_data()
        recommender.build_model()
        recommender.save_artifacts()
        
        result = recommender.recommend("The Hobbit", n=5)
        logger.info(f"Test recommendation result: {result}")
        
    except Exception as e:
        logger.error(f"Failed to build model: {str(e)}")
        raise
