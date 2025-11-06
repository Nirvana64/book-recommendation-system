from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
import os
import logging
from app.recommender import BookRecommender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Book Recommendation API",
    description="Content-based book recommendation system using Goodbooks-10k dataset",
    version="1.0.0"
)

# Global recommender instance
recommender = None

@app.on_event("startup")
async def startup_event():
    """Load or build model on startup"""
    global recommender
    recommender = BookRecommender()
    
    models_exist = all([
        os.path.exists('models/tfidf_vectorizer.pkl'),
        os.path.exists('models/cosine_sim.pkl'),
        os.path.exists('models/books_cleaned.csv')
    ])
    
    try:
        if models_exist:
            logger.info("Loading existing model artifacts")
            recommender.load_artifacts()
        else:
            logger.info("Building new model")
            if not os.path.exists('data/goodbooks-10k/books.csv'):
                logger.error("Dataset not found at data/goodbooks-10k/books.csv")
                raise FileNotFoundError("Dataset file missing")
            recommender.load_and_clean_data()
            recommender.build_model()
            recommender.save_artifacts()
            logger.info("Model built and saved successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {str(e)}")
        raise

class BookRecommendation(BaseModel):
    title: str
    authors: str
    average_rating: float

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[BookRecommendation]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Book Recommendation API",
        "endpoints": {
            "docs": "/docs",
            "recommend": "/recommend?title=The Hobbit&n=5",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    title: str = Query(..., description="Book title to get recommendations for"),
    n: int = Query(5, ge=1, le=20, description="Number of recommendations (1-20)")
):
    """Get book recommendations based on a title"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    
    result = recommender.recommend(title, n)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result
