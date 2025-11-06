
import sys
from app.recommender import BookRecommender

def test_recommender():
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    recommender = BookRecommender()
    recommender.load_and_clean_data()
    recommender.build_model()
    recommender.save_artifacts()
    
    test_books = ["The Hobbit", "1984"]
    
    for book in test_books:
        print(f"\nRecommendations for '{book}':")
        result = recommender.recommend(book, n=3)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"{i}. {rec['title']} by {rec['authors']} (Rating: {rec['average_rating']})")
    
    print("\nTest completed successfully")

if __name__ == "__main__":
    test_recommender()
