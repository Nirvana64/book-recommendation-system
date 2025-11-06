# Smart Book Recommendation System

A production-ready content-based book recommendation API built with FastAPI, scikit-learn, and Docker. Uses the Goodbooks-10k dataset to recommend similar books based on TF-IDF vectorization and cosine similarity.

## Features

- **Content-Based Filtering**: Uses TF-IDF and cosine similarity for recommendations
- **RESTful API**: FastAPI with automatic OpenAPI documentation
- **Dockerized**: Fully containerized for easy deployment
- **Production-Ready**: Health checks, error handling, and model persistence

## Architecture

```
book-recommender/
├── data/goodbooks-10k/     # Dataset directory
├── app/
│   ├── recommender.py      # ML recommendation engine
│   └── api.py              # FastAPI application
├── models/                 # Saved model artifacts
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerization)

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download `books.csv` from [Goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) and place it in:
```
data/goodbooks-10k/books.csv
```

### 3. Run Locally

```bash
uvicorn app.api:app --reload
```

The API will be available at: `http://localhost:8000`

### 4. Test the API

**Interactive Documentation:**
Visit `http://localhost:8000/docs` for Swagger UI

**Example API Call:**
```bash
curl "http://localhost:8000/recommend?title=The%20Hobbit&n=5"
```

**Example Response:**
```json
{
  "query": "The Hobbit",
  "recommendations": [
    {
      "title": "The Fellowship of the Ring",
      "authors": "J.R.R. Tolkien",
      "average_rating": 4.36
    },
    {
      "title": "The Two Towers",
      "authors": "J.R.R. Tolkien",
      "average_rating": 4.42
    }
  ]
}
```

## Docker Deployment

### Build Image

```bash
docker build -t book-recommender .
```

### Run Container

```bash
docker run -p 8000:8000 book-recommender
```

Access at: `http://localhost:8000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check |
| `/recommend` | GET | Get book recommendations |
| `/docs` | GET | Interactive API documentation |

### `/recommend` Parameters

- `title` (required): Book title to get recommendations for
- `n` (optional): Number of recommendations (1-20, default: 5)

## How It Works

1. **Data Processing**: Cleans and preprocesses book metadata
2. **TF-IDF Vectorization**: Converts book titles and authors into numerical vectors
3. **Cosine Similarity**: Computes similarity scores between all books
4. **Recommendation**: Returns top N most similar books

## Tech Stack

- **ML/Data**: pandas, numpy, scikit-learn
- **API**: FastAPI, Pydantic, Uvicorn
- **Serialization**: joblib
- **Containerization**: Docker

## Dataset

Uses the [Goodbooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k) containing:
- 10,000 books
- Metadata: titles, authors, ratings, descriptions
- User ratings and reviews

## Future Enhancements

- [ ] Collaborative filtering using user ratings
- [ ] Hybrid recommendation system
- [ ] Frontend UI with React
- [ ] Deploy to AWS/GCP
- [ ] Add caching with Redis
- [ ] A/B testing framework

## License

MIT License - feel free to use for your portfolio!

## Author

Built as a portfolio project demonstrating end-to-end ML engineering skills.
