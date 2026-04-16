# Anime Recommender System

A **content-based recommendation system** for anime titles using **CountVectorizer** and **Cosine Similarity**. No traditional ML model training required!

## Features

- **Content-Based Recommendations**: Uses CountVectorizer to convert genres/tags to vectors, then applies cosine similarity
- **Text Processing**: Applies stemming and vectorization for better text analysis
- **Interactive CLI**: User-friendly command-line interface
- **Search Functionality**: Search for anime by name or partial matches
- **Model Persistence**: Save and load vectorizers and similarity matrices

## Project Structure

```
anime-recommender/
├── data/                 # Dataset files (anime.csv, rating.csv)
├── src/
│   ├── __init__.py      # Package initialization
│   ├── pipeline.py      # Data processing and model training
│   └── recommend.py     # Recommendation engine and utilities
├── notebooks/
│   └── anime_recomendation_system.ipynb  # Jupyter notebook for exploration
├── model/               # Trained model files
├── requirements.txt     # Python dependencies
├── main.py             # Main application entry point
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Installation

1. **Clone/Download the project**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**:
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

## Usage

### Training the System

Before using the recommender, you need to prepare it with your dataset:

```bash
python main.py --prepare <path_to_data_folder>
```

**Requirements for data folder**:
- `anime.csv` - Contains anime data with columns: `anime_id`, `name`, `genre`, `type`, `rating`
- `rating.csv` - Contains user ratings (optional)

Example:
```bash
python main.py --prepare ./data
```

This will:
1. Load and preprocess the data
2. Apply stemming to genres/tags
3. Create CountVectorizer from anime features
4. Build cosine similarity matrix
5. Save everything to `model/model.pkl`

### Interactive Mode

Once the model is trained, run the application:

```bash
python main.py
```

or explicitly:

```bash
python main.py --interactive
```

**Interactive Menu Options**:

1. **Get recommendations by anime name**
   - Enter the name of an anime you like
   - Get top N similar recommendations
   
2. **Search for anime**
   - Find anime by name or partial matches
   
3. **Get recommendations by genre/tags**
   - Search by genre or tag (e.g., "action", "sci-fi thriller")
   
4. **Exit**
   - Quit the application

### Programmatic Usage

You can also use the recommender as a Python library:

```python
from src.recommend import AnimeRecommender

# Initialize recommender
recommender = AnimeRecommender('model/model.pkl')

# Get recommendations
recommendations = recommender.get_recommendations('Naruto', top_n=5)
print(recommendations)

# Search for anime
results = recommender.search_anime('action')
print(results)

The system uses a **mathematical approach** without traditional ML models:

1. **Data Loading**: Reads anime and rating CSV files
2. **Preprocessing**: Cleans data, handles missing values, and creates feature tags
3. **Text Processing**: Applies Porter Stemming for text normalization
4. **Vectorization**: Converts anime genres/tags to numerical vectors using **CountVectorizer**
5. **Similarity Computation**: Calculates **cosine similarity** between all anime pairs
6. **Recommendation**: Returns anime with highest cosine similarity scores

**Key Point**: Uses pure mathematical similarity (cosine distance) - no neural networks or model training!

```
Anime 1: [Action, Adventure, Shounen]  →  Vector 1
                                            ↓
                                    Cosine Similarity
                                            ↓
Anime 2: [Action, Adventure, Shounen]  →  Vector 2
                                        Score: 0.95
```

1. **Data Loading**: Reads anime and rating CSV files
2. **Preprocessing**: Cleans data, handles missing values, and creates feature tags
3. **Text Processing**: Applies Porter Stemming for text normalization
4. *prepared system is saved as `model/model.pkl` which contains:
- Processed dataframe with anime data
- Cosine similarity matrix (pre-computed)
- CountVectorizer (fitted on anime features)
## Dependencies

- **pandas** (1.5.3) - Data manipulation
- **numpy** (1.24.3) - Numerical computing
- **scikit-learn** (1.2.2) - Machine learning and vectorization
- **nltk** (3.8.1) - Natural language processing

## Model Files

The trained model is saved as `model/model.pkl` which contains:
- Processed dataframe with anime data
- Cosine similarity matrix
- CountVectorizer for text transformation

## Example Output

```
==================================================
ANIME RECOMMENDER SYSTEM
==================================================

✅ Model loaded successfully!

Options:
1. Get recommendations by anime name
2. Search for anime
3. Get recommendations by genre/tags
4. Exit

Enter your choice (1-4): 1
Enter anime name: Naruto
Number of recommendations (default=5): 5

🎬 Recommendations for 'Naruto':
   1. Naruto Shippuden
   2. Bleach
   3. One Piece
   4. Fairy Tail
   5. My Hero Academia
```

## Notes

- The model requires sufficient data to generate meaningful recommendations
- Stemming helps normalize similar words (e.g., "running" → "run")
- Cosine similarity ranges from 0 to 1, where 1 means identical
- Performance depends on the quality and size of your dataset

## Future Enhancements

- Add collaborative filtering recommendations
- Implement user rating-based recommendations
- Add visualization of recommendation similarity
- Create a web UI (Flask/Django/Streamlit)
- Support for real-time model updates

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

