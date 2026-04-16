"""
Configuration file for Anime Recommender System

This system uses:
- CountVectorizer: Convert anime genres/tags to numerical vectors
- Cosine Similarity: Calculate similarity between anime vectors
(No traditional ML model training)
"""

# Data Configuration
DATA_PATH = "data"
ANIME_CSV = "anime.csv"
RATING_CSV = "rating.csv"

# Model Configuration
MODEL_PATH = "model/model.pkl"
MODEL_BACKUP_PATH = "model/model_backup.pkl"

# Feature Engineering
MAX_FEATURES = 5000  # Maximum features for CountVectorizer
SIMILARITY_THRESHOLD = 0.1  # Minimum similarity threshold

# Recommendation
DEFAULT_TOP_N = 5  # Default number of recommendations
MAX_TOP_N = 20  # Maximum number of recommendations

# Text Processing
USE_STEMMING = True  # Enable/disable Porter Stemming
LOWERCASE = True  # Convert text to lowercase

# UI Configuration
SHOW_SIMILARITY_SCORES = False  # Show similarity scores in recommendations
VERBOSE_MODE = True  # Verbose output

# Data Validation
MIN_ANIME_COUNT = 100  # Minimum anime count to proceed
HANDLE_MISSING_VALUES = True  # Handle missing values
MISSING_VALUE_STRATEGY = 'mean'  # Strategy for handling missing values
