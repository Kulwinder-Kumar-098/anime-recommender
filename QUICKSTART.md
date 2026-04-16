# Quick Start Guide - Anime Recommender System

## 📋 Project Overview

Your anime recommender system is now **100% complete and ready to use**! 

This is a **content-based recommendation system** that suggests similar anime using:
- **CountVectorizer**: Converts anime genres/tags to numerical vectors
- **Cosine Similarity**: Measures similarity between anime vectors

No traditional ML model training needed!

---

## 🚀 Quick Start (5 Minutes)

### Option 1: Test with Demo Data (Recommended for first-time users)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the demo (creates sample data and builds recommendation system)
python demo.py

# 3. Launch the interactive app
python main.py
```

### Option 2: Use Your Own Data

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your data folder with:
#    - anime.csv (columns: anime_id, name, genre, type, rating)
#    - rating.csv (columns: user_id, anime_id, rating)

# 3. Build the recommendation system
python main.py --prepare path/to/your/data

# 4. Launch the app
python main.py
```

---

## 📁 What's Included

```
anime-recommender/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── pipeline.py          # CountVectorizer + Cosine Similarity pipeline
│   └── recommend.py         # Recommendation engine
├── data/                    # Your dataset folder
├── model/                   # Similarity matrices and vectorizers
├── notebooks/               # Jupyter notebooks
├── main.py                  # Main application (✨ READY TO USE)
├── demo.py                  # Demo script with sample data
├── config.py                # Configuration settings
├── requirements.txt         # Dependencies
└── README.md                # Full documentation
```

---

## 🎮 Interactive Features

Once you run `python main.py`, you can:

1. **Get recommendations by anime name**
   - Input: "Naruto"
   - Output: Top 5 similar anime (by cosine similarity)

2. **Search for anime**
   - Input: "action"
   - Output: All anime with "action" in the name

3. **Get recommendations by genre/tags**
   - Input: "sci-fi thriller"
   - Output: Top 5 similar anime

---

## 💻 Python API Usage

Use the recommender in your own Python code:

```python
from src.recommend import AnimeRecommender

# Load the recommendation system
recommender = AnimeRecommender('model/model.pkl')

# Get recommendations based on cosine similarity
recommendations = recommender.get_recommendations('Naruto', top_n=5)
print(recommendations)

# Search for anime
results = recommender.search_anime('action')
print(results)

# Get anime information
info = recommender.get_anime_info('Death Note')
print(info)
```

---

## 📦 Dependencies

All dependencies in `requirements.txt`:
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (CountVectorizer & cosine similarity)
- nltk (text processing with stemming)

Install with: `pip install -r requirements.txt`

---

## 🔧 Configuration

Edit `config.py` to customize:
- Model path
- Number of features (MAX_FEATURES)
- Recommendation count
- Text processing options
- And more...

---

## 🔬 How It Works

```
Input (Anime Name/Query)
        ↓
   Preprocessing
   (Clean data, handle missing values)
        ↓
   Text Processing
   (Stemming, lowercase)
        ↓
   CountVectorizer
   (Convert text to numerical vectors)
        ↓
   Cosine Similarity
   (Calculate similarity scores between all anime pairs)
        ↓
   Output (Top N similar anime)
```

**Key Point**: No neural networks or complex ML models. Pure mathematical similarity using cosine distance!

---

## ✅ Files Status

- ✅ `src/pipeline.py` - CountVectorizer + Cosine Similarity pipeline
- ✅ `src/recommend.py` - Recommendation engine
- ✅ `main.py` - Interactive application
- ✅ `demo.py` - Demo with sample data
- ✅ `requirements.txt` - All dependencies
- ✅ `config.py` - Configuration file
- ✅ `.gitignore` - Git settings
- ✅ `README.md` - Full documentation

---

## 🎯 Next Steps

1. **Test the system**:
   ```bash
   python demo.py
   ```

2. **Run the app**:
   ```bash
   python main.py
   ```

3. **Use your own data**:
   - Replace sample data in `data/` folder
   - Build: `python main.py --prepare data`
   - Run: `python main.py`

4. **Customize**:
   - Edit `config.py` for settings
   - Modify `src/` files for advanced features

---

## 🐛 Troubleshooting

**Q: "Recommendation system not prepared" error**
- A: Run demo.py first or prepare with: `python main.py --prepare data`

**Q: Missing dependencies**
- A: Install with: `pip install -r requirements.txt`

**Q: No recommendations found**
- A: Check if anime name matches exactly (case-insensitive)

**Q: NLTK error**
- A: Download nltk data: `python -c "import nltk; nltk.download('punkt')"`

---

## 📞 Support

- See `README.md` for detailed documentation
- Check `config.py` for configuration options
- Run `demo.py` to test with sample data

---

## ✨ You're All Set!

Your **content-based anime recommender** (CountVectorizer + Cosine Similarity) is **production-ready**. 

**Start with**: `python demo.py` and then `python main.py`

Enjoy! 🎬
