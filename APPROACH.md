# Anime Recommender - Technical Approach

## What This System Does

This is a **content-based recommendation system** using **cosine similarity** and **CountVectorizer**. It does **NOT** use traditional machine learning model training.

---

## Technology Stack

### 1. **CountVectorizer** (scikit-learn)
- Converts anime genres/tags (text) into numerical vectors
- Creates a matrix where each row = an anime, each column = a genre/tag
- Each cell contains the count of how many times that tag appears for that anime

**Example**:
```
        Action  Adventure  Shounen  Sci-Fi
Naruto    1        1         1       0
Bleach    1        0         1       0
Steins;G  0        0         0       1
```

### 2. **Cosine Similarity** (scikit-learn)
- Measures the angle between two vectors
- Range: 0 (completely different) to 1 (identical)
- Formula: cos(θ) = (A · B) / (||A|| × ||B||)

**Example**:
```
Naruto vs Bleach:      similarity = 0.95 (very similar)
Naruto vs Steins;Gate: similarity = 0.10 (very different)
```

---

## How It Works Step-by-Step

### Step 1: Data Loading
```python
# Read anime.csv and rating.csv
anime_df = pd.read_csv('anime.csv')
rating_df = pd.read_csv('rating.csv')
```

### Step 2: Preprocessing
```python
# Clean missing values, create tags from genre + type
anime_df['tags'] = anime_df['genre'] + "," + anime_df['type']
# Result: "Action, Adventure" + "TV" = "Action, Adventure,TV"
```

### Step 3: Text Processing (Stemming)
```python
# Apply Porter Stemming to normalize words
"running" → "run"
"genres" → "genre"
# Better matching of similar concepts
```

### Step 4: Vectorization (CountVectorizer)
```python
# Convert text tags to numerical vectors
cv = CountVectorizer(max_features=5000)
vectors = cv.fit_transform(df['tags']).toarray()

# Result: Matrix of shape (num_anime, num_features)
# Example: (1000, 500) = 1000 anime, 500 unique tags
```

### Step 5: Cosine Similarity Computation
```python
# Calculate similarity between ALL anime pairs
similarity_matrix = cosine_similarity(vectors)

# Result: Matrix of shape (num_anime, num_anime)
# similarity_matrix[i][j] = similarity score between anime i and j
```

### Step 6: Get Recommendations
```python
# User asks: "Recommend anime similar to Naruto"
# Find Naruto's index, get its similarity row
similarities = similarity_matrix[naruto_index]
# Sort by highest similarity, skip first (itself)
top_5 = argsort(similarities)[-6:-1]  # indices of top 5
# Return anime names for those indices
```

---

## What Gets Saved

When you run `python main.py --prepare data`, three things are saved to `model/model.pkl`:

1. **DataFrame** (`df`)
   - The preprocessed anime data with original names and tags
   
2. **Cosine Similarity Matrix** (`similarity`)
   - Pre-computed similarity scores between all anime pairs
   - Shape: (num_anime, num_anime)
   - Pre-computed for fast recommendations
   
3. **CountVectorizer** (`cv`)
   - Fitted vectorizer for transforming new text input
   - Contains vocabulary of all tags seen during preparation
   - Used when user searches by genre/tags

---

## Key Differences from Traditional ML

| Aspect | This System | Traditional ML |
|--------|------------|---|
| **Model Training** | ❌ No | ✅ Yes |
| **Parameters to Learn** | ❌ No | ✅ Yes (weights, biases) |
| **Math Used** | Cosine Similarity | Neural Networks, SVM, etc. |
| **Time to Prepare** | Fast (seconds) | Slow (minutes/hours) |
| **Complexity** | Simple | Complex |
| **Interpretability** | 100% transparent | Often a black box |
| **Accuracy** | Content-based only | Can be better with more data |

---

## Recommendation Process

```
User Input: "Naruto"
    ↓
Search for "Naruto" in data
    ↓
Find Naruto's vector
    ↓
Get similarity scores with all anime
    ↓
Sort by highest similarity
    ↓
Return top N names
```

**Time Complexity**: O(n) where n = number of anime (very fast!)

---

## Example: How Cosine Similarity Works

**Anime A** (Naruto): [Action, Adventure, Shounen]
**Anime B** (Bleach): [Action, Adventure, Shounen]
**Anime C** (Steins;Gate): [Sci-Fi, Thriller]

**Vectors** (after CountVectorizer):
```
Naruto:       [1, 1, 1, 0, 0]
Bleach:       [1, 1, 1, 0, 0]
Steins;Gate:  [0, 0, 0, 1, 1]
```

**Cosine Similarity Scores**:
```
Naruto vs Bleach:       cos(0°)   = 1.00 (identical!)
Naruto vs Steins;Gate:  cos(90°)  = 0.00 (completely different)
```

---

## Why This Approach?

✅ **Advantages**:
- Simple and transparent
- No model training needed
- Fast recommendations (pre-computed similarities)
- Interpretable results
- Works well with limited data
- Easy to understand and modify

❌ **Limitations**:
- Only considers content (genres/tags)
- Doesn't learn user preferences
- No collaborative filtering
- Can't recommend novel/niche anime

---

## Customization

You can modify `config.py`:
- `MAX_FEATURES`: Number of unique tags/features to use
- `SIMILARITY_THRESHOLD`: Minimum similarity to include
- Text processing options in `pipeline.py`

---

## Summary

This is a **lightweight, transparent, content-based recommendation system** that:
1. ✅ Converts anime features to vectors (CountVectorizer)
2. ✅ Computes similarity scores (Cosine Similarity)
3. ✅ Recommends based on these scores

**No ML model training needed!**
