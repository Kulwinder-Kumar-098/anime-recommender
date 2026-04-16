# ===============================
# DEMO SCRIPT
# ===============================
"""
Demo script to test the anime recommender with sample data.
This creates a small sample dataset and trains the model.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline import run_pipeline


def create_sample_data():
    """Create sample anime data for testing."""
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Sample anime data
    anime_data = {
        'anime_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'name': [
            'Naruto', 'Bleach', 'One Piece', 'Death Note', 'Attack on Titan',
            'Demon Slayer', 'My Hero Academia', 'Jujutsu Kaisen', 'Steins;Gate', 'Tokyo Ghoul'
        ],
        'genre': [
            'Action, Adventure', 'Action, Supernatural', 'Action, Adventure', 
            'Psychological, Thriller', 'Action, Dark Fantasy',
            'Action, Supernatural', 'Action, School', 'Action, Supernatural',
            'Sci-Fi, Thriller', 'Action, Dark Fantasy'
        ],
        'type': ['TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV', 'TV'],
        'rating': [8.3, 7.8, 8.9, 8.9, 8.9, 8.7, 7.7, 8.7, 9.0, 7.8]
    }
    
    anime_df = pd.DataFrame(anime_data)
    anime_df.to_csv('data/anime.csv', index=False)
    
    # Sample rating data
    rating_data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        'anime_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5],
        'rating': [10, 8, 9, 10, 9, 8, 7, 10, 10, 9, 8, 7, 9, 10, 8]
    }
    
    rating_df = pd.DataFrame(rating_data)
    rating_df.to_csv('data/rating.csv', index=False)
    
    print("✅ Sample data created successfully!")
    print("   - anime.csv (10 anime samples)")
    print("   - rating.csv (15 user ratings)")


def train_sample_model():
    """Build the recommendation system with sample data.
    
    Uses CountVectorizer + Cosine Similarity for recommendations.
    """
    print("\n🚀 Building recommendation system with sample data...")
    print("   Using: CountVectorizer + Cosine Similarity")
    
    try:
        df, similarity, cv, vectors = run_pipeline('data')
        
        print("\n✅ Recommendation system built!")
        print(f"   - Processed {len(df)} anime")
        print(f"   - Similarity matrix shape: {similarity.shape}")
        print(f"   - Saved to: model/model.pkl")
        
        # Show some sample recommendations
        print("\n📝 Sample Recommendations:")
        from pipeline import recommend
        
        sample_anime = ['Naruto', 'Death Note']
        for anime in sample_anime:
            recommendations = recommend(anime, df, similarity, cv, vectors, top_n=3)
            print(f"\n   Similar to '{anime}':")
            for i, rec in enumerate(recommendations, 1):
                print(f"      {i}. {rec}")
                
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")


def test_recommender():
    """Test the recommender system."""
    print("\n🧪 Testing recommender system...")
    
    try:
        from recommend import AnimeRecommender
        
        recommender = AnimeRecommender('model/model.pkl')
        
        # Test 1: Get recommendations
        print("\n   Test 1: Get recommendations for 'Attack on Titan'")
        recs = recommender.get_recommendations('Attack on Titan', top_n=3)
        print(f"   Results: {recs}")
        
        # Test 2: Search anime
        print("\n   Test 2: Search for 'action' anime")
        results = recommender.search_anime('action')
        print(f"   Found {len(results)} anime: {results[:3]}")
        
        # Test 3: Get anime info
        print("\n   Test 3: Get info about 'Naruto'")
        info = recommender.get_anime_info('Naruto')
        if info:
            print(f"   Info: {info}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")


def main():
    """Run the demo."""
    print("\n" + "="*60)
    print("ANIME RECOMMENDER - DEMO")
    print("(Content-Based using Cosine Similarity + CountVectorizer)")
    print("="*60)
    
    # Step 1: Create sample data
    print("\n📊 Step 1: Creating sample data...")
    create_sample_data()
    
    # Step 2: Build recommendation system
    print("\n📚 Step 2: Building recommendation system...")
    train_sample_model()
    
    # Step 3: Test recommender
    print("\n🧪 Step 3: Testing recommender...")
    test_recommender()
    
    print("\n" + "="*60)
    print("✨ Demo completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python main.py")
    print("2. Try the interactive mode")
    print("3. Or replace sample data with your own dataset in 'data/' folder")
    print("4. Build system: python main.py --prepare data")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
