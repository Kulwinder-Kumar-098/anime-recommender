# ===============================
# ANIME RECOMMENDER - MAIN APP
# ===============================
import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline import run_pipeline, load_pipeline, save_pipeline
from recommend import AnimeRecommender


def interactive_mode():
    """Run the recommender in interactive mode."""
    print("\n" + "="*50)
    print("ANIME RECOMMENDER SYSTEM")
    print("="*50)
    
    # Check if recommendation system exists
    model_path = "model/model.pkl"
    if not os.path.exists(model_path):
        print("\n⚠️  Recommendation system not prepared. Please prepare it first.")
        print("Run: python main.py --prepare <data_path>")
        return
    
    # Initialize recommender
    recommender = AnimeRecommender(model_path)
    
    print("\n✅ Model loaded successfully!")
    print("\nOptions:")
    print("1. Get recommendations by anime name")
    print("2. Search for anime")
    print("3. Get recommendations by genre/tags")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            anime_name = input("Enter anime name: ").strip()
            top_n = input("Number of recommendations (default=5): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 5
            
            recommendations = recommender.get_recommendations(anime_name, top_n)
            if recommendations:
                print(f"\n🎬 Recommendations for '{anime_name}':")
                for i, anime in enumerate(recommendations, 1):
                    print(f"   {i}. {anime}")
            else:
                print("No recommendations found.")
        
        elif choice == '2':
            query = input("Enter search query: ").strip()
            results = recommender.search_anime(query)
            if results:
                print(f"\n📺 Search results for '{query}':")
                for i, anime in enumerate(results[:10], 1):
                    print(f"   {i}. {anime}")
            else:
                print("No results found.")
        
        elif choice == '3':
            query = input("Enter genres/tags (e.g., 'action anime'): ").strip()
            recommendations = recommender.get_recommendations(query, top_n=5)
            if recommendations:
                print(f"\n🎬 Recommendations for '{query}':")
                for i, anime in enumerate(recommendations, 1):
                    print(f"   {i}. {anime}")
            else:
                print("No recommendations found.")
        
        elif choice == '4':
            print("\nThank you for using Anime Recommender!")
            break
        
        else:
            print("Invalid choice. Please try again.")


def prepare_mode(data_path):
    """Prepare the recommendation system with the provided data path.
    
    This builds the cosine similarity matrix and CountVectorizer
    for content-based anime recommendations.
    """
    print(f"\n🚀 Building recommendation system with data from: {data_path}")
    print("   Using: CountVectorizer + Cosine Similarity")
    
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        return
    
    try:
        df, similarity, cv, vectors = run_pipeline(data_path)
        
        # Save the prepared system
        model_path = "model/model.pkl"
        save_pipeline(df, similarity, cv, model_path)
        
        print("\n✅ Recommendation system ready!")
        print(f"   - Processed {len(df)} anime")
        print(f"   - Built similarity matrix: {similarity.shape}")
        print(f"   - Saved to: {model_path}")
    except Exception as e:
        print(f"\n❌ Error during preparation: {str(e)}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Anime Recommender System (Cosine Similarity + CountVectorizer)'
    )
    parser.add_argument(
        '--prepare',
        type=str,
        help='Prepare mode: build recommendation system from data folder containing anime.csv and rating.csv'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode (default)'
    )
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    if args.prepare:
        prepare_mode(args.prepare)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
