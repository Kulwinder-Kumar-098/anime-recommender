# ===============================
# RECOMMENDATION UTILITIES
# ===============================
# Content-Based Recommender using Cosine Similarity
# No ML models - uses CountVectorizer + Cosine Similarity scores
# ===============================
import os
from pipeline import load_pipeline, recommend as pipeline_recommend


class AnimeRecommender:
    """Content-based anime recommender using cosine similarity."""
    
    def __init__(self, model_path="model/model.pkl"):
        """Initialize the recommender with prepared system data.
        
        Args:
            model_path (str): Path to the saved system pickle file
                            (contains: dataframe, similarity matrix, vectorizer)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"System not prepared at {model_path}. Please run --prepare first.")
        
        self.df, self.similarity, self.cv = load_pipeline(model_path)
        self.vectors = None
    
    def get_recommendations(self, anime_name, top_n=5):
        """Get anime recommendations based on input.
        
        Args:
            anime_name (str): Name of the anime or search query
            top_n (int): Number of recommendations to return
            
        Returns:
            list: List of recommended anime names
        """
        try:
            recommendations = pipeline_recommend(
                anime_name, 
                self.df, 
                self.similarity, 
                self.cv, 
                self.vectors, 
                top_n=top_n
            )
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []
    
    def search_anime(self, query):
        """Search for anime by name or partial match.
        
        Args:
            query (str): Search query
            
        Returns:
            list: List of matching anime
        """
        query_lower = query.lower()
        matches = self.df[self.df['name'].str.lower().str.contains(query_lower, na=False)]
        return matches['name'].tolist()
    
    def get_anime_info(self, anime_name):
        """Get information about a specific anime.
        
        Args:
            anime_name (str): Name of the anime
            
        Returns:
            dict: Anime information or None if not found
        """
        anime = self.df[self.df['name'].str.lower() == anime_name.lower()]
        if not anime.empty:
            return anime.iloc[0].to_dict()
        return None
