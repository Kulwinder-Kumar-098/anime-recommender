"""
Anime Recommender Package

This package contains the core modules for the anime recommendation system.
"""

from pipeline import run_pipeline, load_pipeline, save_pipeline
from recommend import AnimeRecommender

__all__ = ['run_pipeline', 'load_pipeline', 'save_pipeline', 'AnimeRecommender']
