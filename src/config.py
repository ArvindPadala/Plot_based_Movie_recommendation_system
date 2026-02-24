import os

# Base directory for the data and models
BASE_DATA_DIR = "/Users/arvindpadala/Documents/projects/Movie Recommendation System/Stat_Software_Project_demo"

# Paths to models
GENRE_MODEL_PATH = os.path.join(BASE_DATA_DIR, "ex_finetuned_distilbert_genre/fine_tuned_distilbert_model_final")
QA_MODEL_PATH = os.path.join(BASE_DATA_DIR, "ex_finetuned_distilbert_qa")

# Semantic matching model name
SEMANTIC_MODEL_NAME = 'all-MiniLM-L6-v2'

# Paths to data and index
FAISS_INDEX_PATH = os.path.join(BASE_DATA_DIR, "movie_plots_index.faiss")
TITLES_MAPPING_PATH = os.path.join(BASE_DATA_DIR, "titles_mapping.csv")
CONTEXT_DATA_PATH = os.path.join(BASE_DATA_DIR, "context.csv")

# Class labels mapping (from notebook)
GENRES = [
    'Action', 'Adventure', 'Animated', 'Comedy', 'Cult', 'Drama', 'Family', 
    'Historical/Documentary', 'Horror', 'International', 'Musical', 'Romance', 
    'Science Fiction', 'Short Film', 'Thriller'
]
