import streamlit as st
import torch
import pandas as pd
import faiss
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertForQuestionAnswering
from sentence_transformers import SentenceTransformer
from src import config

@st.cache_resource
def load_models_and_data():
    """
    Load and cache all models and datasets using Streamlit's caching
    so they are only loaded once.
    """
    with st.spinner("Loading models and datasets... This might take a minute."):
        # 1. Load Genre Classification Model
        genre_model = DistilBertForSequenceClassification.from_pretrained(config.GENRE_MODEL_PATH)
        genre_tokenizer = DistilBertTokenizer.from_pretrained(config.GENRE_MODEL_PATH)
        
        # 2. Load QA Model
        qa_model = DistilBertForQuestionAnswering.from_pretrained(config.QA_MODEL_PATH)
        qa_tokenizer = DistilBertTokenizer.from_pretrained(config.QA_MODEL_PATH)
        
        # 3. Load Semantic Model
        semantic_model = SentenceTransformer(config.SEMANTIC_MODEL_NAME)
        
        # 4. Load FAISS Index
        index = faiss.read_index(config.FAISS_INDEX_PATH)
        
        # 5. Load Datasets
        titles_df = pd.read_csv(config.TITLES_MAPPING_PATH)
        titles = titles_df["title"].tolist()
        
        context_df = pd.read_csv(config.CONTEXT_DATA_PATH)
        
        return {
            "genre_model": genre_model,
            "genre_tokenizer": genre_tokenizer,
            "qa_model": qa_model,
            "qa_tokenizer": qa_tokenizer,
            "semantic_model": semantic_model,
            "index": index,
            "titles": titles,
            "context_df": context_df
        }
