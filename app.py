import streamlit as st
from src.models import load_models_and_data
from src.engine import classify_genre, find_top_k_matches, answer_qa

def main():
    st.set_page_config(
        page_title="Semantic Movie Recommender",
        page_icon="🎬",
        layout="centered"
    )

    st.title("🎬 Transformer-Powered Semantic Movie Recommendation System")
    st.markdown("""
        **Discover movies based on the story you want to see!**
        Unlike traditional recommendation systems that rely on ratings or viewing history, this system 
        uses advanced **Transformer models** to find movies based purely on their narrative plot.
    """)

    # --- 1. Check Data Availability ---
    import os
    from src import config
    if not os.path.exists(config.FAISS_INDEX_PATH) or not os.path.exists(config.GENRE_MODEL_PATH):
        st.error("🚨 **Error: Required models and datasets not found locally!**")
        st.write("Since the machine learning models and data exceed GitHub's file limits, they must be downloaded from Hugging Face.")
        st.info("### How to fix this:\nOpen your terminal, navigate to this project's folder, and run:\n```bash\npython download_data.py\n```\nAfter the download completes, refresh this page!")
        st.stop()

    # --- 2. Load Models ---
    try:
        models = load_models_and_data()
    except Exception as e:
        st.error(f"Error loading models or datasets: {e}")
        st.stop()

    st.divider()

    # --- 3. User Input ---
    st.subheader("What kind of movie are you looking for?")
    user_query = st.text_area(
        "Describe the plot or themes you want to watch:",
        placeholder="e.g., A hacker discovers a simulated reality controlled by machines.",
        height=100
    )

    if st.button("Find Movies", type="primary") and user_query:
        # Step 1: Predict Genre
        with st.spinner("Classifying genre..."):
            genre = classify_genre(user_query, models["genre_model"], models["genre_tokenizer"])
            st.success(f"**Predicted Genre:** {genre}")

        # Step 2: Semantic Search
        with st.spinner("Finding best matches..."):
            query_with_genre = f"{genre}: {user_query}"
            top_k_results = find_top_k_matches(
                query_with_genre, 
                models["semantic_model"], 
                models["index"], 
                models["titles"], 
                k=3
            )

        st.subheader("Top Movie Suggestions")
        st.session_state.recommended_movies = top_k_results
        
        for i, res in enumerate(top_k_results):
            st.markdown(f"**{i+1}. {res['title']}** *(Match Score: {res['score']:.4f})*")

    st.divider()

    # --- 3. Follow-up Q&A ---
    if "recommended_movies" in st.session_state and len(st.session_state.recommended_movies) > 0:
        st.subheader("Ask Questions About Recommended Movies")
        st.write("Want to know who starred in these movies or more about their plots? Ask here!")
        
        # Select one of the top recommendations to ask about
        movie_titles = [m["title"] for m in st.session_state.recommended_movies]
        selected_movie = st.selectbox("Select a movie to ask about:", movie_titles)
        
        follow_up_question = st.text_input(
            "Your Question:", 
            placeholder=f"e.g., What is the plot of {selected_movie}? Who are the main actors?"
        )
        
        if st.button("Ask") and follow_up_question:
            with st.spinner("Finding answer..."):
                # Get context for selected movie
                df = models["context_df"]
                try:
                    # In dataset 'context.csv', we need to match it with the correct title.
                    # Looking at Demo_NLP (1).ipynb: df2[df['title'] == movie_title]['context'].values[0]
                    # We have titles_mapping.csv for mapping titles, let's just do a string check or index check
                    
                    # Assuming we can find the plot in context_df:
                    # A better way is using the exact index from FAISS but the demo code does:
                    # df2 = pd.read_csv("context.csv") 
                    # movie_plot = df2[df['title'] == movie_title]['context'].values[0]
                    
                    # We will reproduce the demo logic:
                    # titles_df is not available here, so we reload or access it
                    # Let's read titles mapping locally in function to avoid storing massive data in state
                    import pandas as pd
                    from src import config
                    tdf = pd.read_csv(config.TITLES_MAPPING_PATH)
                    cdf = pd.read_csv(config.CONTEXT_DATA_PATH)
                    
                    movie_plot_array = cdf[tdf['title'] == selected_movie]['context'].values
                    
                    if len(movie_plot_array) > 0:
                        movie_plot = movie_plot_array[0]
                        answer = answer_qa(follow_up_question, movie_plot, models["qa_model"], models["qa_tokenizer"])
                        st.info(f"**Answer:** {answer}")
                    else:
                        st.warning(f"Could not find plot context for '{selected_movie}' in the dataset.")
                        
                except Exception as e:
                    st.error(f"Error executing QA: {e}")

if __name__ == "__main__":
    main()
