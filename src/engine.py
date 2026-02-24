import torch
from src import config

def classify_genre(user_query, genre_model, genre_tokenizer):
    """
    Tokenize and classify genre using the fine-tuned DistilBERT model.
    """
    inputs = genre_tokenizer(user_query, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = genre_model(**inputs)
        
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_genre = config.GENRES[predicted_class]
    return predicted_genre

def find_top_k_matches(user_query, semantic_model, index, titles, k=3):
    """
    Search in the FAISS index for the top-k matches using semantic embeddings.
    """
    # Generate embedding for the user query
    query_embedding = semantic_model.encode([user_query], convert_to_numpy=True, normalize_embeddings=True)
    
    # Search in the FAISS index for the top-k matches
    distances, indices = index.search(query_embedding, k=k)
    
    # Retrieve titles and scores for the top-k matches
    results = []
    for i in range(k):
        match_index = indices[0][i]
        match_score = distances[0][i]
        match_title = titles[match_index]
        results.append({"title": match_title, "score": float(match_score)})
        
    return results

def answer_qa(question, context, qa_model, qa_tokenizer):
    """
    Use DistilBERT to answer a follow-up question based on the movie plot.
    """
    inputs = qa_tokenizer(question, context, return_tensors="pt")
    
    with torch.no_grad():
        outputs = qa_model(**inputs)
        
    start_logits, end_logits = outputs.start_logits, outputs.end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)
    
    # Decode the answer
    answer_tokens = inputs['input_ids'][0][start_index:end_index+1]
    prediction = qa_tokenizer.decode(answer_tokens)
    
    return prediction.strip()
