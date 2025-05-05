#search.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the SentenceTransformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# keyword-based search using TF-IDF
def perform_keyword_search(query: str, corpus: list, top_n: int = 3) -> list:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [query]) 
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1] 
    return [corpus[i] for i in top_indices]

# semantic search using Sentence Transformers
def perform_semantic_search(query: str, corpus: list, top_n: int = 3) -> list:
    corpus_embeddings = embedder.encode(corpus)  
    query_embedding = embedder.encode([query])[0]  
    similarities = cosine_similarity([query_embedding], corpus_embeddings).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]  
    return [corpus[i] for i in top_indices]

# hybrid search combining keyword and semantic search
def perform_hybrid_search(query: str, corpus: list, top_n: int = 3, alpha: float = 0.5) -> list:
    keyword_scores = get_keyword_scores(query, corpus)
    semantic_scores = get_semantic_scores(query, corpus)
    hybrid_scores = alpha * np.array(keyword_scores) + (1 - alpha) * np.array(semantic_scores)  
    top_indices = hybrid_scores.argsort()[-top_n:][::-1]  # Get the top N results
    return [corpus[i] for i in top_indices]

# keyword-based similarity scores (TF-IDF cosine similarity)
def get_keyword_scores(query, corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [query])  
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten() 
    return cosine_similarities.tolist()

# semantic-based similarity scores (using SentenceTransformer embeddings)
def get_semantic_scores(query, corpus):
    corpus_embeddings = embedder.encode(corpus) 
    query_embedding = embedder.encode([query])[0] 
    similarities = cosine_similarity([query_embedding], corpus_embeddings).flatten()  
    return similarities.tolist()
