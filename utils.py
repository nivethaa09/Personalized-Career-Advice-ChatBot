# utils.py
from typing import List, Dict, Any
import json
from sentence_transformers import SentenceTransformer
import numpy as np

import PyPDF2

def load_documents():
    """Load career documents from files."""
    documents = [
        {
            "id": "career-trends",
            "title": "Career Trends 2025",
            "content": extract_text_from_pdf("data/market_trends.pdf")
        },
        {
            "id": "resume-tips",
            "title": "Modern Resume Tips",
            "content": extract_text_from_pdf("data/resume_tips.pdf")
        },
        {
            "id": "career-advice",
            "title": "Career Development Strategies",
            "content": extract_text_from_pdf("data/career_advice.pdf")
        },
        {
            "id": "data-scientist",
            "title": "Data Scientist Career Roadmap",
            # Use UTF-8 encoding when reading the text file
            "content": open("data/Data Scientist Roadmap.txt", "r", encoding="utf-8").read()
        },
        {
            "id": "software-developer",
            "title": "Software Developer Career Path",
            # Use UTF-8 encoding when reading the text file
            "content": open("data/Software Developer Roadmap.txt", "r", encoding="utf-8").read()
        }
    ]
    return documents

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() if page.extract_text() else ""
            return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""


def perform_keyword_search(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Perform keyword-based search on documents."""
    results = []
    query_terms = query.lower().split()
    
    for doc in documents:
        score = 0
        content = doc["content"].lower()
        
        # Calculate simple term frequency score
        for term in query_terms:
            if term in content:
                score += content.count(term)
        
        if score > 0:
            # Find most relevant snippet
            paragraphs = doc["content"].split('\n\n')
            best_snippet = max(paragraphs, 
                             key=lambda p: sum(term in p.lower() for term in query_terms))
            
            results.append({
                "title": doc["title"],
                "snippet": best_snippet.strip(),
                "score": score / len(query_terms)
            })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)[:3]

def perform_semantic_search(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Perform semantic search using sentence transformers."""
    
    # Load model (first time will download it)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode query
    query_embedding = model.encode(query)
    
    results = []
    for doc in documents:
        # Split document into paragraphs and encode them
        paragraphs = doc["content"].split('\n\n')
        paragraph_embeddings = model.encode(paragraphs)
        
        # Calculate similarities
        similarities = np.dot(paragraph_embeddings, query_embedding)
        best_idx = np.argmax(similarities)
        
        results.append({
            "title": doc["title"],
            "snippet": paragraphs[best_idx].strip(),
            "score": float(similarities[best_idx])
        })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)[:3]

def perform_hybrid_search(query: str) -> List[Dict[str, Any]]:
    """Combine keyword and semantic search results."""
    documents = load_documents()
    
    keyword_results = perform_keyword_search(query, documents)
    semantic_results = perform_semantic_search(query, documents)
    
    # Combine and deduplicate results
    combined_results = {}
    
    # Process keyword results (40% weight)
    for result in keyword_results:
        combined_results[result["title"]] = {
            "title": result["title"],
            "snippet": result["snippet"],
            "score": result["score"] * 0.4
        }
    
    # Process semantic results (60% weight)
    for result in semantic_results:
        if result["title"] in combined_results:
            # Combine scores if result exists
            combined_results[result["title"]]["score"] += result["score"] * 0.6
            # Keep the better snippet
            if len(result["snippet"]) > len(combined_results[result["title"]]["snippet"]):
                combined_results[result["title"]]["snippet"] = result["snippet"]
        else:
            combined_results[result["title"]] = {
                "title": result["title"],
                "snippet": result["snippet"],
                "score": result["score"] * 0.6
            }
    
    return sorted(list(combined_results.values()), 
                 key=lambda x: x["score"], 
                 reverse=True)[:3]

def generate_follow_up_questions(topic: str, sources: List[Dict[str, Any]]) -> List[str]:
    """Generate follow-up questions based on the topic and sources."""
    questions = {
        "resume": [
            "Would you like specific tips for formatting your resume?",
            "Do you need help highlighting achievements?",
            "Should we discuss ATS optimization strategies?"
        ],
        "data science": [
            "Would you like to know about required skills for data science?",
            "Should we discuss data science certifications?",
            "Would you like to learn about career progression in data science?"
        ],
        "software development": [
            "Would you like to explore different specializations?",
            "Should we discuss in-demand programming languages?",
            "Would you like to know about portfolio projects?"
        ],
        "career trends": [
            "Would you like to focus on specific industry trends?",
            "Should we discuss future skill requirements?",
            "Would you like to learn about remote work opportunities?"
        ],
        "general": [
            "Would you like more specific advice for your situation?",
            "Should we discuss skill development strategies?",
            "Would you like to know about networking opportunities?"
        ]
    }
    
    # Select appropriate questions based on topic
    if "resume" in topic.lower():
        return questions["resume"]
    elif "data" in topic.lower():
        return questions["data science"]
    elif "software" in topic.lower() or "developer" in topic.lower():
        return questions["software development"]
    elif "trend" in topic.lower() or "market" in topic.lower():
        return questions["career trends"]
    else:
        return questions["general"]
