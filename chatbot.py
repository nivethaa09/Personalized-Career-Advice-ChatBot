from typing import List, Dict, Any
from utils import perform_hybrid_search, generate_follow_up_questions
import openai
from dotenv import load_dotenv
import os

#variables from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_response_with_rag(query: str, sources: List[Dict[str, Any]]) -> str:
    """Generate a response using the retrieved context and the user's query."""
    #the prompt by combining query and retrieved documents
    context = "\n".join([source["snippet"] for source in sources])
    prompt = f"User Query: {query}\n\nContext:\n{context}\n\nAnswer:"

    try:
        #OpenAI to generate a response
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.7
        )
        generated_answer = response.choices[0].text.strip()
        return generated_answer
    except Exception as e:
        return f"Error generating response: {str(e)}"

def process_query(query: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process user query and generate response with sources and follow-up questions."""
    
    # Perform hybrid search
    sources = perform_hybrid_search(query)
    
    # Determine the topic based on the query
    query_lower = query.lower()
    if "resume" in query_lower:
        topic = "resume writing"
    elif "data" in query_lower and ("science" in query_lower or "scientist" in query_lower):
        topic = "data science careers"
    elif "software" in query_lower or "developer" in query_lower:
        topic = "software development"
    elif "trend" in query_lower or "market" in query_lower:
        topic = "career trends"
    else:
        topic = "career development"
    
    #RAG (Retrieval-Augmented Generation)
    if not sources:
        content = ("I don't have specific information about that in my knowledge base. "
                  "However, I can offer some general career advice. Would you like to "
                  "focus on a different career-related topic?")
    else:
        #RAG to generate a response based on query + retrieved context
        content = generate_response_with_rag(query, sources)
    
    # Follow-up questions based on topic
    follow_up_questions = generate_follow_up_questions(topic, sources)
    
    return {
        "content": content,
        "sources": sources,
        "follow_up_questions": follow_up_questions
    }
