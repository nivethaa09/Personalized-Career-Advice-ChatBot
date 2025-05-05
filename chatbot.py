# chatbot.py
import os
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from search import perform_hybrid_search

load_dotenv()

class CareerChatbot:
    def __init__(self):
        self.vector_db = self._setup_vector_db()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.chat_history = []

    def _setup_vector_db(self):
        client = chromadb.Client()
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        return client.get_or_create_collection(name="career_guidance", embedding_function=embedding_fn)
    
    def add_documents(self, documents: list):
        for i, doc in enumerate(documents):
            self.vector_db.add(documents=[doc], ids=[f"doc_{i}"])

    def generate_response(self, user_input: str, relevant_texts: list):
        max_context_chars = 8000  
        context = ""

        for text in relevant_texts:
            if len(context) + len(text) > max_context_chars:
                break
            context += text + "\n\n"

        prompt = f"Context: {context}\n\nUser: {user_input}\nAssistant:"
        chat_response = self.client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = chat_response.choices[0].message.content

        # Dynamically generate a follow-up question
        followup_question = self.generate_dynamic_followup(user_input, relevant_texts)

        # sources for display
        sources = [(f"doc_{i}", 0.0) for i, _ in enumerate(relevant_texts)]

        return response_text, sources, followup_question

    def generate_dynamic_followup(self, user_input, relevant_texts):
        """
        This function dynamically generates a follow-up question based on the user's input
        and the content of the relevant texts retrieved.
        """
        # Default follow-up 
        followup = None
        
        # Check for specific keywords or context within the relevant texts
        if any("resume" in text.lower() for text in relevant_texts):
            followup = "Would you like to learn more about tailoring your resume for specific job roles?"
        elif any("interview" in text.lower() for text in relevant_texts):
            followup = "Would you like advice on common interview questions or strategies to improve your interview skills?"
        elif any("job market" in text.lower() for text in relevant_texts):
            followup = "Would you like insights on the current job market and industries that are hiring?"
        elif "skills" in user_input.lower() or "learning" in user_input.lower():
            followup = "Would you like to explore the top skills needed for your career progression?"
        else:
            followup = "Would you like to dive deeper into related topics or get advice on another career aspect?"

        return followup

    def handle_query(self, query: str, corpus: list):
        top_chunks = perform_hybrid_search(query, corpus)
        return self.generate_response(query, top_chunks)

    def handle_followup(self, followup: str, corpus: list):
        return self.handle_query(followup, corpus)
