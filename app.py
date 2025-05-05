# app.py
import streamlit as st
from chatbot import CareerChatbot
from utils import load_data  

# Streamlit page configuration
st.set_page_config(page_title="Career Advisor", layout="centered")
st.title("🎯 Personalized Career Advice Chatbot")

corpus = load_data()  

# Initialize chatbot and vector DB
if 'chatbot' not in st.session_state:
    st.session_state['chatbot'] = CareerChatbot()
    st.session_state['chatbot'].add_documents(corpus)

# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

# Display conversation
def display_conversation():
    for query, response, sources, followup in st.session_state['conversation_history']:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            st.markdown(response)
            if sources:
                st.markdown("📚 **Sources:**")
                for source, score in sources:
                    st.markdown(f"- `{source}` — Score: `{score:.4f}`")
            if followup:
                st.markdown(f"🤔 **Follow-up Suggestion:** {followup}")

# Input box for user query
user_query = st.chat_input("Ask me anything about your career...")

# Handle new query
if user_query:
    chatbot = st.session_state['chatbot']
    response_text, sources, followup = chatbot.handle_query(user_query, corpus)

    # Store the conversation history
    st.session_state['conversation_history'].append(
        (user_query, response_text, sources, followup)
    )

    # track latest values separately for display
    st.session_state['last_response'] = response_text
    st.session_state['last_sources'] = sources
    st.session_state['last_followup'] = followup
    st.session_state['previous_query'] = user_query

# Display the chat history
display_conversation()
