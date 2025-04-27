import streamlit as st
from utils import perform_hybrid_search
from chatbot import process_query

st.set_page_config(page_title="Career Advisor Chatbot", layout="wide")
st.title("Personalized Career Advisor Chatbot")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(f"- {source['title']}: {source['snippet']}")

# Chat input
if prompt := st.chat_input("Ask me about career advice..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = process_query(prompt, st.session_state.messages)
            
            st.write(response["content"])
            
            if response["sources"]:
                with st.expander("View Sources"):
                    for source in response["sources"]:
                        st.write(f"- {source['title']}: {source['snippet']}")
            
            # follow-up questions
            if response["follow_up_questions"]:
                st.write("\nFollow-up questions:")
                for question in response["follow_up_questions"]:
                    if st.button(question, key=question):
                        st.session_state.messages.append({
                            "role": "user",
                            "content": question
                        })
                        st.rerun()
    
    # assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["content"],
        "sources": response["sources"]
    })