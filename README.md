# Personalized Career Advice ChatBot

This project implements a **Personalized Career Advice ChatBot** using a **Retrieval-Augmented Generation (RAG)** approach. The chatbot provides personalized career advice based on a curated dataset of career documents (including career tips, industry trends, and roadmap guides). The system integrates both traditional keyword-based search and semantic search to retrieve relevant information and generate accurate responses.

## Dataset Selection

1. **Career Trends 2025** - Insights into the future of the job market, including high-demand careers and emerging fields.
2. **Resume Tips** - Advice on creating an effective resume for job applications.
3. **Career Advice** - Practical steps for advancing in a career.
4. **Career Roadmap** - A comprehensive roadmaps for different career options , covers skills required, tailored roadmap.
    1.**Software Developer Career Path**
    2.**Data Scientist Roadmap**
    3.**Advanced Machine Learning Engineer**
    4.**Computer Vision Engineer**
    5.**Embedded Engineer**
    6.**Cyber Security**

These documents were chosen because they cover a broad spectrum of career-related topics and provide actionable insights for individuals seeking career advice. The selection spans diverse areas such as resume writing, career progression, job market trends, and specific industry paths like data science and software development.

## Implementation

The chatbot uses a **hybrid search** mechanism combining two search methods:

1. **Keyword-based Search**: This search looks for exact matches of terms in the documents. It assigns a score based on the occurrence of query terms and returns the most relevant snippets.
   
2. **Semantic Search**: This search uses **Sentence Transformers** to encode the text into vector embeddings, measuring semantic similarity. It finds the most relevant paragraphs based on meaning, rather than exact keywords.

3. **Hybrid Search**: The results from the keyword-based search and semantic search are combined, with keyword results weighted at 40% and semantic results at 60%. This provides a more accurate and balanced search mechanism, improving the quality of the chatbot's answers.

### Key Components:
- **Sentence Transformers** for semantic search
- **PyPDF2** to extract text from PDF documents
- **ChromaDB** to store and retrieve embeddings

## How to Run the App

### Prerequisites:
- Python 3.12.3
- Install required Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

### Run 
1. Clone this repository:

```bash
git clone https://github.com/nivethaa09/Personalized-Career-Advice-ChatBot.git
cd Personalized-Career-Advice-ChatBot
```

2. Set up your **.env** file with any necessary API keys.
   
3. Start the Streamlit application:

```bash
streamlit run app.py
```

4. navigate to `http://localhost:8501` to interact with the chatbot.





