# ==========================================================
# NLP LEARNING ASSISTANT CHATBOT
# BERT + INTENT DETECTION + ENTITY EXTRACTION + RETRIEVAL
# ==========================================================

import os
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# CONFIG (ENV VARIABLES)
# ==========================================================

MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

# ==========================================================
# KNOWLEDGE BASE
# ==========================================================

knowledge_base = {
    "what is nlp": "Natural Language Processing (NLP) enables computers to understand and process human language.",
    "what is machine learning": "Machine Learning is a subset of Artificial Intelligence that enables systems to learn from data.",
    "what is artificial intelligence": "Artificial Intelligence is the simulation of human intelligence by machines.",
    "what is deep learning": "Deep Learning is a subset of Machine Learning based on neural networks.",
    "what is tokenization": "Tokenization splits text into smaller units called tokens.",
    "what is stemming": "Stemming reduces words to their root form by removing suffixes.",
    "what is lemmatization": "Lemmatization converts words into their dictionary form.",
    "what is stopword removal": "Stopword removal removes frequently occurring words that carry little meaning.",
    "what is tfidf": "TF-IDF measures the importance of a word in a document.",
    "what is bag of words": "Bag of Words converts text into numerical vectors based on word frequency.",
    "what is word embedding": "Word Embeddings represent words as dense numerical vectors.",
    "what is word2vec": "Word2Vec learns vector representations of words.",
    "what is glove": "GloVe is a word embedding model based on word co-occurrence.",
    "what is bert": "BERT is a Transformer-based language model developed by Google.",
    "what is transformer": "Transformer is a neural network architecture based on self-attention.",
    "what is attention": "Attention allows models to focus on important words in a sentence.",
    "what is self attention": "Self-attention allows every word to interact with every other word.",
    "what is rnn": "RNN is a recurrent neural network designed for sequential data.",
    "what is lstm": "LSTM is a special type of RNN that learns long-term dependencies.",
    "what is gru": "GRU is a simplified version of LSTM.",
    "what is named entity recognition": "Named Entity Recognition identifies entities like people, locations and organizations.",
    "what is sentiment analysis": "Sentiment Analysis identifies whether text is positive, negative or neutral.",
    "what is text classification": "Text Classification assigns predefined categories to text.",
    "what is chatbot": "A chatbot is software that simulates human conversation.",
    "what is cosine similarity": "Cosine Similarity measures similarity between two vectors."
}

questions = list(knowledge_base.keys())

# ==========================================================
# LOAD MODEL (CACHED)
# ==========================================================

print("Loading BERT Model...")
model = SentenceTransformer(MODEL_NAME)
question_embeddings = model.encode(questions)
print("Model Loaded Successfully!")

# ==========================================================
# ENTITY LIST
# ==========================================================

entities = [
    "nlp", "machine learning", "artificial intelligence", "deep learning",
    "tokenization", "stemming", "lemmatization", "tfidf", "bag of words",
    "word2vec", "glove", "bert", "transformer", "attention",
    "self attention", "rnn", "lstm", "gru", "chatbot",
    "sentiment analysis", "named entity recognition"
]

# ==========================================================
# ENTITY EXTRACTION
# ==========================================================

def extract_entities(text):
    text = text.lower()
    return [e for e in entities if e in text]

# ==========================================================
# CHATBOT FUNCTION
# ==========================================================

def chatbot(query):
    if not query.strip():
        return "None", "None", "0.00", "Please enter a question."

    query_embedding = model.encode([query])

    similarity = cosine_similarity(query_embedding, question_embeddings)

    best_index = np.argmax(similarity)
    confidence = float(similarity[0][best_index])

    intent = questions[best_index]
    extracted = extract_entities(query)

    if confidence < 0.40:
        return (
            "Unknown",
            ", ".join(extracted) if extracted else "None",
            f"{confidence:.2f}",
            "Sorry! I couldn't understand your question."
        )

    return (
        intent,
        ", ".join(extracted) if extracted else "None",
        f"{confidence:.2f}",
        knowledge_base[intent]
    )

# ==========================================================
# GRADIO UI
# ==========================================================

demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask any NLP-related question...",
        label="Your Question"
    ),
    outputs=[
        gr.Textbox(label="Detected Intent"),
        gr.Textbox(label="Extracted Entities"),
        gr.Textbox(label="Confidence Score"),
        gr.Textbox(label="Bot Response")
    ],
    title="🤖 NLP Learning Assistant",
    description="""
✅ Intent Detection  
✅ Entity Extraction  
✅ BERT Embeddings  
✅ Retrieval-Based Response  
""",
    examples=[
        ["What is NLP?"],
        ["Explain BERT"],
        ["Tell me about LSTM"],
        ["Define TF-IDF"],
        ["What is Cosine Similarity?"]
    ]
)

# ==========================================================
# LAUNCH
# ==========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
