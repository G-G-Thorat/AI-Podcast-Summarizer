import openai
import faiss
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Load FAISS index & transcript dataset
index = faiss.read_index("podcast_index.faiss")
df = pd.read_csv("indexed_podcast_data.csv")

def query_podcast(question):
    """
    Finds the most relevant podcast segment for a user's query using FAISS.
    """
    question_embedding = np.array(openai.Embedding.create(
        model="text-embedding-ada-002",
        input=question
    )["data"][0]["embedding"]).astype('float32')
    
    # Search FAISS index for the closest match
    D, I = index.search(np.array([question_embedding]), k=1)  # Get top 1 result
    
    return df.iloc[I[0][0]]["transcript_segment"]  # Return the closest transcript

def generate_answer(question):
    """
    Uses GPT-4 to generate an answer based on the retrieved podcast segment.
    """
    relevant_segment = query_podcast(question)
    
    prompt = f"""
    You are an AI assistant answering questions based only on the retrieved transcript below. Do NOT generate information that is not present in the transcript.
    
    Podcast Transcript Segment:
    "{relevant_segment}"
    
    Question: {question}
    
    Answer:
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[{"role": "system", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]

# Example Usage
if __name__ == "__main__":
    user_question = input("Tell me about AI advancements in 2025.\n")
    answer = generate_answer(user_question)
    print("\n🤖 Chatbot Response:\n", answer)
