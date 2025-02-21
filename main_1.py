import pandas as pd
import os
import openai
import faiss
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv



def get_embedding(text):
    """
    Generates an embedding for the given text using OpenAI API.
    """
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response["data"][0]["embedding"])

def index_transcript(csv_path="podcast_dataset.csv"):
    """
    Converts transcript segments into embeddings and stores in FAISS.
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    segments = df["transcript_segment"].tolist()
    
    # Create FAISS index
    d = 1536  # OpenAI embeddings have 1536 dimensions
    index = faiss.IndexFlatL2(d)  # L2 Distance-based FAISS index

    embeddings = []
    for segment in segments:
        embedding = get_embedding(segment)
        embeddings.append(embedding)

    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, "podcast_index.faiss")
    df.to_csv("indexed_podcast_data.csv", index=False)  # Save indexed data for reference

    print("✅ Transcript embedded & indexed in FAISS.")

if __name__ == "__main__":
    index_transcript()
