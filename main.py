import os
import openai
import re
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()   
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

audio_file = open("sample.wav", "rb")
transcription = openai.Audio.transcribe("whisper-1", audio_file)
#print("Transcription:", transcription["text"])

def clean_transcript(transcript):
    """
    Removes filler words, unnecessary speaker tags, and other noise.
    """
    # Remove filler words (extend list if needed)
    filler_words = ["um", "uh", "you know", "like"]
    for filler in filler_words:
        transcript = re.sub(r"\b" + filler + r"\b", "", transcript)

    # Remove extra spaces and line breaks
    transcript = re.sub(r'\s+', ' ', transcript).strip()
    
    return transcript

transcript = clean_transcript(transcription["text"])
print("Cleaned Transcript:", transcript)

def summarize_text(text, summary_type="detailed"):
    """
    Summarizes the given text using GPT-4.
    """
    prompt = f"Summarize the following transcript in a {summary_type} way:\n\n{text}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]

# Example usage
if __name__ == "__main__":
    transcript = transcript.replace("\n", " ").strip() 
    print("🔹 Short Summary:\n", summarize_text(transcript, "short"))
    print("🔹 Detailed Summary:\n", summarize_text(transcript, "detailed"))
    print("🔹 Bullet Points Summary:\n", summarize_text(transcript, "bullet points"))
