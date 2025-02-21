from pydub import AudioSegment
from pydub.utils import which
import openai
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Explicitly set the FFmpeg path
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffmpeg = "D:\\ffmpeg-7.0.2-full_build\\ffmpeg-7.0.2-full_build\\bin\\ffmpeg.exe"  # Change path if installed elsewhere
AudioSegment.ffprobe = "D:\\ffmpeg-7.0.2-full_build\\ffmpeg-7.0.2-full_build\\bin\\ffprobe.exe"  # Change path if needed

def split_audio_by_size(input_file, chunk_size_mb=20):
    """
    Splits an audio file into multiple parts of approximately `chunk_size_mb` MB each.
    Returns a list of chunk file paths.
    """
    audio = AudioSegment.from_file(input_file)
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)

    estimated_duration = len(audio) * (chunk_size_mb / file_size_mb)

    chunks = []
    for i, start_time in enumerate(range(0, len(audio), int(estimated_duration))):
        chunk = audio[start_time:start_time + int(estimated_duration)]
        chunk_filename = f"chunk_{i+1}.mp3"
        chunk.export(chunk_filename, format="mp3")
        chunks.append(chunk_filename)

    return chunks

# Example usage
input_file = "podcast.mp3"  # Your large podcast file
chunk_files = split_audio_by_size(input_file)

def transcribe_audio_chunks(chunk_files):
    """
    Transcribes each chunk separately and merges results.
    """
    full_transcript = ""

    for chunk in chunk_files:
        try:
            with open(chunk, "rb") as audio_file:
                transcription = openai.Audio.transcribe("whisper-1", audio_file)
                full_transcript += transcription["text"] + " "  # Append transcript
                print(f"✅ Transcribed: {chunk}")
        except Exception as e:
            print(f"❌ Error processing {chunk}: {e}")

    return full_transcript.strip()

chunk_files = ["chunk_1.mp3", "chunk_2.mp3", "chunk_3.mp3"]  
final_transcript = transcribe_audio_chunks(chunk_files)
with open("final_transcript.txt", "w", encoding="utf-8") as f:
        f.write(final_transcript)

print("\n✅ Full podcast transcription saved as 'final_transcript.txt'")