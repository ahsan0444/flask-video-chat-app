import os
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

def get_transcript(video_id):
    """Fetch the transcript for a given YouTube video ID."""
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        print(f"Error fetching transcript for video ID {video_id}: {e}")
        return []

def is_relevant_text(text):
    """Check if the transcript text is relevant."""
    return len(text) >= 2 and not (text.strip().startswith('[') and text.strip().endswith(']'))

def tokenize_text(transcript_segments, model_name="all-MiniLM-L6-v2", video_url=None):
    """Tokenize transcript segments using SentenceTransformer."""
    # Initialize the SentenceTransformer embedding model
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
    tokenized_chunks = []

    for segment in transcript_segments:
        text = segment['text']
        if not is_relevant_text(text):
            continue

        # Generate embedding for the text
        embedding = embedding_function.embed(text)

        tokenized_chunks.append({
            'embedding': embedding.tolist(),  # Convert numpy array to list
            'metadata': {
                'video_url': video_url,
                'text': text,
                'start_time': segment['start'],
                'duration': segment['duration']
            }
        })

    return tokenized_chunks

def process_videos_from_file(file_path):
    """Process videos from a JSON file, fetch transcripts, and tokenize."""
    with open(file_path, 'r') as f:
        video_data = json.load(f)

    for video in video_data:
        video_url = video['video_url']
        video_id = video_url.split('v=')[-1]

        if not video['vectorized']:
            transcript = get_transcript(video_id)
        
            if transcript:
                tokenized_chunks = tokenize_text(transcript, video_url=video_url)
                yield video_id, tokenized_chunks
            else:
                print(f"No transcript found for video: {video_url}")
        else:
            print(f"Video already vectorized: {video_url}")
