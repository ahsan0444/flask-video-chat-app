import os
from youtube_transcript_api import YouTubeTranscriptApi
from azure_embeddings import generate_embeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_transcript(video_id):
    """Fetch the transcript for a given YouTube video ID."""
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        print(f"Error fetching transcript for video ID {video_id}: {e}")
        return None

def tokenize_text(transcript, video_url):
    """Tokenize the transcript text into manageable chunks and generate embeddings."""
    
    tokenized_chunks = []
    
    for entry in transcript:
        text_chunk = entry['text']
        embedding = generate_embeddings(text_chunk)  # Use Azure to generate embeddings
        
        tokenized_chunks.append({
            'text': text_chunk,
            'embedding': embedding,
            'metadata': {
                'video_url': video_url,
                'start': entry['start'],
                'duration': entry['duration'],
                'speaker': entry.get('speaker', 'unknown')
            }
        })

    return tokenized_chunks
