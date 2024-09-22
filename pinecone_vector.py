import os
import time
import json
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the Pinecone API key from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")

def initialize_pinecone_index():
    """Initialize Pinecone index."""
    # Initialize Pinecone with the API key
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "youtube-transcripts"

    # Check if the index exists; if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension= 1536,  # Match this with your vector dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        )
        # Wait for the index to be initialized
        time.sleep(1)

    index = pc.Index(index_name)

    return index

def upsert_to_pinecone(index, tokenized_chunks, video_id):
    vectors_with_ids = []
    for i, chunk in enumerate(tokenized_chunks):
        if 'embedding' in chunk and 'metadata' in chunk and 'text' in chunk:  # Ensure 'text' is present
            vectors_with_ids.append((
                f"{video_id}_{i}",
                chunk['embedding'],
                {
                    'text': chunk['text'],
                    **chunk['metadata']
                }
            ))
    if vectors_with_ids:
        index.upsert(vectors_with_ids)
        print(f"Upserted {len(vectors_with_ids)} chunks to Pinecone")
    else:
        print("No valid vectors to upsert.")

def update_video_status(file_path, video_id):
    """Update the video vectorization status in the JSON file."""
    with open(file_path, 'r+') as f:
        video_data = json.load(f)
        for video in video_data:
            if video_id in video['video_url']:
                video['vectorized'] = True
        f.seek(0)
        json.dump(video_data, f, indent=4)
        f.truncate()

def get_relevant_context(index, query_embedding, top_k=5):
    """Fetch the relevant context from Pinecone based on the query embedding."""
    results = index.query(query_embedding, top_k=top_k)
    return results
